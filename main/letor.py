import os
import numpy as np
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from .bsbi import BSBIIndex
from .compression import VBEPostings
import csv
from tqdm import tqdm
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from nltk.util import ngrams
from transformers import AutoTokenizer, AutoModel
import torch
from django.conf import settings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


# Initialize BSBIIndex instance
BSBI_instance = BSBIIndex(data_dir='wikIR1k',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries_dict = {}
document_mapping = {}

def load_qrels(qrels_file):
    """Load qrels file into a dictionary."""
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, iter, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)
    return qrels

# TF-IDF Vectorizer (shared instance to avoid reinitialization)
tfidf_vectorizer = TfidfVectorizer()

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bertmodel = AutoModel.from_pretrained("bert-base-uncased")

# Helper function: Embed text using BERT
def bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bertmodel(**inputs)
    # Use the CLS token's embedding as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def bert_cosine_similarity(doc, query):
    doc_embedding = bert_embedding(doc)
    query_embedding = bert_embedding(query)
    return cosine_similarity([doc_embedding], [query_embedding])[0, 0]

def extract_features(query, top_docs, BSBI_instance):
    """Extract features for query-document pairs."""
    if isinstance(query, int):
        query = queries_dict[str(query)]
    print("query = ", query)
    csv_path = os.path.join(settings.BASE_DIR, "main", "wikIR1k", "documents-trimmed.csv")
    features = []
    for doc_id in top_docs:

        # Retrieve the BM25 and TF-IDF scores
        bm25_score = BSBI_instance.retrieve_bm25_taat_letor(query, doc_id)
        tfidf_score = BSBI_instance.retrieve_tfidf_taat_letor(query, doc_id)

        # Read the document content
        text = BSBI_instance.get_text_by_doc_id(doc_id, csv_path)

        # Calculate BERT cosine similarity
        bert_similarity = bert_cosine_similarity(text, query)

        
        

        # Append all features
        features.append([bm25_score, tfidf_score, bert_similarity])

    return features




def prepare_training_data(qrels, BSBI_instance):
    """Prepare training data for LambdaMART."""
    features = []
    labels = []
    group_sizes = []

    for query_id, docs in tqdm(qrels.items(), desc="Processing queries"):
        print(query_id, docs)
        top_docs = list(docs.keys())
        query_features = extract_features(int(query_id), top_docs, BSBI_instance)
        query_labels = [docs[doc_id] for doc_id in top_docs]
        
        features.extend(query_features)
        labels.extend(query_labels)
        group_sizes.append(len(query_features))
    
    return features, labels, group_sizes

def train_lambda_mart(features, labels, group_sizes, test_size=0.2, random_state=42):
    """
    Train an XGBoost ranking model.

    Args:
        features: 2D array-like, feature matrix.
        labels: 1D array-like, relevance labels.
        group_sizes: 1D array-like, sizes of query groups.
        test_size: Fraction of data to use as validation (default=0.2).
        random_state: Random seed for reproducibility (default=42).

    Returns:
        model: Trained XGBoost model.
    """
    # Convert inputs to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)
    group_sizes = np.array(group_sizes)

    # Verify consistency
    assert sum(group_sizes) == len(features) == len(labels), "Mismatch between group sizes and number of samples."
    assert all(group_sizes > 0), "All group sizes must be positive integers."

    # Generate cumulative indices for splitting groups
    group_cumsum = np.cumsum(group_sizes)
    group_indices = np.arange(len(group_sizes))

    # Split groups into training and validation sets
    train_indices, val_indices = train_test_split(
        group_indices, test_size=test_size, random_state=random_state
    )

    # Create masks for training and validation sets
    train_mask = np.isin(np.arange(len(features)), np.concatenate([range(group_cumsum[i] - group_sizes[i], group_cumsum[i]) for i in train_indices]))
    val_mask = ~train_mask

    # Create subsets
    train_features, val_features = features[train_mask], features[val_mask]
    train_labels, val_labels = labels[train_mask], labels[val_mask]

    train_group_sizes = group_sizes[train_indices]
    val_group_sizes = group_sizes[val_indices]

    # Create DMatrix objects for XGBoost
    train_data = xgb.DMatrix(train_features, label=train_labels)
    val_data = xgb.DMatrix(val_features, label=val_labels)

    # Add group information
    train_data.set_group(train_group_sizes)
    val_data.set_group(val_group_sizes)

    # Parameters
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'eta': 0.1,
        'max_depth': 6,
    }

    # Train the model
    model = xgb.train(
        params,
        train_data,
        num_boost_round=100,
        evals=[(train_data, 'train'), (val_data, 'valid')],
        early_stopping_rounds=10
    )

    return model

def rerank(query, top_docs, model, BSBI_instance):
    """
    Rerank top documents using the trained XGBoost ranking model.

    Args:
        query: The query string.
        top_docs: List of top document IDs retrieved from the initial ranking.
        model: Trained XGBoost ranking model.
        BSBI_instance: An instance of the BSBIIndex class to extract features.

    Returns:
        reranked_docs: List of document IDs sorted by reranked scores.
    """
    # Extract features for the given query and top documents
    features = extract_features(query, top_docs, BSBI_instance)

    # Convert features to a DMatrix for XGBoost
    features_dmatrix = xgb.DMatrix(features)

    # Predict scores using the trained model
    scores = model.predict(features_dmatrix)

    # Sort documents by their predicted scores (descending order)
    reranked_docs = sorted(zip(scores, top_docs), reverse=True, key=lambda x: x[0])
    return reranked_docs


def perform_search(query, model=None):
    """
    Perform a search using the selected query and optionally rerank the results.
    """
    print("Hasil pencarian (sebelum reranking):")
    initial_results = BSBI_instance.retrieve_bm25_taat(query, k=20, k1=1.065, b=0)
    for score, doc in initial_results:
        print(f"{doc} \t\t {score}")

    if model:
        top_docs = [doc for _, doc in initial_results]
        reranked_results = rerank(query, top_docs, model, BSBI_instance)
        print("\nHasil pencarian (setelah reranking):")
        for score, doc in reranked_results:
            print(f"{doc} \t\t {score}")

def get_query_text(query_file):

    # Open the CSV file and read its contents
    with open(query_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # Read as a dictionary
        for row in csv_reader:
            # Assign id_left as the key and text_left as the value
            queries_dict[row['id_left']] = row['text_left']



def create_doc_id_mappings(filename):
    # Read the original CSV file
    df = pd.read_csv(filename)

    # Create a new column that maps each id_right to an integer starting from zero
    df['id_mapped'] = range(len(df))

    # Save the modified DataFrame to a new CSV file
    df[['id_right', 'id_mapped']].to_csv(f'{filename[:-4]}-mapping.csv', index=False)

def get_document_mapping(document_mapping_file):

    # Open the CSV file and read its contents
    with open(document_mapping_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # Read as a dictionary
        for row in csv_reader:
            # Assign id_left as the key and text_left as the value
            document_mapping[row['id_right']] = row['id_mapped']


if __name__ == "__main__":
    pass
    create_doc_id_mappings('wikIR1k/documents-trimmed.csv')

    BSBI_instance.load()
    # Load qrels and prepare training data
    qrels_file = 'wikIR1k/training/qrels_trimmed'
    qrels = load_qrels(qrels_file)

    get_query_text('wikIR1k/training/queries.csv')
    get_document_mapping('wikIR1k/documents-trimmed-mapping.csv')
    features, labels, group_sizes = prepare_training_data(qrels, BSBI_instance)
    joblib.dump(features, 'features.joblib')
    joblib.dump(labels, 'labels.joblib')
    joblib.dump(group_sizes, 'group_sizes.joblib')

    # features = joblib.load('features.joblib')
    # labels = joblib.load('labels.joblib')
    # group_sizes = joblib.load('group_sizes.joblib')

    print(type(features))
    print(features[:100])
    print(type(labels))
    print(type(group_sizes))

    for i, row in enumerate(features):
        if len(row) != 2:
            print(f"Row {i}: {len(row)}")
    max_length = max(len(row) for row in features)
    print(max_length)
    print("features 1 :", features[:1])
    print("label 1 :", labels[:1])


    # Train LambdaMART model
    print("Training LambdaMART model...")
    model = train_lambda_mart(features, labels, group_sizes)
    joblib.dump(model, 'model.joblib')
    print("Training completed.")

    # Perform search with reranking
    user_query = input("Masukkan query Anda: ")
    perform_search(user_query, model=model)