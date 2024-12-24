import csv
import os
import pickle
import contextlib
import heapq
import math
import re
from porter2stemmer import Porter2Stemmer
#import requests
import string

from nltk.corpus import stopwords
from main.index import InvertedIndexReader, InvertedIndexWriter
from main.trie import Trie
from main.util import IdMap, merge_and_sort_posts_and_tfs
from main.compression import VBEPostings

# from index import InvertedIndexReader, InvertedIndexWriter
# from trie import Trie
# from util import IdMap, merge_and_sort_posts_and_tfs
# from compression import VBEPostings

from tqdm import tqdm
from django.conf import settings


import os
import sys

# Add the root directory (above 'main') to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'search_engine.settings')

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    trie(Trie): Class Trie untuk query auto-completion
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.trie = Trie()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        self.index_file_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'main_index.index')
        try:
            # self.index_file = open(self.index_file_path, 'rb+')
            pass
        except FileNotFoundError:
            # Handle the error: log it, create the file, etc.
            print(f"Index file not found at {self.index_file_path}")
            # Optionally, create the file or raise an error
            # open(self.index_file_path, 'wb').close()
            raise

    def save(self):
        try:
            terms_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'terms.dict')
            docs_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'docs.dict')
            trie_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'trie.pkl')

            # with open(terms_path, 'wb') as f:
            #     pickle.dump(self.term_id_map, f)
            # with open(docs_path, 'wb') as f:
            #     pickle.dump(self.doc_id_map, f)
            # with open(trie_path, 'wb') as f:
            #     pickle.dump(self.trie, f)
        except Exception as e:
            print(f"Error saving files: {e}")
            # Handle the error appropriately

    def load(self):
        try:
            terms_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'terms.dict')
            docs_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'docs.dict')
            trie_path = os.path.join(settings.BASE_DIR, 'main', 'index', 'trie.pkl')
            print(trie_path)
            print("c")
            with open(terms_path, 'rb') as f:
                print("d")
                self.term_id_map = pickle.load(f)
            
            with open(docs_path, 'rb') as f:
                self.doc_id_map = pickle.load(f)
            with open(trie_path, 'rb') as f:
                self.trie = pickle.load(f)
            
        except FileNotFoundError as e:
            print(f"File not found during load: {e}")
            # Handle the error appropriately
        except Exception as e:
            print(f"Error loading files: {e}")
            # Handle the error appropriately

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        stop_words = set(stopwords.words('english'))
        stemmer = Porter2Stemmer()

        td_pairs = []
        block_full_path = os.path.join(self.data_dir, block_path)
        
        for doc in sorted(os.listdir(block_full_path)):
            with open(os.path.join(block_full_path, doc), 'r') as f:
                doc_content = f.read().lower() 
                tokens = re.findall(r'\w+', doc_content)
                
                doc_id = self.doc_id_map[os.path.join(block_path, doc)]
                
                for token in tokens:
                    if token not in stop_words: 
                        self.trie.insert(token, 1)
                        stemmed_token = stemmer.stem(token)
                        term_id = self.term_id_map[stemmed_token]
                        td_pairs.append((term_id, doc_id))
                        
        
        return td_pairs

    def parsing_csv(self, csv_path):
        """
        Parses a CSV file with columns 'id_right' and 'text_right', converting its content
        to a sequence of <termID, docID> pairs.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file to be parsed.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the CSV file.
        """
        result = []
        stemmer = Porter2Stemmer()
        stop_words = set(stopwords.words('english'))

        tokenizer_pattern = r"""\b\w+(?:-\w+)*(?:'s|'ll|'ve|'d|'re|'m|'t)?\b"""

        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                doc_id = self.doc_id_map[row['id_right']]
                text = row['text_right']

                # Tokenization and stopword removal
                tokens = re.findall(tokenizer_pattern, text)
                raw_tokens = [token for token in tokens if token.lower() not in stop_words]

                # Insert raw tokens into trie before pre-processing
                for token in raw_tokens:
                    self.trie.insert(token, 1)

                # Stemming and token pre-processing
                stemmed_tokens = [stemmer.stem(token) for token in raw_tokens if token]

                for token in stemmed_tokens:
                    term_id = self.term_id_map[token]
                    result.append((term_id, doc_id))

        return result

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # term_dict merupakan dictionary yang berisi dictionary yang
        # melakukan mapping dari doc_id ke tf
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            # Mengupdate juga TF (yang merupakan value dari dictionary yang di dalam)
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            # Sort postings list (dan tf list yang bersesuaian)
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            # Postings list adalah keys, TF list adalah values
            index.append(term_id, list(sorted_postings_tf.keys()), 
                         list(sorted_postings_tf.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)


    def _compute_score_tfidf(self, tf, df, N):
        """
        Fungsi ini melakukan komputasi skor TF-IDF.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        score = w(t, Q) x w(t, D)
        Tidak perlu lakukan normalisasi pada score.

        Gunakan log basis 10.

        Parameters
        ----------
        tf: int
            Term frequency.

        df: int
            Document frequency.

        N: int
            Jumlah dokumen di corpus. 

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        if tf > 0:
            w_t_D = 1 + math.log10(tf)
            w_t_Q = math.log10(N / df)
            return w_t_D * w_t_Q
        else:
            return 0.
    
    def _compute_score_bm25(self, tf, df, N, k1, b, dl, avdl):
        """
        Fungsi ini melakukan komputasi skor BM25.
        Gunakan log basis 10 dan tidak perlu lakukan normalisasi.
        Silakan lihat penjelasan parameters di slide.

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """

        idf = math.log10((N) / (df))
        term_weight = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avdl)) + tf)
        return idf * term_weight
    

    def query_to_token(self, query):
        stemmer = Porter2Stemmer()
        stop_words = set(stopwords.words('english'))
        tokens = re.findall(r'\w+', query)

        query_terms = []
        for token in tokens:
            if token not in stop_words:
                stemmed_token = stemmer.stem(token)
                query_terms.append(stemmed_token)
        return query_terms


    def retrieve_tfidf_daat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema DaaT.
        Method akan mengembalikan top-K retrieval results.

        Program tidak perlu paralel sepenuhnya. Untuk mengecek dan mengevaluasi
        dokumen yang di-point oleh pointer pada waktu tertentu dapat dilakukan
        secara sekuensial, i.e., seperti menggunakan for loop.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
   
        query_terms = self.query_to_token(query)
        postings_lists = []
        term_data = []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)

            # Initialisasi postings lists
            for term in query_terms:
                term_id = self.term_id_map[term]
                if term_id is not None:
                    postings_list, tf_list = index.get_postings_list(term_id)
                    postings_lists.append(list(zip(postings_list, tf_list)))
                    term_data.append((term_id, len(postings_list)))

            if not postings_lists:
                return []

            # Initialisasi pointer untuk tiap term
            pointers = [0] * len(postings_lists)
            score_docs = {}

            while True:
                # List Dokumen id yang dipoint oleh semua pointer
                current_doc_ids = [
                    postings_lists[i][pointers[i]][0] if pointers[i] < len(postings_lists[i]) else float('inf')
                    for i in range(len(postings_lists))
                ]

                # Cari doc id minimum
                min_doc_id = min(current_doc_ids)
                if min_doc_id == float('inf'):
                    break

                # Kalkulasi score
                score = 0
                for i in range(len(postings_lists)):
                    if current_doc_ids[i] == min_doc_id:
                        tf = postings_lists[i][pointers[i]][1]
                        df = term_data[i][1]
                        score += self._compute_score_tfidf(tf, df, N)
                        pointers[i] += 1

                score_docs[min_doc_id] = score

            return self.get_top_k_by_score(score_docs, k)



    def retrieve_tfidf_taat_letor(self, query, doc_id, k=10):

        query_terms = self.query_to_token(query)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            score = 0

            for term in query_terms:
                term_id = self.term_id_map[term]
                postings_list, tf_list = index.get_postings_list(term_id)
                df = len(postings_list)

                for i, id in enumerate(postings_list):
                    tf = tf_list[i]
                    if int(id) == int(doc_id):
                        score += self._compute_score_tfidf(tf, df, N)
                        break

        return score




    def retrieve_tfidf_taat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        query_terms = self.query_to_token(query)
        score_docs = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)

            for term in query_terms:
                term_id = self.term_id_map[term]
                postings_list, tf_list = index.get_postings_list(term_id)
                df = len(postings_list)

                for i, doc_id in enumerate(postings_list):
                    tf = tf_list[i]
                    score = self._compute_score_tfidf(tf, df, N)
                    if doc_id in score_docs:
                        score_docs[doc_id] += score
                    else:
                        score_docs[doc_id] = score

        return self.get_top_k_by_score(score_docs, k)


    def retrieve_bm25_taat_letor(self, query, doc_id, k1=1.2, b=0.75):
        query_terms = self.query_to_token(query)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            score = 0
            avdl = index.get_average_document_length()

            for term in query_terms:
                term_id = self.term_id_map[term]
                postings_list, tf_list = index.get_postings_list(term_id)
                df = len(postings_list)

                for i, id in enumerate(postings_list):
                    if int(id) == int(doc_id):
                        tf = tf_list[i]
                        dl = index.doc_length[id]
                        score += self._compute_score_bm25(tf, df, N, k1, b, dl, avdl)
                        break
        return score



    def retrieve_bm25_taat(self, query, k=10, k1=1.2, b=0.75):
        """
        Lakukan retrieval BM-25 dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        query_terms = self.query_to_token(query)
        score_docs = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            avdl = index.get_average_document_length()

            for term in query_terms:
                term_id = self.term_id_map[term]
                postings_list, tf_list = index.get_postings_list(term_id)
                df = len(postings_list)

                for i, doc_id in enumerate(postings_list):
                    tf = tf_list[i]
                    dl = index.doc_length[doc_id]
                    score = self._compute_score_bm25(tf, df, N, k1, b, dl, avdl)
                    if doc_id in score_docs:
                        score_docs[doc_id] += score
                    else:
                        score_docs[doc_id] = score

        return self.get_top_k_by_score(score_docs, k)


    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of bsbi.py
        csv_path = os.path.join(base_dir, self.data_dir, 'documents-trimmed.csv')
        td_pairs = self.parsing_csv(csv_path)
        index_id = "intermediate_index"
        self.intermediate_indices.append(index_id)

        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
            self.write_to_index(td_pairs, index)

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)
    def get_top_k_by_score(self, score_docs, k):
        """
        Method ini berfungsi untuk melakukan sorting terhadap dokumen berdasarkan score
        yang dihitung, lalu mengembalikan top-k dokumen tersebut dalam bentuk tuple
        (score, document). Silakan gunakan heap agar lebih efisien.

        Parameters
        ----------
        score_docs: Dictionary[int -> float]
            Dictionary yang berisi mapping docID ke score masing-masing dokumen tersebut.

        k: Int
            Jumlah dokumen yang ingin di-retrieve berdasarkan score-nya.

        Result
        -------
        List[(float, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        return heapq.nlargest(k, [(score, self.doc_id_map[doc_id]) for doc_id, score in score_docs.items()])

    
    def get_query_recommendations(self, query, k=5):
        print("!b")
        # Method untuk mendapatkan rekomendasi untuk QAC
        # Tidak perlu mengubah ini
        # self.load()
        last_token = query.split()[-1]
        
        recc = self.trie.get_recommendations(last_token, k)
        return recc

if __name__ == "__main__":


    import os
    import sys

    # Add the root directory (above 'main') to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Set the DJANGO_SETTINGS_MODULE environment variable
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'search_engine.settings')

    # Initialize Django
    import django
    django.setup()

    BSBI_instance = BSBIIndex(data_dir='wikIR1k',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    # BSBI_instance.do_indexing()  # memulai indexing!