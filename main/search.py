from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='wikIR1k',
                          postings_encoding=VBEPostings,
                          output_dir='index')


def query_auto_completion(query):
    """
    Get auto-completion suggestions for the input query.
    """

    completions = [query]
    completions += (BSBI_instance.get_query_recommendations(query))
    for i, completion in enumerate(completions, start=1):
        if i == 1:
            print(f"{i}. {query}")
            completions[i-1] = query
        else:
            print(f"{i}. {query}{completion}")
            completions[i-1] = query + completion
    return completions

def select_query(completions):
    """
    Ask the user to select a query from the auto-completion list.
    """
    selection = int(input("\nMasukkan nomor query yang Anda maksud: "))
    selected_query = completions[selection - 1]
    print(f"\nPilihan Anda adalah '{selected_query}'.\n")
    return selected_query

def perform_search(query):
    """
    Perform a search using the selected query and display the results.
    """


    print("Hasil pencarian:")
    
    results = BSBI_instance.retrieve_bm25_taat(query, k=20, k1=1.065, b=0)
    #results = BSBI_instance.retrieve_tfidf_taat(query, k=10)
    #results = BSBI_instance.retrieve_tfidf_daat(query, k=20)
    for score, doc in results:
        print(f"{doc} \t\t {score}")





if __name__ == "__main__":

    import os
    import sys

    # Add the root directory (above 'main') to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Set the DJANGO_SETTINGS_MODULE environment variable
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'search_engine.settings')
    
    user_query = input("Masukkan query Anda: ")
    query_completions = query_auto_completion(user_query)
    selected_query = select_query(query_completions)
    perform_search(selected_query)