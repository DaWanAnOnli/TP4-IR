from django.conf import settings
from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import JsonResponse

from .bsbi import BSBIIndex
from .compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='wikIR1k',
                          postings_encoding=VBEPostings,
                          output_dir='index')
BSBI_instance.do_indexing()
print("done indexing")





def home(request):
    
    return render(request, 'home.html')

def perform_search(query):
    print("Hasil pencarian:")
    # Retrieve and limit to 100
    results = BSBI_instance.retrieve_bm25_taat(query, k=100, k1=1.065, b=0)
    # for score, doc in results:
    #     print(f"{doc} \t\t {score}")
    return results  # Return actual results

def results(request):
    query = request.GET.get('q', '')
    search_results = perform_search(query)  # Get the results
    
    # Convert tuples into list of dictionaries while preserving the pairs
    all_results = [{'score': score, 'id': doc_id} for score, doc_id in search_results]

    # Paginate
    paginator = Paginator(all_results, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    context = {
        'query': query,
        'page_obj': page_obj,
    }
    return render(request, 'results.html', context)


import csv
import os

def doc_detail(request, doc_id):
    if doc_id:
        # Read from CSV file
        csv_path = os.path.join(settings.BASE_DIR, 'main', 'wikIR1k', 'documents-trimmed.csv')
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row['id_right'] == str(doc_id):
                    doc_info = {'id': row['id_right'], 'text': row['text_right']}

                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    print(doc_info)
                    return render(request, 'doc_detail.html', {'doc': doc_info})

    return render(request, 'doc_detail.html', {'error': 'Document not found'})



def autocomplete(request):
    """
    Handle auto-complete queries by using query_auto_completion.
    """
    query = request.GET.get('q', '').strip()
    if len(query) < 3:
        return JsonResponse({'suggestions': []})  # Return empty if less than 3 chars

    # Use the hardcoded query_auto_completion function
    suggestions = query_auto_completion(query)

    return JsonResponse({'suggestions': suggestions})


#TODO: masih salah
def query_auto_completion(query):
    """
    Get auto-completion suggestions for the input query.
    """

    completions = [query]
    print("!a")
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

