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




def home(request):
    return render(request, 'home.html')

def results(request):
    query = request.GET.get('q', '')  # Get the query parameter

    # Perform the search
    all_results = perform_search(query)

    # Paginate results: 10 per page
    paginator = Paginator(all_results, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'query': query,
        'page_obj': page_obj,  # Paginated results
    }
    return render(request, 'results.html', context)



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
    Perform a search for the given query.
    """
    results_data = {
        "search engine": [
            f"Document {i + 1}" for i in range(50)
        ],  # Simulating 50 results
        "search algorithm": [
            f"Document {i + 1}" for i in range(35)
        ],
    }
    return results_data.get(query, ["No results found."])