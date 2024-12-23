from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import JsonResponse



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


#TODO: diganti ama query_auto_complete beneren
def query_auto_completion(query):
    """
    Hardcoded query auto-completion logic.
    """
    # Hardcoded recommendations based on the query
    all_suggestions = {
        "search": ["engine", "algorithm", "system", "query", "results"],
        "sea": ["search", "seal", "season", "seashore", "seattle"],
        "sys": ["system", "syntax", "synthesis", "synergy", "systematic"],
    }

    # Use the first 3 letters as a key to find suggestions
    key = query[:3].lower()
    return all_suggestions.get(key, [])[:5] 

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