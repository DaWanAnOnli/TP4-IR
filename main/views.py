from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def results(request):
    query = request.GET.get('q', '')  # Get the query parameter
    context = {'query': query}
    return render(request, 'results.html', context)