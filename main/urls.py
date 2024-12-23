from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.home, name='home'),
    path('results/', views.results, name='results'),
    path('autocomplete/', views.autocomplete, name='autocomplete'),
]