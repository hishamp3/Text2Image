from django.urls import path
from . import views

urlpatterns = [
    path('', views.retrieval, name="retrieval"),
]
