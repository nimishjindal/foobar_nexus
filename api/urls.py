from django.urls import path
from .views import process_prompt_api

urlpatterns = [
    path('process_prompt/', process_prompt_api, name='process_prompt'),
]
