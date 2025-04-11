from django.urls import path
from .views import reduce, cool, predict

urlpatterns = [
    path('reduce', reduce, name='reduce'),
    path('cool', cool, name='cool'),
    path('predict', predict, name='predict'),
]