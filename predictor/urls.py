from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('result/<int:prediction_id>/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/model-info/', views.model_info, name='model_info'),
    path('feedback/', views.feedback, name='feedback'),
   # path('test-prediction/', views.test_prediction, name='test_prediction'),  # Debug endpoint
]