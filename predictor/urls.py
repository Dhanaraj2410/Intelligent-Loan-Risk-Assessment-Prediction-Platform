# =============================================================================
# urls.py - URL Routing Configuration for the Predictor App
# =============================================================================
# Maps URL patterns to their corresponding view functions.
# All URLs are namespaced under 'predictor' to avoid conflicts with other apps.
# =============================================================================

from django.urls import path
from . import views

# Namespace for reverse URL lookups (e.g., 'predictor:predict', 'predictor:result')
app_name = 'predictor'

urlpatterns = [
    # --- Web Page Routes ---
    path('', views.landing, name='index'),                                      # Landing/home page
    path('dashboard/', views.dashboard, name='dashboard'),                      # Analytics dashboard with charts
    path('predict/', views.predict, name='predict'),                            # Loan prediction form (GET/POST)
    path('result/<int:prediction_id>/', views.result, name='result'),            # Prediction result detail page
    path('history/', views.history, name='history'),                            # All past predictions list
    path('history/<int:prediction_id>/', views.history_detail, name='history_detail'),  # Single history record detail
    path('models/', views.model_comparison, name='model_comparison'),           # ML vs DL model comparison page

    # --- REST API Endpoints ---
    path('api/predict/', views.api_predict, name='api_predict'),                # JSON API for programmatic predictions
    path('api/model-info/', views.model_info, name='model_info'),              # JSON API for model metadata
    path('feedback/', views.feedback, name='feedback'),                        # User feedback submission endpoint
]