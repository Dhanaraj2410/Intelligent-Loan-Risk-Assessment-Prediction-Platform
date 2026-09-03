# =============================================================================
# apps.py - Django Application Configuration for Predictor App
# =============================================================================
# Registers the 'predictor' app with Django and sets display name for
# the admin panel. The verbose_name appears in Django Admin sidebar.
# =============================================================================

from django.apps import AppConfig


class PredictorConfig(AppConfig):
    """Django application configuration for the Loan Prediction module.
    
    This app handles all loan risk assessment functionality including:
    - Multi-model ML/DL inference pipeline
    - Loan prediction form processing and validation
    - REST API endpoints for programmatic predictions
    - Historical prediction data management
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'
    verbose_name = 'Loan Prediction'