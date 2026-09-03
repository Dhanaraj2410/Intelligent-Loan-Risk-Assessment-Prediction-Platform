# =============================================================================
# test_api.py - Integration Tests for Web Routes and REST API Endpoints
# =============================================================================
# Tests HTTP response codes for all web pages (landing, predict, dashboard)
# and validates the JSON prediction API returns correct response structure.
# =============================================================================

from django.test import TestCase, RequestFactory
from django.urls import reverse
import json
from predictor import views


class PredictionPipelineAndAPITestCase(TestCase):
    """Integration tests for web routes and the /api/predict/ REST endpoint."""

    def setUp(self):
        """Initialize Django RequestFactory for simulating HTTP requests."""
        self.factory = RequestFactory()

    def test_web_routes(self):
        """Verify all web page routes return HTTP 200 OK status codes."""
        # Landing page
        request = self.factory.get('/')
        response = views.landing(request)
        self.assertEqual(response.status_code, 200)

        # Predict form page
        request = self.factory.get('/predict/')
        response = views.predict(request)
        self.assertEqual(response.status_code, 200)

        # Model comparison dashboard
        request = self.factory.get('/model-comparison/')
        response = views.model_comparison(request)
        self.assertEqual(response.status_code, 200)

        # History log page
        request = self.factory.get('/history/')
        response = views.history(request)
        self.assertEqual(response.status_code, 200)

    def test_rest_api_prediction(self):
        """Verify the REST API returns valid JSON with prediction results."""
        # Simulate a JSON POST request with sample applicant data
        payload = {
            "gender": "Male",
            "married": "Yes",
            "dependents": "1",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 5849,
            "coapplicant_income": 0,
            "loan_amount": 130,
            "loan_amount_term": 360,
            "credit_history": 1.0,
            "property_area": "Urban"
        }
        request = self.factory.post(
            '/api/predict/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        response = views.api_predict(request)

        # Validate response structure and value ranges
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertIn(data['prediction'], ['Approved', 'Rejected'])
        self.assertGreaterEqual(data['approval_probability'], 0.0)
        self.assertLessEqual(data['approval_probability'], 100.0)
        self.assertIn(data['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])
