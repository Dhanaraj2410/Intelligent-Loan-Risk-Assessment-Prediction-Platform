# =============================================================================
# test_models.py - Unit Tests for LoanPrediction Django ORM Model
# =============================================================================
# Validates database record creation, field constraints, and the
# get_statistics() aggregate method using Django's test framework.
# =============================================================================

from django.test import TestCase
from predictor.models import LoanPrediction, UserFeedback


class LoanPredictionModelTestCase(TestCase):
    """Test suite for the LoanPrediction model CRUD operations and statistics."""

    def setUp(self):
        """Create a sample prediction record for use across all test methods."""
        self.prediction = LoanPrediction.objects.create(
            gender='Male',
            married='Yes',
            dependents='1',
            education='Graduate',
            self_employed='No',
            applicant_income=5000,
            coapplicant_income=1500,
            loan_amount=150,
            loan_amount_term=360,
            credit_history=1.0,
            property_area='Urban',
            prediction='Approved',
            probability=88.5,
            approval_probability=88.5,
            risk_score=11.5,
            risk_level='LOW',
            model_name='XGBoost'
        )

    def test_prediction_creation(self):
        """Verify that a prediction record is saved correctly with valid field values."""
        self.assertEqual(self.prediction.prediction, 'Approved')
        self.assertEqual(self.prediction.risk_level, 'LOW')
        self.assertGreaterEqual(self.prediction.approval_probability, 0.0)
        self.assertLessEqual(self.prediction.approval_probability, 100.0)

    def test_statistics_helper(self):
        """Verify the get_statistics() class method computes correct aggregates."""
        stats = LoanPrediction.get_statistics()
        self.assertEqual(stats['total'], 1)
        self.assertEqual(stats['approved'], 1)
        self.assertEqual(stats['approval_rate'], 100.0)
