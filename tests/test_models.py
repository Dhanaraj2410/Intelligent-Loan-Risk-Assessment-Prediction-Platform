from django.test import TestCase
from predictor.models import LoanPrediction, UserFeedback


class LoanPredictionModelTestCase(TestCase):
    def setUp(self):
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
        self.assertEqual(self.prediction.prediction, 'Approved')
        self.assertEqual(self.prediction.risk_level, 'LOW')
        self.assertGreaterEqual(self.prediction.approval_probability, 0.0)
        self.assertLessEqual(self.prediction.approval_probability, 100.0)

    def test_statistics_helper(self):
        stats = LoanPrediction.get_statistics()
        self.assertEqual(stats['total'], 1)
        self.assertEqual(stats['approved'], 1)
        self.assertEqual(stats['approval_rate'], 100.0)
