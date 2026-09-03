from django.test import TestCase
from predictor.forms import LoanPredictionForm


class LoanPredictionFormTestCase(TestCase):
    def test_valid_form_data(self):
        form_data = {
            'gender': 'Male',
            'married': 'Yes',
            'dependents': '0',
            'education': 'Graduate',
            'self_employed': 'No',
            'applicant_income': 6000,
            'coapplicant_income': 0,
            'loan_amount': 120,
            'loan_amount_term': 360,
            'interest_rate': 10.5,
            'credit_history': 1.0,
            'property_area': 'Urban'
        }
        form = LoanPredictionForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_invalid_form_data_negative_income(self):
        form_data = {
            'gender': 'Male',
            'married': 'Yes',
            'dependents': '0',
            'education': 'Graduate',
            'self_employed': 'No',
            'applicant_income': -500,
            'coapplicant_income': 0,
            'loan_amount': 120,
            'loan_amount_term': 360,
            'credit_history': 1.0,
            'property_area': 'Urban'
        }
        form = LoanPredictionForm(data=form_data)
        self.assertFalse(form.is_valid())
