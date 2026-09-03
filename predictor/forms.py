# =============================================================================
# forms.py - Django ModelForm for Loan Prediction Input Validation
# =============================================================================
# Handles form rendering with Bootstrap-styled widgets, input validation,
# type coercion, and cross-field validation for the loan application form.
# =============================================================================

from django import forms
from .models import LoanPrediction

class LoanPredictionForm(forms.ModelForm):
    CREDIT_HISTORY_CHOICES = [
        (1.0, 'Good Credit Record (No Defaults / CIBIL 750+)'),
        (0.0, 'Poor Credit Record (Past Defaults / Low CIBIL)'),
    ]

    MODEL_ENGINE_CHOICES = [
        ('best', 'Auto (Empirical Champion Model)'),
        ('xgboost', 'XGBoost Classifier'),
        ('random_forest', 'Random Forest'),
        ('logistic_regression', 'Logistic Regression'),
        ('ann', 'Artificial Neural Network (ANN)'),
    ]

    credit_history = forms.TypedChoiceField(
        choices=CREDIT_HISTORY_CHOICES,
        coerce=float,
        initial=1.0,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Credit Score / Repayment Record'
    )

    interest_rate = forms.FloatField(
        initial=10.5,
        min_value=0.0,
        max_value=30.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 10.5', 'step': '0.1', 'value': '10.5'}),
        label='Annual Interest Rate (%)'
    )

    model_engine = forms.ChoiceField(
        choices=MODEL_ENGINE_CHOICES,
        initial='best',
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Select AI Model Engine'
    )

    class Meta:
        model = LoanPrediction
        fields = [
            'gender', 'married', 'dependents', 'education', 'self_employed',
            'applicant_income', 'coapplicant_income', 'loan_amount',
            'loan_amount_term', 'interest_rate', 'credit_history', 'property_area'
        ]
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'married': forms.Select(attrs={'class': 'form-select'}),
            'dependents': forms.Select(attrs={'class': 'form-select'}),
            'education': forms.Select(attrs={'class': 'form-select'}),
            'self_employed': forms.Select(attrs={'class': 'form-select'}),
            'applicant_income': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. ₹50,000', 'min': 1, 'value': '50000'}),
            'coapplicant_income': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. ₹15,000', 'min': 0, 'value': '15000'}),
            'loan_amount': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 15 (₹15 Lakhs)', 'min': 0.1, 'value': '15'}),
            'loan_amount_term': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 360 (30 Yrs) or 24 (2 Yrs)', 'min': 1, 'max': 480, 'value': '360'}),
            'property_area': forms.Select(attrs={'class': 'form-select'}),
        }
        labels = {
            'applicant_income': 'Applicant Monthly Income (₹)',
            'coapplicant_income': 'Co-applicant Monthly Income (₹)',
            'loan_amount': 'Loan Amount Requested (₹ in Lakhs)',
            'loan_amount_term': 'Loan Tenure (Months)',
            'property_area': 'Property Location',
        }

    # --- Custom field-level validators (clean_<fieldname>) ---
    # Each method validates a single field after Django's built-in type coercion

    def clean_applicant_income(self):
        """Ensures applicant income is a positive number (required for EMI calculation)."""
        val = self.cleaned_data.get('applicant_income')
        if val is None or val <= 0:
            raise forms.ValidationError("Applicant monthly income must be greater than zero.")
        return val

    def clean_coapplicant_income(self):
        val = self.cleaned_data.get('coapplicant_income')
        if val is not None and val < 0:
            raise forms.ValidationError("Co-applicant monthly income cannot be negative.")
        return val or 0.0

    def clean_loan_amount(self):
        val = self.cleaned_data.get('loan_amount')
        if val is None or val <= 0:
            raise forms.ValidationError("Loan amount requested must be greater than zero.")
        return val

    def clean_loan_amount_term(self):
        val = self.cleaned_data.get('loan_amount_term')
        if val is None or val <= 0:
            raise forms.ValidationError("Loan tenure term must be greater than zero.")
        return val

    def clean_interest_rate(self):
        val = self.cleaned_data.get('interest_rate')
        if val is not None and val < 0:
            raise forms.ValidationError("Interest rate cannot be negative.")
        return val or 10.5

    def clean(self):
        """Cross-field validation: ensures combined household income is positive.
        This runs after all individual field clean methods have passed."""
        cleaned_data = super().clean()
        app_inc = cleaned_data.get('applicant_income') or 0.0
        coapp_inc = cleaned_data.get('coapplicant_income') or 0.0
        total_inc = app_inc + coapp_inc

        # Combined income must be positive to calculate meaningful EMI ratios
        if total_inc <= 0:
            raise forms.ValidationError("Combined household monthly income must be greater than zero.")

        return cleaned_data