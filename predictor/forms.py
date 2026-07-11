from django import forms
from .models import LoanPrediction

class LoanPredictionForm(forms.ModelForm):
    class Meta:
        model = LoanPrediction
        fields = [
            'gender', 'married', 'dependents', 'education', 'self_employed',
            'applicant_income', 'coapplicant_income', 'loan_amount',
            'loan_amount_term', 'credit_history', 'property_area'
        ]
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'married': forms.Select(attrs={'class': 'form-control'}),
            'dependents': forms.Select(attrs={'class': 'form-control'}),
            'education': forms.Select(attrs={'class': 'form-control'}),
            'self_employed': forms.Select(attrs={'class': 'form-control'}),
            'applicant_income': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter applicant income'}),
            'coapplicant_income': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter coapplicant income'}),
            'loan_amount': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter loan amount'}),
            'loan_amount_term': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter loan term (months)'}),
            'credit_history': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter credit history (0 or 1)'}),
            'property_area': forms.Select(attrs={'class': 'form-control'}),
        }
        labels = {
            'applicant_income': 'Applicant Income',
            'coapplicant_income': 'Coapplicant Income',
            'loan_amount': 'Loan Amount',
            'loan_amount_term': 'Loan Amount Term (Months)',
            'credit_history': 'Credit History (0 or 1)',
        }