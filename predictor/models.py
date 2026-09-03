# =============================================================================
# models.py - Django ORM Models for LoanIQ Risk Assessment Platform
# =============================================================================
# Defines the MySQL database schema for storing loan prediction records,
# ML model training logs, and user feedback on prediction accuracy.
# Uses Django's ORM to map Python classes to relational database tables.
# =============================================================================

import os
from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class LoanPrediction(models.Model):
    """
    Stores complete loan assessment inputs, financial calculations,
    and multi-model prediction outputs.
    """
    # 1. Applicant Demographic Inputs
    gender = models.CharField(
        max_length=10,
        choices=[('Male', 'Male'), ('Female', 'Female')],
        default='Male',
        help_text="Gender of the applicant"
    )

    married = models.CharField(
        max_length=5,
        choices=[('Yes', 'Yes'), ('No', 'No')],
        default='No',
        help_text="Marital Status"
    )

    dependents = models.CharField(
        max_length=5,
        choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3+', '3+')],
        default='0',
        help_text="Number of dependents"
    )

    education = models.CharField(
        max_length=20,
        choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')],
        default='Graduate',
        help_text="Education Level"
    )

    self_employed = models.CharField(
        max_length=5,
        choices=[('Yes', 'Yes'), ('No', 'No')],
        default='No',
        help_text="Self employment status"
    )

    # 2. Financial Profile Inputs
    applicant_income = models.FloatField(
        validators=[MinValueValidator(1.0)],
        help_text="Applicant monthly income in INR (₹)"
    )

    coapplicant_income = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Co-applicant monthly income in INR (₹)"
    )

    total_income = models.FloatField(
        default=0.0,
        help_text="Combined monthly income (Applicant + Coapplicant)"
    )

    annual_income = models.FloatField(
        default=0.0,
        help_text="Annual total income (Total Monthly Income * 12)"
    )

    loan_amount = models.FloatField(
        validators=[MinValueValidator(0.1)],
        help_text="Loan amount requested in INR Lakhs"
    )

    loan_amount_term = models.FloatField(
        default=360,
        validators=[MinValueValidator(1.0), MaxValueValidator(480.0)],
        help_text="Loan tenure term in months"
    )

    interest_rate = models.FloatField(
        default=10.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(30.0)],
        help_text="Annual Interest Rate percentage (e.g. 10.5%)"
    )

    estimated_emi = models.FloatField(
        default=0.0,
        help_text="Calculated monthly bank EMI in INR (₹)"
    )

    emi_income_ratio = models.FloatField(
        default=0.0,
        help_text="EMI to monthly income ratio percentage"
    )

    credit_history = models.FloatField(
        choices=[(1.0, 'Good Credit History (1.0)'), (0.0, 'Poor Credit History (0.0)')],
        default=1.0,
        help_text="Credit History Record"
    )

    property_area = models.CharField(
        max_length=20,
        choices=[('Urban', 'Urban'), ('Semiurban', 'Semiurban'), ('Rural', 'Rural')],
        default='Urban',
        help_text="Property Area Location"
    )

    # 3. Model Output & Risk Scores
    prediction = models.CharField(
        max_length=20,
        default='Approved',
        help_text="Prediction Outcome: Approved or Rejected"
    )

    probability = models.FloatField(
        blank=True, 
        null=True,
        help_text="Confidence percentage of the prediction"
    )

    approval_probability = models.FloatField(
        default=0.0,
        help_text="Calculated approval probability percentage (0-100)"
    )

    risk_score = models.FloatField(
        default=0.0,
        help_text="Calculated risk score (0-100)"
    )

    risk_level = models.CharField(
        max_length=20,
        default='LOW',
        help_text="Risk Category: LOW, MEDIUM, HIGH"
    )

    model_name = models.CharField(
        max_length=100,
        default='XGBoost',
        help_text="Name of the model used for primary prediction"
    )

    # 4. Multi-model individual prediction outputs
    # Each field stores the Approved/Rejected outcome from a specific ML/DL model.
    # This enables the model comparison dashboard to show per-model consensus.
    logistic_prediction = models.CharField(max_length=20, default='Approved')
    random_forest_prediction = models.CharField(max_length=20, default='Approved')
    decision_tree_prediction = models.CharField(max_length=20, default='Approved')
    svm_prediction = models.CharField(max_length=20, default='Approved')
    knn_prediction = models.CharField(max_length=20, default='Approved')
    gradient_boosting_prediction = models.CharField(max_length=20, default='Approved')
    xgboost_prediction = models.CharField(max_length=20, default='Approved')
    ann_prediction = models.CharField(max_length=20, default='Approved')  # Deep Learning ANN model
    
    # Metadata
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time of prediction"
    )
    
    ip_address = models.GenericIPAddressField(
        blank=True, 
        null=True,
        help_text="IP Address of the user"
    )

    class Meta:
        # Default ordering: newest predictions first (reverse chronological)
        ordering = ['-created_at']
        verbose_name = 'Loan Prediction'
        verbose_name_plural = 'Loan Predictions'

    def __str__(self):
        return f"Prediction #{self.id} - {self.prediction} ({self.approval_probability:.1f}%)"

    @classmethod
    def get_statistics(cls):
        """Calculates dashboard summary statistics using Django ORM aggregation."""
        total = cls.objects.count()
        # Return zero-initialized stats if no predictions exist yet
        if total == 0:
            return {
                'total': 0,
                'approved': 0,
                'rejected': 0,
                'approval_rate': 0.0,
                'avg_probability': 0.0,
                'avg_risk_score': 0.0
            }
        
        # Count approved and rejected predictions separately
        approved = cls.objects.filter(prediction='Approved').count()
        rejected = cls.objects.filter(prediction='Rejected').count()
        
        # Use Django's Avg aggregate function for database-level averaging
        # (more efficient than loading all records into Python memory)
        avg_prob = cls.objects.aggregate(models.Avg('approval_probability'))['approval_probability__avg'] or 0.0
        avg_risk = cls.objects.aggregate(models.Avg('risk_score'))['risk_score__avg'] or 0.0
        
        return {
            'total': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': round((approved / total) * 100, 1),
            'avg_probability': round(avg_prob, 1),
            'avg_risk_score': round(avg_risk, 1)
        }


class ModelLog(models.Model):
    """Logs model training and performance metrics."""
    model_name = models.CharField(max_length=100, default='Model')
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    roc_auc = models.FloatField(default=0.0)
    trained_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-f1_score']
        verbose_name = 'Model Log'
        verbose_name_plural = 'Model Logs'

    def __str__(self):
        return f"{self.model_name} - F1: {self.f1_score:.4f}"


class UserFeedback(models.Model):
    """Captures user feedback on prediction accuracy."""
    prediction = models.ForeignKey(LoanPrediction, on_delete=models.CASCADE, related_name='feedbacks')
    is_correct = models.BooleanField(help_text="Was the AI prediction accurate?")
    comment = models.TextField(blank=True, null=True, help_text="Optional feedback comments")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'User Feedback'
        verbose_name_plural = 'User Feedbacks'

    def __str__(self):
        return f"Feedback for #{self.prediction.id} - {'Accurate' if self.is_correct else 'Inaccurate'}"