from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from django.db.models import Avg, Count, Case, When, IntegerField

class LoanPrediction(models.Model):
    # Personal Information Choices
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]
    
    MARRIED_CHOICES = [
        ('Yes', 'Yes'),
        ('No', 'No'),
    ]
    
    DEPENDENTS_CHOICES = [
        ('0', '0'),
        ('1', '1'),
        ('2', '2'),
        ('3+', '3+'),
    ]
    
    EDUCATION_CHOICES = [
        ('Graduate', 'Graduate'),
        ('Not Graduate', 'Not Graduate'),
    ]
    
    SELF_EMPLOYED_CHOICES = [
        ('Yes', 'Yes'),
        ('No', 'No'),
    ]
    
    PROPERTY_AREA_CHOICES = [
        ('Urban', 'Urban'),
        ('Semiurban', 'Semiurban'),
        ('Rural', 'Rural'),
    ]
    
    # Input Fields
    gender = models.CharField(
        max_length=10, 
        choices=GENDER_CHOICES,
        help_text="Select your gender"
    )
    
    married = models.CharField(
        max_length=3, 
        choices=MARRIED_CHOICES,
        help_text="Are you married?"
    )
    
    dependents = models.CharField(
        max_length=2, 
        choices=DEPENDENTS_CHOICES,
        help_text="Number of dependents"
    )
    
    education = models.CharField(
        max_length=20, 
        choices=EDUCATION_CHOICES,
        help_text="Your education level"
    )
    
    self_employed = models.CharField(
        max_length=3, 
        choices=SELF_EMPLOYED_CHOICES,
        help_text="Are you self-employed?"
    )
    
    applicant_income = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Your annual income in dollars"
    )
    
    coapplicant_income = models.FloatField(
        default=0, 
        validators=[MinValueValidator(0)],
        help_text="Co-applicant's annual income in dollars"
    )
    
    loan_amount = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Loan amount requested (in thousands)"
    )
    
    loan_amount_term = models.FloatField(
        default=360,
        validators=[
            MinValueValidator(6, message="Loan term must be at least 6 months"),
            MaxValueValidator(480, message="Loan term cannot exceed 480 months (40 years)")
        ],
        help_text="Loan term in months (typically 360 for 30-year mortgage)"
    )
    
    credit_history = models.FloatField(
        validators=[
            MinValueValidator(0, message="Credit history must be 0 or 1"),
            MaxValueValidator(1, message="Credit history must be 0 or 1")
        ],
        help_text="1 for good credit history, 0 for bad credit history"
    )
    
    property_area = models.CharField(
        max_length=20, 
        choices=PROPERTY_AREA_CHOICES,
        help_text="Area where the property is located"
    )
    
    # Result Fields
    prediction = models.CharField(
        max_length=10, 
        blank=True, 
        null=True,
        help_text="Prediction result: Approved or Rejected"
    )
    
    probability = models.FloatField(
        blank=True, 
        null=True,
        help_text="Confidence percentage of the prediction"
    )
    
    # Metadata
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time of prediction"
    )
    
    ip_address = models.GenericIPAddressField(
        blank=True, 
        null=True,
        help_text="IP address of the user"
    )
    
    user_agent = models.TextField(
        blank=True, 
        null=True,
        help_text="User's browser information"
    )
    
    def __str__(self):
        return f"Prediction #{self.id} - {self.prediction} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Loan Prediction'
        verbose_name_plural = 'Loan Predictions'
    
    @classmethod
    def get_statistics(cls):
        """
        Get comprehensive statistics for all predictions
        """
        total = cls.objects.count()
        
        if total == 0:
            return {
                'total': 0,
                'approved': 0,
                'rejected': 0,
                'avg_confidence': 0,
                'approval_rate': 0,
                'rejection_rate': 0,
                'total_feedback': 0,
                'feedback_accuracy': 0
            }
        
        approved = cls.objects.filter(prediction='Approved').count()
        rejected = cls.objects.filter(prediction='Rejected').count()
        
        # Calculate average confidence
        avg_confidence = cls.objects.aggregate(Avg('probability'))['probability__avg'] or 0
        
        # Get feedback statistics - use the related name
        # We need to count feedbacks through the relation
        total_feedback = 0
        correct_feedback = 0
        
        for pred in cls.objects.all():
            total_feedback += pred.feedbacks.count()
            correct_feedback += pred.feedbacks.filter(is_correct=True).count()
        
        feedback_accuracy = (correct_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            'total': total,
            'approved': approved,
            'rejected': rejected,
            'avg_confidence': round(avg_confidence, 1),
            'approval_rate': round(approved / total * 100, 1),
            'rejection_rate': round(rejected / total * 100, 1),
            'total_feedback': total_feedback,
            'feedback_accuracy': round(feedback_accuracy, 1)
        }
    
    @classmethod
    def get_recent_predictions(cls, limit=50):
        """
        Get recent predictions with related feedback counts
        """
        return cls.objects.all()[:limit]


class ModelLog(models.Model):
    """Log model performance metrics"""
    
    prediction = models.ForeignKey(
        LoanPrediction, 
        on_delete=models.CASCADE, 
        related_name='logs',
        help_text="Associated prediction"
    )
    
    model_accuracy = models.FloatField(
        help_text="Model accuracy percentage"
    )
    
    response_time = models.FloatField(
        help_text="Response time in seconds"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time of logging"
    )
    
    def __str__(self):
        return f"Log for Prediction #{self.prediction.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Model Log'
        verbose_name_plural = 'Model Logs'
    
    @classmethod
    def get_performance_stats(cls):
        """Get model performance statistics"""
        avg_response_time = cls.objects.aggregate(Avg('response_time'))['response_time__avg'] or 0
        avg_accuracy = cls.objects.aggregate(Avg('model_accuracy'))['model_accuracy__avg'] or 0
        
        return {
            'avg_response_time': round(avg_response_time, 3),
            'avg_accuracy': round(avg_accuracy, 1),
            'total_logs': cls.objects.count()
        }


class UserFeedback(models.Model):
    """User feedback on predictions"""
    
    prediction = models.ForeignKey(
        LoanPrediction, 
        on_delete=models.CASCADE, 
        related_name='feedbacks',
        help_text="Prediction being rated"
    )
    
    is_correct = models.BooleanField(
        help_text="Did the user find the prediction correct?"
    )
    
    comment = models.TextField(
        blank=True, 
        null=True,
        help_text="Optional user comment"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Date and time of feedback"
    )
    
    def __str__(self):
        status = "Correct" if self.is_correct else "Incorrect"
        return f"Feedback #{self.id} - {status} - Prediction #{self.prediction.id}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'User Feedback'
        verbose_name_plural = 'User Feedbacks'
    
    @classmethod
    def get_feedback_stats(cls):
        """Get feedback statistics"""
        total = cls.objects.count()
        
        if total == 0:
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0
            }
        
        correct = cls.objects.filter(is_correct=True).count()
        incorrect = cls.objects.filter(is_correct=False).count()
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(correct / total * 100, 1)
        }