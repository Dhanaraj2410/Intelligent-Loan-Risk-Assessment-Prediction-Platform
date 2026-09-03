# =============================================================================
# admin.py - Django Admin Panel Configuration for LoanIQ Platform
# =============================================================================
# Customizes the Django Admin interface with colored badges, organized fieldsets,
# search/filter capabilities, and date-based navigation for all models.
# =============================================================================

from django.contrib import admin
from django.utils.html import format_html
from .models import LoanPrediction, ModelLog, UserFeedback

@admin.register(LoanPrediction)
class LoanPredictionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 
        'prediction_badge', 
        'probability_badge',
        'applicant_income', 
        'loan_amount', 
        'credit_history_badge',
        'created_at'
    ]
    list_filter = [
        'prediction', 
        'gender', 
        'married', 
        'education', 
        'property_area',
        'credit_history'
    ]
    search_fields = ['id', 'applicant_income', 'loan_amount', 'ip_address']
    readonly_fields = ['created_at', 'ip_address']
    date_hierarchy = 'created_at'
    list_per_page = 50
    
    fieldsets = (
        ('Personal Information', {
            'fields': ('gender', 'married', 'dependents', 'education', 'self_employed')
        }),
        ('Financial Information', {
            'fields': ('applicant_income', 'coapplicant_income', 'total_income', 'annual_income',
                      'loan_amount', 'loan_amount_term', 'interest_rate', 'estimated_emi',
                      'emi_income_ratio', 'credit_history', 'property_area')
        }),
        ('Prediction Results', {
            'fields': ('prediction', 'approval_probability', 'risk_score', 'risk_level', 'model_name'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'ip_address'),
            'classes': ('collapse',)
        })
    )
    
    def prediction_badge(self, obj):
        """Renders the prediction outcome as a color-coded Bootstrap badge
        in the admin list view (green=Approved, red=Rejected, grey=Pending)."""
        if obj.prediction == 'Approved':
            color = 'success'
        elif obj.prediction == 'Rejected':
            color = 'danger'
        else:
            color = 'secondary'
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            color,
            obj.prediction or 'Pending'
        )
    prediction_badge.short_description = 'Prediction'
    
    def probability_badge(self, obj):
        prob = obj.approval_probability or obj.probability
        if prob:
            if prob >= 70:
                color = 'success'
            elif prob >= 50:
                color = 'warning'
            else:
                color = 'danger'
            return format_html(
                '<span class="badge bg-{}">{:.1f}%</span>',
                color,
                prob
            )
        return format_html('<span class="badge bg-secondary">N/A</span>')
    probability_badge.short_description = 'Confidence'
    
    def credit_history_badge(self, obj):
        if obj.credit_history == 1:
            color = 'success'
            text = 'Good'
        else:
            color = 'danger'
            text = 'Bad'
        return format_html('<span class="badge bg-{}">{}</span>', color, text)
    credit_history_badge.short_description = 'Credit History'


@admin.register(ModelLog)
class ModelLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'trained_at', 'is_active']
    list_filter = ['is_active', 'trained_at']
    readonly_fields = ['trained_at']
    search_fields = ['model_name']


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'prediction', 'is_correct_badge', 'created_at']
    list_filter = ['is_correct', 'created_at']
    search_fields = ['comment', 'prediction__id']
    readonly_fields = ['created_at']
    
    def is_correct_badge(self, obj):
        if obj.is_correct:
            return format_html('<span class="badge bg-success">Yes</span>')
        return format_html('<span class="badge bg-danger">No</span>')
    is_correct_badge.short_description = 'Correct'