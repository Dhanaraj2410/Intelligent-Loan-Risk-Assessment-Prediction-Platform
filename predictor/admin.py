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
    readonly_fields = ['created_at', 'ip_address', 'user_agent']
    date_hierarchy = 'created_at'
    list_per_page = 50
    
    fieldsets = (
        ('Personal Information', {
            'fields': ('gender', 'married', 'dependents', 'education', 'self_employed')
        }),
        ('Financial Information', {
            'fields': ('applicant_income', 'coapplicant_income', 'loan_amount', 
                      'loan_amount_term', 'credit_history', 'property_area')
        }),
        ('Prediction Results', {
            'fields': ('prediction', 'probability'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'ip_address', 'user_agent'),
            'classes': ('collapse',)
        })
    )
    
    def prediction_badge(self, obj):
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
        if obj.probability:
            if obj.probability >= 70:
                color = 'success'
            elif obj.probability >= 50:
                color = 'warning'
            else:
                color = 'danger'
            return format_html(
                '<span class="badge bg-{}">{:.1f}%</span>',
                color,
                obj.probability
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
    list_display = ['id', 'prediction', 'model_accuracy', 'response_time', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['created_at']
    search_fields = ['prediction__id']


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