# =============================================================================
# views.py - Django View Controllers for LoanIQ Risk Assessment Platform
# =============================================================================
# This module contains all HTTP request handlers (views) for the loan prediction
# web application. It handles page rendering, form processing, multi-model ML
# inference orchestration, REST API endpoints, and user feedback collection.
# =============================================================================

import os
import json
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Avg, Count

# Import application-specific models, forms, and the ML prediction pipeline
from .models import LoanPrediction, UserFeedback
from .forms import LoanPredictionForm
from .utils import get_prediction_pipeline


def landing(request):
    """Landing homepage presenting LoanIQ platform capabilities."""
    # Fetch aggregate statistics (total predictions, approval rate, etc.)
    # from the database to display on the landing page hero section
    stats = LoanPrediction.get_statistics()
    return render(request, 'predictor/landing.html', {
        'stats': stats
    })


def dashboard(request):
    """SaaS Analytics Dashboard with live metrics, charts, and recent assessments."""
    total_count = LoanPrediction.objects.count()
    approved_count = LoanPrediction.objects.filter(prediction='Approved').count()
    rejected_count = LoanPrediction.objects.filter(prediction='Rejected').count()

    approval_rate = round((approved_count / total_count * 100), 1) if total_count > 0 else 0.0
    avg_risk_score = LoanPrediction.objects.aggregate(Avg('risk_score'))['risk_score__avg'] or 0.0

    # Load best model name from report
    report_csv_path = settings.BASE_DIR / 'reports' / 'model_comparison.csv'
    best_model_name = 'XGBoost (Tuned)'
    if os.path.exists(report_csv_path):
        try:
            df_comp = pd.read_csv(report_csv_path)
            if not df_comp.empty:
                best_model_name = df_comp.iloc[0]['Model']
        except Exception:
            pass

    recent_assessments = LoanPrediction.objects.all().order_by('-created_at')[:10]

    # Chart data: Risk category distribution
    low_risk_count = LoanPrediction.objects.filter(risk_level='LOW').count()
    med_risk_count = LoanPrediction.objects.filter(risk_level='MEDIUM').count()
    high_risk_count = LoanPrediction.objects.filter(risk_level='HIGH').count()

    chart_risk_dist = {
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'data': [low_risk_count, med_risk_count, high_risk_count]
    }

    return render(request, 'predictor/dashboard.html', {
        'total_count': total_count,
        'approval_rate': approval_rate,
        'avg_risk_score': round(avg_risk_score, 1),
        'best_model_name': best_model_name,
        'recent_assessments': recent_assessments,
        'chart_risk_dist_json': json.dumps(chart_risk_dist)
    })


def predict(request):
    """Multi-step loan risk assessment form with strict validation."""
    if request.method == 'POST':
        # Bind submitted form data and validate all fields (income, loan amount, etc.)
        form = LoanPredictionForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            
            # Execute the full ML/DL inference pipeline which runs prediction
            # across all 8 trained models (Logistic Regression, Random Forest,
            # Decision Tree, SVM, KNN, Gradient Boosting, XGBoost, ANN)
            pipeline = get_prediction_pipeline()
            pred_res = pipeline.predict(cleaned_data)
            exp_details = pred_res['explanation_details']
            m_preds = pred_res['model_predictions']

            # Create database record using commit=False to populate computed
            # fields (risk score, EMI, etc.) before the actual database INSERT
            prediction_record = form.save(commit=False)
            prediction_record.prediction = pred_res['prediction']
            prediction_record.probability = pred_res['approval_probability']
            prediction_record.approval_probability = pred_res['approval_probability']
            prediction_record.risk_score = pred_res['risk_score']
            prediction_record.risk_level = pred_res['risk_level']
            prediction_record.model_name = pred_res['model_name']

            # Populate calculated financial fields from the explanation breakdown
            # These are derived values: total income, annual income, EMI, and ratios
            prediction_record.total_income = exp_details['financial_calcs']['total_income_monthly']
            prediction_record.annual_income = exp_details['financial_calcs']['annual_total_income']
            prediction_record.estimated_emi = exp_details['financial_calcs']['monthly_emi']
            prediction_record.emi_income_ratio = exp_details['financial_calcs']['emi_to_income_pct']
            prediction_record.interest_rate = cleaned_data.get('interest_rate', 10.5)

            # Store each individual model's prediction outcome for the comparison
            # dashboard (allows per-model analysis in history and result views)
            prediction_record.logistic_prediction = m_preds.get('Logistic Regression', {}).get('prediction', 'Approved')
            prediction_record.random_forest_prediction = m_preds.get('Random Forest', {}).get('prediction', 'Approved')
            prediction_record.decision_tree_prediction = m_preds.get('Decision Tree', {}).get('prediction', 'Approved')
            prediction_record.svm_prediction = m_preds.get('Svm', {}).get('prediction', 'Approved')
            prediction_record.knn_prediction = m_preds.get('Knn', {}).get('prediction', 'Approved')
            prediction_record.gradient_boosting_prediction = m_preds.get('Gradient Boosting', {}).get('prediction', 'Approved')
            prediction_record.xgboost_prediction = m_preds.get('Xgboost', {}).get('prediction', 'Approved')
            prediction_record.ann_prediction = m_preds.get('Neural Network (ANN)', {}).get('prediction', 'Approved')

            # Persist the complete prediction record to the MySQL database
            prediction_record.save()

            # Cache multi-model predictions in Django session for the result page
            # (avoids re-running inference when redirecting to the result view)
            request.session['model_predictions'] = pred_res['model_predictions']
            request.session['explanation_details'] = pred_res['explanation_details']

            # Redirect to result page using POST-Redirect-GET pattern
            # to prevent duplicate form submissions on browser refresh
            return redirect('predictor:result', prediction_id=prediction_record.id)
    else:
        # GET request: render an empty loan prediction form
        form = LoanPredictionForm()

    return render(request, 'predictor/predict.html', {'form': form})


def result(request, prediction_id):
    """Detailed loan risk assessment report with 4-section layout & calculation breakdown."""
    # Retrieve the prediction record from database or return 404 if not found
    prediction_record = get_object_or_404(LoanPrediction, id=prediction_id)
    
    # Try to load cached predictions from session (set during predict view)
    model_predictions = request.session.get('model_predictions', {})
    explanation_details = request.session.get('explanation_details')

    if not explanation_details:
        # Session data expired or user accessed result page directly via URL.
        # Reconstruct the full explanation by re-running inference with the
        # saved form data from the database record.
        form_data = {
            'gender': prediction_record.gender,
            'married': prediction_record.married,
            'dependents': prediction_record.dependents,
            'education': prediction_record.education,
            'self_employed': prediction_record.self_employed,
            'applicant_income': prediction_record.applicant_income,
            'coapplicant_income': prediction_record.coapplicant_income,
            'loan_amount': prediction_record.loan_amount,
            'loan_amount_term': prediction_record.loan_amount_term,
            'interest_rate': prediction_record.interest_rate,
            'credit_history': prediction_record.credit_history,
            'property_area': prediction_record.property_area,
        }
        pipeline = get_prediction_pipeline()
        reconstructed = pipeline.predict(form_data)
        model_predictions = reconstructed['model_predictions']
        explanation_details = reconstructed['explanation_details']

    return render(request, 'predictor/result.html', {
        'record': prediction_record,
        'model_predictions': model_predictions,
        'explanation_details': explanation_details
    })


def history(request):
    """Data table log of previous loan risk assessments."""
    predictions = LoanPrediction.objects.all().order_by('-created_at')
    stats = LoanPrediction.get_statistics()
    return render(request, 'predictor/history.html', {
        'predictions': predictions,
        'stats': stats
    })


def history_detail(request, prediction_id):
    """Detailed view for a specific historical record."""
    prediction_record = get_object_or_404(LoanPrediction, id=prediction_id)
    form_data = {
        'gender': prediction_record.gender,
        'married': prediction_record.married,
        'dependents': prediction_record.dependents,
        'education': prediction_record.education,
        'self_employed': prediction_record.self_employed,
        'applicant_income': prediction_record.applicant_income,
        'coapplicant_income': prediction_record.coapplicant_income,
        'loan_amount': prediction_record.loan_amount,
        'loan_amount_term': prediction_record.loan_amount_term,
        'interest_rate': prediction_record.interest_rate,
        'credit_history': prediction_record.credit_history,
        'property_area': prediction_record.property_area,
    }
    pipeline = get_prediction_pipeline()
    reconstructed = pipeline.predict(form_data)

    return render(request, 'predictor/history_detail.html', {
        'record': prediction_record,
        'explanation_details': reconstructed['explanation_details'],
        'model_predictions': reconstructed['model_predictions']
    })


def model_comparison(request):
    """ML vs DL model performance comparison dashboard."""
    report_csv_path = settings.BASE_DIR / 'reports' / 'model_comparison.csv'
    models_data = []
    
    if os.path.exists(report_csv_path):
        df_comp = pd.read_csv(report_csv_path)
        models_data = df_comp.to_dict(orient='records')

    best_ml = next((m for m in models_data if 'ANN' not in m['Model']), None)
    best_dl = next((m for m in models_data if 'ANN' in m['Model']), None)
    overall_best = models_data[0] if models_data else None

    return render(request, 'predictor/models.html', {
        'models_data': models_data,
        'best_ml': best_ml,
        'best_dl': best_dl,
        'overall_best': overall_best,
        'json_data': json.dumps(models_data)
    })


# CSRF exemption applied because this is a stateless REST API endpoint
# intended for programmatic access (e.g., Postman, frontend JS, external services)
@csrf_exempt
def api_predict(request):
    """REST API endpoint for programmatic loan risk predictions."""
    # Only accept POST requests; reject GET, PUT, DELETE with 405 Method Not Allowed
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=405)

    try:
        # First try parsing request body as JSON (for API clients like Postman)
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        # Fallback to form-encoded POST data (for HTML form submissions)
        data = request.POST.dict()

    if not data:
        return JsonResponse({'error': 'No input data provided.'}, status=400)

    try:
        # Run inference pipeline and return full prediction results as JSON
        pipeline = get_prediction_pipeline()
        pred_res = pipeline.predict(data)

        return JsonResponse({
            'status': 'success',
            'prediction': pred_res['prediction'],
            'approval_probability': pred_res['approval_probability'],
            'risk_score': pred_res['risk_score'],
            'risk_level': pred_res['risk_level'],
            'model': pred_res['model_name'],
            'model_predictions': pred_res['model_predictions'],
            'explanation_details': pred_res['explanation_details']
        })
    except Exception as e:
        # Return 500 Internal Server Error with error message for debugging
        return JsonResponse({'error': str(e)}, status=500)


def model_info(request):
    """API endpoint providing current deployed model metadata."""
    report_csv_path = settings.BASE_DIR / 'reports' / 'model_comparison.csv'
    models_summary = []
    if os.path.exists(report_csv_path):
        df_comp = pd.read_csv(report_csv_path)
        models_summary = df_comp.to_dict(orient='records')

    return JsonResponse({
        'system': 'LoanIQ Intelligent Risk Platform',
        'version': '2.0-SaaS-ML-DL',
        'models': models_summary
    })


@csrf_exempt
def feedback(request):
    """Endpoint for user feedback on prediction accuracy."""
    if request.method == 'POST':
        prediction_id = request.POST.get('prediction_id')
        is_correct = request.POST.get('is_correct') == 'true'
        comment = request.POST.get('comment', '')

        if prediction_id:
            pred_record = get_object_or_404(LoanPrediction, id=prediction_id)
            UserFeedback.objects.create(
                prediction=pred_record,
                is_correct=is_correct,
                comment=comment
            )
            return JsonResponse({'status': 'success', 'message': 'Feedback recorded.'})

    return JsonResponse({'error': 'Invalid request.'}, status=400)