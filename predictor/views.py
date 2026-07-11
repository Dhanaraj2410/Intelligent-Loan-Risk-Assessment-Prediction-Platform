from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Avg
import json
import time
import logging
from .forms import LoanPredictionForm
from .models import LoanPrediction, ModelLog, UserFeedback
from .utils import LoanPredictionModel

logger = logging.getLogger(__name__)

# Initialize model once
model = LoanPredictionModel()

def index(request):
    """Home page view"""
    context = {
        'title': 'Home - Loan Prediction',
        'features': model.get_feature_names() if model.is_loaded() else [],
        'model_loaded': model.is_loaded()
    }
    return render(request, 'predictor/index.html', context)


def predict(request):
    """Prediction form and processing view"""
    if not model.is_loaded():
        messages.error(request, 'Model is not loaded. Please contact administrator.')
        return redirect('index')
    
    if request.method == 'POST':
        form = LoanPredictionForm(request.POST)
        
        if form.is_valid():
            # Prepare data for prediction
            data = {
                'gender': form.cleaned_data.get('gender'),
                'married': form.cleaned_data.get('married'),
                'dependents': form.cleaned_data.get('dependents'),
                'education': form.cleaned_data.get('education'),
                'self_employed': form.cleaned_data.get('self_employed'),
                'applicant_income': float(form.cleaned_data.get('applicant_income', 0)),
                'coapplicant_income': float(form.cleaned_data.get('coapplicant_income', 0)),
                'loan_amount': float(form.cleaned_data.get('loan_amount', 0)),
                'loan_amount_term': float(form.cleaned_data.get('loan_amount_term', 360)),
                'credit_history': float(form.cleaned_data.get('credit_history', 0)),
                'property_area': form.cleaned_data.get('property_area')
            }
            
            logger.info(f"Prediction data: {data}")
            
            # Make prediction
            start_time = time.time()
            result, probability = model.predict(data)
            end_time = time.time()
            response_time = end_time - start_time
            
            if result:
                # Save form data to database
                prediction_record = form.save(commit=False)
                prediction_record.prediction = result
                prediction_record.probability = probability
                prediction_record.ip_address = get_client_ip(request)
                prediction_record.user_agent = request.META.get('HTTP_USER_AGENT', '')
                prediction_record.save()
                
                # Log model performance
                ModelLog.objects.create(
                    prediction=prediction_record,
                    model_accuracy=0.94,
                    response_time=response_time
                )
                
                request.session['last_prediction_id'] = prediction_record.id
                
                context = {
                    'prediction': result,
                    'probability': probability,
                    'prediction_id': prediction_record.id,
                    'data': data,
                    'title': 'Prediction Result'
                }
                return render(request, 'predictor/result.html', context)
            else:
                messages.error(request, 'Prediction failed. Please try again.')
        else:
            messages.error(request, 'Please correct the errors below.')
            logger.error(f"Form errors: {form.errors}")
    else:
        form = LoanPredictionForm()
    
    context = {
        'form': form,
        'title': 'Predict Loan',
        'model_loaded': model.is_loaded()
    }
    return render(request, 'predictor/predict.html', context)


def result(request, prediction_id):
    """View specific prediction result"""
    try:
        prediction = LoanPrediction.objects.get(id=prediction_id)
        context = {
            'prediction': prediction,
            'title': 'Prediction Result'
        }
        return render(request, 'predictor/result.html', context)
    except LoanPrediction.DoesNotExist:
        messages.error(request, 'Prediction not found.')
        return redirect('predict')


def history(request):
    """View prediction history with statistics"""
    # Get all predictions
    all_predictions = LoanPrediction.objects.all()
    
    # Calculate statistics
    total_predictions = all_predictions.count()
    
    # Count approved and rejected
    approved_count = all_predictions.filter(prediction='Approved').count()
    rejected_count = all_predictions.filter(prediction='Rejected').count()
    
    # Calculate average confidence
    avg_confidence = all_predictions.aggregate(Avg('probability'))['probability__avg']
    
    # Get latest predictions (limit to 50 for display)
    predictions = all_predictions[:50]
    
    # Get feedback statistics
    feedback_stats = {}
    feedback_correct = 0
    feedback_total = 0
    
    for pred in predictions:
        feedbacks = pred.feedbacks.all()
        feedback_total += feedbacks.count()
        feedback_correct += feedbacks.filter(is_correct=True).count()
    
    feedback_accuracy = (feedback_correct / feedback_total * 100) if feedback_total > 0 else 0
    
    context = {
        'predictions': predictions,
        'total_predictions': total_predictions,
        'approved_count': approved_count,
        'rejected_count': rejected_count,
        'avg_confidence': avg_confidence if avg_confidence else 0,
        'feedback_accuracy': round(feedback_accuracy, 1),
        'feedback_total': feedback_total,
        'title': 'Prediction History'
    }
    return render(request, 'predictor/history.html', context)

@csrf_exempt
def feedback(request):
    """Handle user feedback via AJAX"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prediction_id = data.get('prediction_id')
            is_correct = data.get('is_correct')
            comment = data.get('comment', '')
            
            if not prediction_id:
                return JsonResponse({'error': 'Prediction ID required'}, status=400)
            
            prediction = LoanPrediction.objects.get(id=prediction_id)
            feedback = UserFeedback.objects.create(
                prediction=prediction,
                is_correct=is_correct,
                comment=comment
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Feedback submitted successfully'
            })
            
        except LoanPrediction.DoesNotExist:
            return JsonResponse({'error': 'Prediction not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@csrf_exempt
def api_predict(request):
    """API endpoint for prediction"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Make prediction
        result, probability = model.predict(data)
        
        if result is None:
            return JsonResponse({'error': 'Prediction failed'}, status=500)
        
        return JsonResponse({
            'success': True,
            'prediction': result,
            'probability': round(probability, 2),
            'message': f'Loan {result.lower()} with {round(probability, 2)}% confidence'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def model_info(request):
    """Get model information"""
    info = {
        'loaded': model.is_loaded(),
        'features': model.get_feature_names() if model.is_loaded() else [],
        'total_predictions': LoanPrediction.objects.count(),
        'approved_count': LoanPrediction.objects.filter(prediction='Approved').count(),
        'rejected_count': LoanPrediction.objects.filter(prediction='Rejected').count(),
    }
    return JsonResponse(info)


@csrf_exempt
def test_prediction(request):
    """Test endpoint to verify model prediction with sample data"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Make prediction
            result, probability = model.predict(data)
            
            return JsonResponse({
                'success': True,
                'prediction': result,
                'probability': probability,
                'input_data': data
            })
        except Exception as e:
            logger.error(f"Test prediction error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    # GET request - return sample data
    sample_data = {
        'gender': 'Male',
        'married': 'Yes',
        'dependents': '0',
        'education': 'Graduate',
        'self_employed': 'No',
        'applicant_income': 5000,
        'coapplicant_income': 1000,
        'loan_amount': 150,
        'loan_amount_term': 360,
        'credit_history': 1,
        'property_area': 'Urban'
    }
    
    return JsonResponse({
        'sample_data': sample_data,
        'features': model.get_feature_names() if model.is_loaded() else []
    })