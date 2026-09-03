# =============================================================================
# utils.py - ML/DL Inference Pipeline for LoanIQ Risk Assessment Platform
# =============================================================================
# This module implements the core LoanRiskAssessmentPipeline class that:
# 1. Loads all serialized ML models (7 sklearn classifiers) and DL model (ANN)
# 2. Prepares input DataFrames with automatic unit scaling for inference
# 3. Runs multi-model prediction and selects the best-performing model
# 4. Computes detailed financial explanations (EMI, risk score, ratios)
# The pipeline is instantiated as a singleton to avoid reloading models per request.
# =============================================================================

import os
import sys

# Dynamically add project root to Python path so that 'src' package is importable
# when this module is loaded by Django (which may have a different working directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
import math
import logging

from src.data.feature_engineering import add_engineered_features

logger = logging.getLogger(__name__)

# Directories where serialized model pipelines (.pkl and .keras files) are stored
MODELS_ML_DIR = os.path.join(PROJECT_ROOT, 'models', 'ml')
MODELS_DL_DIR = os.path.join(PROJECT_ROOT, 'models', 'dl')


class LoanRiskAssessmentPipeline:
    def __init__(self):
        self.best_model_pipeline = None
        self.ml_models = {}
        self.dl_model = None
        self.top_model_name = 'Logistic Regression'
        self.load_all_models()

    def load_all_models(self):
        """Loads serialized ML/DL pipelines and reads empirical top model name."""
        try:
            # Load top model name from model comparison report
            report_path = os.path.join(PROJECT_ROOT, 'reports', 'model_comparison.csv')
            if os.path.exists(report_path):
                try:
                    df_comp = pd.read_csv(report_path)
                    if not df_comp.empty:
                        self.top_model_name = df_comp.iloc[0]['Model']
                except Exception:
                    pass

            # Load Best Model Pipeline
            best_model_path = os.path.join(MODELS_ML_DIR, 'best_model.pkl')
            if os.path.exists(best_model_path):
                self.best_model_pipeline = joblib.load(best_model_path)
                logger.info(f"Loaded Best ML Model Pipeline: {self.top_model_name}")

            # Load ML comparison models
            for model_name in ['logistic_regression', 'decision_tree', 'random_forest', 'svm', 'knn', 'gradient_boosting', 'xgboost']:
                path = os.path.join(MODELS_ML_DIR, f"{model_name}.pkl")
                if os.path.exists(path):
                    self.ml_models[model_name] = joblib.load(path)

            # Load DL model
            dl_path = os.path.join(MODELS_DL_DIR, 'loan_risk_nn.keras')
            if os.path.exists(dl_path):
                self.dl_model = joblib.load(dl_path)
                logger.info("Loaded Deep Learning ANN Wrapper")

        except Exception as e:
            logger.error(f"Error loading prediction models: {e}")

    def prepare_input_dataframe(self, form_data: dict) -> pd.DataFrame:
        """
        Constructs pandas DataFrame for ML models with automatic unit scaling:
        - Monthly Income in ₹ (e.g. ₹50,000 -> dataset scale 5000)
        - Loan Amount in ₹ Lakhs (e.g. ₹15 Lakhs or ₹1,500,000 -> dataset scale 150)
        """
        raw_app_inc = float(form_data.get('applicant_income', 50000))
        raw_coapp_inc = float(form_data.get('coapplicant_income', 15000))
        raw_loan_amt = float(form_data.get('loan_amount', 15))

        # Income Scaling: The training dataset uses income values in hundreds
        # (e.g. 5000 = ₹50,000). Web form collects actual INR values, so
        # divide by 10 to match the scale the models were trained on.
        scaled_app_inc = raw_app_inc / 10.0 if raw_app_inc >= 10000.0 else raw_app_inc
        scaled_coapp_inc = raw_coapp_inc / 10.0 if raw_coapp_inc >= 10000.0 else raw_coapp_inc

        # Loan Amount Scaling: Training data uses loan amount in thousands.
        # Handle three input formats: absolute INR (>=10000), lakhs (<=100),
        # or already in the correct scale (100-10000).
        if raw_loan_amt >= 10000.0:
            scaled_loan_amt = raw_loan_amt / 10000.0
        elif raw_loan_amt <= 100.0:
            scaled_loan_amt = raw_loan_amt * 10.0
        else:
            scaled_loan_amt = raw_loan_amt

        # Build a dictionary matching the exact column names used during training
        # (e.g., 'ApplicantIncome' not 'applicant_income')
        raw_dict = {
            'Gender': form_data.get('gender', 'Male'),
            'Married': form_data.get('married', 'Yes'),
            'Dependents': form_data.get('dependents', '0'),
            'Education': form_data.get('education', 'Graduate'),
            'Self_Employed': form_data.get('self_employed', 'No'),
            'ApplicantIncome': scaled_app_inc,
            'CoapplicantIncome': scaled_coapp_inc,
            'LoanAmount': scaled_loan_amt,
            'Loan_Amount_Term': float(form_data.get('loan_amount_term', 360)),
            'Credit_History': float(form_data.get('credit_history', 1.0)),
            'Property_Area': form_data.get('property_area', 'Urban')
        }

        # Create DataFrame and apply feature engineering (TotalIncome, LoanIncomeRatio,
        # log transforms, EstimatedEMI, etc.) to match the training feature set
        df = pd.DataFrame([raw_dict])
        df = add_engineered_features(df)
        return df

    def calculate_explanation_details(self, form_data: dict, model_predictions: dict, approval_prob: float, risk_score: float, risk_level: str) -> dict:
        """
        Calculates complete step-by-step financial breakdown, formulas, and model mechanics.
        """
        raw_app_inc = float(form_data.get('applicant_income', 50000))
        raw_coapp_inc = float(form_data.get('coapplicant_income', 15000))
        raw_loan_amt = float(form_data.get('loan_amount', 15))
        tenure_months = float(form_data.get('loan_amount_term', 360))
        annual_rate = float(form_data.get('interest_rate', 10.5))

        if raw_loan_amt >= 10000.0:
            loan_inr = raw_loan_amt
            loan_lakhs = raw_loan_amt / 100000.0
        else:
            loan_lakhs = raw_loan_amt
            loan_inr = raw_loan_amt * 100000.0

        total_income_monthly = raw_app_inc + raw_coapp_inc
        annual_total_income = total_income_monthly * 12.0  # Always 12 months for annual income
        tenure_years = round(tenure_months / 12.0, 1)

        # Loan to Annual Income Ratio: measures how large the loan is relative
        # to the borrower's annual earning capacity (lower is safer)
        loan_to_income_ratio = round(loan_inr / annual_total_income, 2) if annual_total_income > 0 else 0.0

        # Principal-only monthly repayment (without interest, used as fallback)
        principal_monthly = round(loan_inr / (tenure_months if tenure_months > 0 else 1.0), 0)

        # Bank Standard EMI Calculation using the reducing balance formula:
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        # where P = principal, r = monthly interest rate, n = number of months
        r = annual_rate / (12.0 * 100.0)  # Convert annual % to monthly decimal rate
        n = tenure_months
        if n > 0 and r > 0:
            emi = loan_inr * r * (math.pow(1 + r, n)) / (math.pow(1 + r, n) - 1)
        else:
            # If interest rate is 0, fall back to simple principal division
            emi = principal_monthly

        emi_rounded = round(emi, 0)
        emi_to_income_pct = round((emi / total_income_monthly * 100.0), 1) if total_income_monthly > 0 else 0.0

        # Financial Health Status Badge & Label
        if emi_to_income_pct <= 30.0:
            burden_badge = "Healthy Repayment Burden (≤ 30%)"
            burden_class = "success"
            is_high_emi_burden = False
        elif emi_to_income_pct <= 40.0:
            burden_badge = "Moderate Repayment Burden (30% - 40%)"
            burden_class = "warning"
            is_high_emi_burden = False
        else:
            burden_badge = "⚠️ High Repayment Burden (> 40%)"
            burden_class = "danger"
            is_high_emi_burden = True

        # Feature transformations
        app_inc_log = round(float(np.log1p(raw_app_inc)), 2)
        loan_amt_log = round(float(np.log1p(loan_lakhs)), 2)

        # Consensus stats
        approved_count = sum(1 for m in model_predictions.values() if m['prediction'] == 'Approved')
        rejected_count = sum(1 for m in model_predictions.values() if m['prediction'] == 'Rejected')
        total_models = len(model_predictions)

        return {
            'applicant_info': {
                'gender': form_data.get('gender', 'Male'),
                'married': form_data.get('married', 'Yes'),
                'dependents': form_data.get('dependents', '0'),
                'education': form_data.get('education', 'Graduate'),
                'self_employed': form_data.get('self_employed', 'No'),
                'applicant_income': raw_app_inc,
                'coapplicant_income': raw_coapp_inc,
                'loan_amount_lakhs': loan_lakhs,
                'loan_amount_inr': loan_inr,
                'loan_term_months': int(tenure_months),
                'loan_term_years': tenure_years,
                'interest_rate': annual_rate,
                'credit_history': float(form_data.get('credit_history', 1.0)),
                'property_area': form_data.get('property_area', 'Urban')
            },
            'financial_calcs': {
                'total_income_monthly': total_income_monthly,
                'annual_total_income': annual_total_income,
                'loan_to_income_ratio': loan_to_income_ratio,
                'annual_interest_rate': annual_rate,
                'principal_monthly': principal_monthly,
                'monthly_emi': emi_rounded,
                'emi_to_income_pct': emi_to_income_pct,
                'burden_badge': burden_badge,
                'burden_class': burden_class,
                'is_high_emi_burden': is_high_emi_burden
            },
            'feature_engineering': {
                'app_inc_log': app_inc_log,
                'loan_amt_log': loan_amt_log,
                'loan_income_ratio': loan_to_income_ratio
            },
            'consensus': {
                'approved_count': approved_count,
                'rejected_count': rejected_count,
                'total_models': total_models
            },
            'final_decision': {
                'approval_prob': approval_prob,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'formula': 'Risk Score = 100 - Approval Probability'
            }
        }

    def predict(self, form_data: dict) -> dict:
        """
        Executes full inference workflow across all models and sets dynamic winning model name.
        """
        df_input = self.prepare_input_dataframe(form_data)

        if self.best_model_pipeline is None:
            self.load_all_models()

        # Compute predictions across all ML models + ANN
        model_predictions = {}
        for name, pipe in self.ml_models.items():
            m_prob = float(pipe.predict_proba(df_input)[0][1]) * 100.0
            model_predictions[name.replace('_', ' ').title()] = {
                'prediction': 'Approved' if m_prob >= 50.0 else 'Rejected',
                'probability': round(m_prob, 1)
            }

        if self.dl_model is not None:
            dl_prob = float(self.dl_model.predict_proba(df_input)[0][1]) * 100.0
            model_predictions['Neural Network (ANN)'] = {
                'prediction': 'Approved' if dl_prob >= 50.0 else 'Rejected',
                'probability': round(dl_prob, 1)
            }

        # Check model engine selection or select top model dynamically
        requested_engine = form_data.get('model_engine', 'best')
        if requested_engine == 'xgboost' and 'Xgboost' in model_predictions:
            primary_model_name = 'XGBoost'
            approval_prob = model_predictions['Xgboost']['probability']
        elif requested_engine == 'random_forest' and 'Random Forest' in model_predictions:
            primary_model_name = 'Random Forest'
            approval_prob = model_predictions['Random Forest']['probability']
        elif requested_engine == 'logistic_regression' and 'Logistic Regression' in model_predictions:
            primary_model_name = 'Logistic Regression'
            approval_prob = model_predictions['Logistic Regression']['probability']
        elif requested_engine == 'ann' and 'Neural Network (ANN)' in model_predictions:
            primary_model_name = 'Neural Network (ANN)'
            approval_prob = model_predictions['Neural Network (ANN)']['probability']
        else:
            # Auto: Use top model from comparison report
            primary_model_name = self.top_model_name
            if primary_model_name in model_predictions:
                approval_prob = model_predictions[primary_model_name]['probability']
            else:
                probs = self.best_model_pipeline.predict_proba(df_input)[0]
                approval_prob = float(probs[1]) * 100.0

        # Risk Score is the complement of approval probability (higher = riskier)
        risk_score = round((100.0 - approval_prob), 1)

        # Three-tier approval threshold system:
        # >= 70%: Approved with LOW risk (strong candidate)
        # 50-69%: Approved with MEDIUM risk (marginal, needs review)
        # < 50% : Rejected with HIGH risk (insufficient creditworthiness)
        if approval_prob >= 70.0:
            prediction_label = 'Approved'
            risk_level = 'LOW'
        elif approval_prob >= 50.0:
            prediction_label = 'Approved'
            risk_level = 'MEDIUM'
        else:
            prediction_label = 'Rejected'
            risk_level = 'HIGH'

        # Detailed calculation explanation breakdown
        explanation_details = self.calculate_explanation_details(
            form_data, model_predictions, round(approval_prob, 1), risk_score, risk_level
        )

        return {
            'prediction': prediction_label,
            'approval_probability': round(approval_prob, 1),
            'probability': round(approval_prob, 1),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'model_name': primary_model_name,
            'model_predictions': model_predictions,
            'explanation_details': explanation_details,
            'input_summary': form_data
        }


# Singleton pattern: instantiate the pipeline once at module load time.
# All Django views share this single instance to avoid reloading 8+ ML models
# from disk on every HTTP request. Models are loaded into memory once and reused.
pipeline_instance = LoanRiskAssessmentPipeline()


def get_prediction_pipeline():
    """Returns the shared singleton pipeline instance used by all views."""
    return pipeline_instance