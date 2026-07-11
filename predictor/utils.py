import pickle
import numpy as np
import pandas as pd
from django.conf import settings
import os
import logging

logger = logging.getLogger(__name__)

class LoanPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load all required model files"""
        try:
            model_path = settings.ML_MODELS_DIR / 'model.pkl'
            scaler_path = settings.ML_MODELS_DIR / 'scaler.pkl'
            encoders_path = settings.ML_MODELS_DIR / 'label_encoders.pkl'
            features_path = settings.ML_MODELS_DIR / 'feature_names.pkl'
            
            # Check if files exist
            for path in [model_path, scaler_path, encoders_path, features_path]:
                if not path.exists():
                    logger.error(f"Model file not found: {path}")
                    return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            logger.info(f"✅ Models loaded successfully")
            logger.info(f"Features: {self.feature_names}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, data):
        """
        Make prediction using the loaded model
        data: dict containing form data
        Returns: (prediction, probability)
        """
        try:
            # Prepare input data in the correct order
            input_data = self.prepare_input(data)
            
            logger.info(f"Prepared input data: {input_data}")
            
            # Scale the data
            scaled_input = self.scaler.transform(input_data)
            
            logger.info(f"Scaled input: {scaled_input}")
            
            # Make prediction
            prediction = self.model.predict(scaled_input)[0]
            probabilities = self.model.predict_proba(scaled_input)[0]
            
            logger.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
            
            # Convert prediction to human-readable format
            if prediction == 1:
                result = 'Approved'
                prob = probabilities[1] * 100
            else:
                result = 'Rejected'
                prob = probabilities[0] * 100
            
            return result, round(prob, 2)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def prepare_input(self, data):
        """Convert form data to model input format with correct feature order"""
        # Define the exact feature order from your model
        feature_order = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        encoded_values = []
        
        for feature in feature_order:
            # Map form field names to model feature names
            form_field = self._map_feature_name(feature)
            value = data.get(form_field, None)
            
            if value is None or value == '':
                logger.warning(f"Missing value for {feature}, using default")
                # Use appropriate default values
                if feature == 'Gender':
                    value = 'Male'
                elif feature == 'Married':
                    value = 'No'
                elif feature == 'Dependents':
                    value = '0'
                elif feature == 'Education':
                    value = 'Graduate'
                elif feature == 'Self_Employed':
                    value = 'No'
                elif feature == 'Property_Area':
                    value = 'Urban'
                else:
                    value = 0
            
            # Encode categorical variables
            if feature in self.label_encoders:
                try:
                    # Convert to string for encoding
                    str_value = str(value)
                    encoded_value = self.label_encoders[feature].transform([str_value])[0]
                    encoded_values.append(encoded_value)
                    logger.info(f"{feature}: {value} -> {encoded_value}")
                except Exception as e:
                    logger.error(f"Error encoding {feature} with value {value}: {e}")
                    # Try to handle common encodings
                    encoded_values.append(self._get_default_encoding(feature, value))
            else:
                # Numerical features - convert to float
                try:
                    encoded_values.append(float(value))
                except:
                    logger.error(f"Could not convert {feature} value {value} to float")
                    encoded_values.append(0)
        
        # Convert to numpy array and reshape
        return np.array(encoded_values, dtype=np.float64).reshape(1, -1)
    
    def _map_feature_name(self, feature_name):
        """Map model feature names to form field names"""
        mapping = {
            'Gender': 'gender',
            'Married': 'married',
            'Dependents': 'dependents',
            'Education': 'education',
            'Self_Employed': 'self_employed',
            'ApplicantIncome': 'applicant_income',
            'CoapplicantIncome': 'coapplicant_income',
            'LoanAmount': 'loan_amount',
            'Loan_Amount_Term': 'loan_amount_term',
            'Credit_History': 'credit_history',
            'Property_Area': 'property_area'
        }
        return mapping.get(feature_name, feature_name.lower())
    
    def _get_default_encoding(self, feature, value):
        """Get default encoding for categorical features"""
        default_encodings = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
            'Education': {'Graduate': 0, 'Not Graduate': 1},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        }
        
        if feature in default_encodings:
            str_value = str(value)
            return default_encodings[feature].get(str_value, 0)
        return 0
    
    def get_feature_names(self):
        return self.feature_names
    
    def is_loaded(self):
        return all([self.model, self.scaler, self.label_encoders, self.feature_names])