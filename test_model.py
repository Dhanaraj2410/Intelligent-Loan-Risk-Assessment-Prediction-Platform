# test_model.py
import pickle
import numpy as np
from pathlib import Path

def test_model():
    """Test the model with sample data"""
    ml_models_dir = Path('ml_models')
    
    # Load model files
    with open(ml_models_dir / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(ml_models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(ml_models_dir / 'label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(ml_models_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Feature names:", feature_names)
    print("Label encoders:", label_encoders.keys())
    
    # Test sample data
    sample_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 1000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Urban'
    }
    
    # Encode categorical variables
    encoded_data = []
    for feature in feature_names:
        value = sample_data.get(feature)
        if feature in label_encoders:
            try:
                encoded = label_encoders[feature].transform([str(value)])[0]
                print(f"{feature}: {value} -> {encoded}")
                encoded_data.append(encoded)
            except Exception as e:
                print(f"Error encoding {feature}: {e}")
                encoded_data.append(0)
        else:
            encoded_data.append(float(value))
    
    print("\nEncoded data:", encoded_data)
    
    # Reshape and scale
    input_array = np.array(encoded_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    
    print("Scaled input:", scaled_input)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    print(f"\nPrediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    result = 'Approved' if prediction == 1 else 'Rejected'
    prob = probabilities[1] if prediction == 1 else probabilities[0]
    
    print(f"Result: {result}")
    print(f"Confidence: {prob * 100:.2f}%")

if __name__ == "__main__":
    test_model()