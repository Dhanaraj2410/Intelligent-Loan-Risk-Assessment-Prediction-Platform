# =============================================================================
# preprocessing.py - Scikit-learn Data Preprocessing Pipeline
# =============================================================================
# Builds the end-to-end preprocessing pipeline that transforms raw loan data
# into model-ready feature vectors. Handles missing values (imputation),
# categorical encoding (OneHotEncoder), and numerical scaling (StandardScaler).
# =============================================================================

import os
import sys
# Add project root directory to PYTHONPATH so 'src' package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.data.feature_engineering import add_engineered_features

# Numerical columns: includes both raw features and engineered features
# (TotalIncome, LoanIncomeRatio, etc. created by feature_engineering.py)
NUMERICAL_COLS = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'TotalIncome', 'LoanIncomeRatio', 'ApplicantIncomeLog', 'LoanAmountLog',
    'LoanTermYears', 'EstimatedEMI'
]

# Categorical columns: will be one-hot encoded by the preprocessing pipeline
CATEGORICAL_COLS = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'Credit_History', 'Property_Area'
]

# Target variable column name in the training dataset (Y=Approved, N=Rejected)
TARGET_COL = 'Loan_Status'
# Columns to drop before training (Loan_ID is just an identifier, not a feature)
DROP_COLS = ['Loan_ID']


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)


def prepare_dataset(df: pd.DataFrame, is_training: bool = True):
    """
    Applies feature engineering, separates features X and target y.
    """
    df = df.copy()

    # Drop Loan_ID if present
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Apply feature engineering
    df = add_engineered_features(df)

    if is_training and TARGET_COL in df.columns:
        # Drop rows where target is missing
        df = df.dropna(subset=[TARGET_COL])
        # Map target Y -> 1, N -> 0
        y = df[TARGET_COL].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).astype(int)
        X = df.drop(columns=[TARGET_COL])
        return X, y
    else:
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        return df


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Builds unified Scikit-learn ColumnTransformer pipeline for numerical and categorical features.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Retrieves feature names after OneHotEncoding and scaling."""
    feature_names = []
    
    # Numerical features
    feature_names.extend(NUMERICAL_COLS)
    
    # Categorical features from OneHotEncoder
    cat_transformer = preprocessor.named_transformers_['cat']
    encoder = cat_transformer.named_steps['encoder']
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS)
    feature_names.extend(cat_feature_names.tolist())
    
    return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str):
    """Serializes the fitted preprocessor to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str) -> ColumnTransformer:
    """Loads a preprocessor pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
    return joblib.load(filepath)
