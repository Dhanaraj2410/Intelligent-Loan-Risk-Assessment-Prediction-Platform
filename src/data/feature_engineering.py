# =============================================================================
# feature_engineering.py - Business-Oriented Feature Transformations
# =============================================================================
# Creates derived financial features from raw loan application data.
# These engineered features improve model accuracy by capturing domain
# knowledge about loan repayment capacity, income stability, and risk ratios.
# =============================================================================

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies business-oriented feature engineering to loan application data.
    
    Features engineered:
    - TotalIncome: ApplicantIncome + CoapplicantIncome
    - LoanIncomeRatio: LoanAmount / (TotalIncome + 1)
    - ApplicantIncomeLog: log1p(ApplicantIncome)
    - LoanAmountLog: log1p(LoanAmount)
    - LoanTermYears: Loan_Amount_Term / 12.0
    - EstimatedEMI: (LoanAmount * 1000) / (Loan_Amount_Term + 1e-5)
    """
    df = df.copy()

    # Numeric conversions if needed
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Total Income
    if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
        df['TotalIncome'] = df['ApplicantIncome'].fillna(0) + df['CoapplicantIncome'].fillna(0)
    
    # Loan-to-Income Ratio
    if 'LoanAmount' in df.columns and 'TotalIncome' in df.columns:
        # Note: LoanAmount is in thousands
        df['LoanIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1.0)

    # Log transformations: reduce skewness in income and loan amount distributions.
    # log1p(x) = log(1+x) handles zero values safely without producing -inf.
    if 'ApplicantIncome' in df.columns:
        df['ApplicantIncomeLog'] = np.log1p(df['ApplicantIncome'].clip(lower=0))
    if 'LoanAmount' in df.columns:
        df['LoanAmountLog'] = np.log1p(df['LoanAmount'].clip(lower=0))

    # Loan Term Years
    if 'Loan_Amount_Term' in df.columns:
        df['LoanTermYears'] = df['Loan_Amount_Term'] / 12.0
        # Estimated monthly installment proxy (EMI)
        df['EstimatedEMI'] = (df['LoanAmount'] * 1000.0) / (df['Loan_Amount_Term'] + 1e-5)

    return df
