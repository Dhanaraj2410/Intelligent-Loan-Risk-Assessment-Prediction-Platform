# Intelligent Loan Risk Assessment & Prediction Platform

> **An End-to-End Machine Learning & Deep Learning SaaS Application**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.1-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-2A9D8F?style=for-the-badge)](https://xgboost.readthedocs.io)
[![PyTorch/Keras](https://img.shields.io/badge/PyTorch%2FKeras-ANN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

---

## 📌 Executive Overview

The **Intelligent Loan Risk Assessment & Prediction Platform** is a production-grade machine learning and deep learning application designed to evaluate applicant financial profiles, calculate loan approval probabilities, estimate 0–100 risk scores, and categorize risk levels (**LOW**, **MEDIUM**, **HIGH**).

Featuring a **Simple Math Loan Evaluation** engine, the platform breaks down exact financial metrics in Indian Rupees (₹) (Total Household Income, Loan-to-Income Ratio, Bank EMI via standard banking formula at 10.5% p.a.), feature transformations, and model decision mechanics across 8 algorithms.

---

## 🛢️ MySQL Database Integration

The system connects directly to **MySQL Database** (`loan_prediction`):

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'loan_prediction',
        'USER': 'root',
        'PASSWORD': 'Dhanaraj2410',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
            'charset': 'utf8mb4',
        },
    }
}
```

---

## 🧮 Simple Math Loan Approval & Rejection Logic

The result page presents an easy-to-understand 4-step financial breakdown:

1. **Total Monthly Household Income**: $\text{Applicant Income} + \text{Co-applicant Income} = ₹65,000 / \text{month}$ ($\text{Annual Income} = ₹7.80\text{ Lakhs}$).
2. **Loan Amount vs Income Ratio**: $\frac{\text{Requested Loan}}{\text{Annual Income}} = \mathbf{1.92x}$ (*Safe Ratio < 3x*).
3. **Monthly Bank EMI Payment**: Calculated using standard banking formula at **10.5% p.a. interest rate** ($\text{EMI} = ₹13,721 / \text{month}$, representing $21.1\%$ of income).
4. **Credit Record & AI Voting**: Checks CIBIL credit record and summarizes votes across 8 AI models.

---

## 📈 Empirical Model Evaluation & Comparison

Models are trained on an 80/20 train-test split using 5-Fold Cross-Validation.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8699** | 0.8485 | 0.9882 | **0.9130** | 0.8613 | 0.8560 |
| **Random Forest (Tuned)** | **0.8699** | 0.8485 | 0.9882 | **0.9130** | 0.8505 | 0.8681 |
| **XGBoost (Tuned)** | 0.8537 | 0.8317 | 0.9882 | 0.9032 | **0.8788** | 0.8699 |
| **SVM** | 0.8455 | 0.8235 | 0.9882 | 0.8984 | 0.8690 | 0.8662 |
| **ANN (Deep Learning)** | 0.8455 | 0.8750 | 0.9059 | 0.8902 | 0.8480 | 0.8902 |
| **Gradient Boosting** | 0.8374 | 0.8495 | 0.9294 | 0.8876 | 0.8080 | 0.8533 |
| **Decision Tree** | 0.8211 | 0.8182 | 0.9529 | 0.8804 | 0.7368 | 0.8626 |
| **KNN** | 0.7886 | 0.8105 | 0.9059 | 0.8556 | 0.8150 | 0.8250 |

---

## 🛠️ Execution Guide

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Apply Migrations to MySQL
python manage.py makemigrations
python manage.py migrate

# 3. Launch Dev Server
python manage.py runserver 8005
```

Visit `http://127.0.0.1:8005/` in your browser.

---

## 📁 Project Directory Structure

```
loan_prediction/
├── data/raw/                  # Raw training CSV dataset
├── models/
│   ├── ml/                    # Serialized sklearn pipelines (.pkl)
│   ├── dl/                    # Deep Learning ANN model (.keras)
│   └── preprocessing/         # Fitted preprocessor & feature names
├── src/
│   ├── data/                  # Preprocessing & feature engineering modules
│   ├── ml/                    # ML model training, evaluation & tuning
│   └── dl/                    # Deep Learning ANN training & inference
├── predictor/                 # Django app (models, views, forms, urls)
├── templates/predictor/       # HTML templates (landing, dashboard, predict, result)
├── static/                    # CSS stylesheets & JavaScript files
├── tests/                     # Unit tests (models, forms, API)
├── reports/                   # Model comparison CSV & visualization figures
├── notebooks/                 # Jupyter notebooks for EDA & training
├── scripts/                   # Utility scripts (notebook generation)
├── manage.py                  # Django management command entry point
├── requirements.txt           # Python package dependencies
└── Procfile                   # Gunicorn deployment configuration
```
