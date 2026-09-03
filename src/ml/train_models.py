# =============================================================================
# train_models.py - Multi-Model ML Training & Evaluation Pipeline
# =============================================================================
# This script trains 7 ML classifiers (Logistic Regression, Decision Tree,
# Random Forest, SVM, KNN, Gradient Boosting, XGBoost) on the loan dataset,
# evaluates each with accuracy/precision/recall/F1/ROC-AUC metrics, and saves
# the best-performing model as 'best_model.pkl' for deployment in Django.
# =============================================================================

import os
import sys
# Add project root directory to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.data.preprocessing import (
    load_raw_data, prepare_dataset, build_preprocessor_pipeline,
    get_feature_names, save_preprocessor
)
from src.ml.evaluate_models import evaluate_binary_classifier
from src.ml.hyperparameter_tuning import tune_random_forest, tune_xgboost

# Set paths
DATA_PATH = 'data/raw/loan_data.csv'
MODELS_DIR = 'models/ml'
PREPROC_DIR = 'models/preprocessing'
REPORTS_DIR = 'reports'
FIGURES_DIR = 'reports/figures'


def train_and_evaluate_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREPROC_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("--- 1. Loading Data & Preparing Features ---")
    df_raw = load_raw_data(DATA_PATH)
    X, y = prepare_dataset(df_raw, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 2. Fitting Preprocessing Pipeline ---")
    preprocessor = build_preprocessor_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # Save fitted preprocessor and feature names
    save_preprocessor(preprocessor, os.path.join(PREPROC_DIR, 'preprocessor.pkl'))
    joblib.dump(feature_names, os.path.join(PREPROC_DIR, 'feature_names.pkl'))
    print(f"Preprocessor saved to {PREPROC_DIR}/preprocessor.pkl ({len(feature_names)} features)")

    print("--- 3. Defining Base Models & Hyperparameter Tuning ---")

    # Base classifiers
    base_models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    # Hyperparameter tuning for Random Forest and XGBoost
    print("Tuning Random Forest...")
    rf_tuned, rf_params = tune_random_forest(X_train_trans, y_train)
    print(f"Random Forest Best Params: {rf_params}")

    print("Tuning XGBoost...")
    xgb_tuned, xgb_params = tune_xgboost(X_train_trans, y_train)
    print(f"XGBoost Best Params: {xgb_params}")

    all_models = {
        **base_models,
        'random_forest': rf_tuned,
        'xgboost': xgb_tuned
    }

    print("--- 4. Evaluating Models & Saving Pipelines ---")
    results = []
    trained_pipelines = {}

    plt.figure(figsize=(10, 8))
    plt.title('ROC Curves - Machine Learning Models', fontsize=14, fontweight='bold')

    for name, model in all_models.items():
        eval_metrics = evaluate_binary_classifier(
            model, X_train_trans, y_train, X_test_trans, y_test, model_name=name
        )
        results.append({
            'Model': name.replace('_', ' ').title(),
            'Accuracy': round(eval_metrics['accuracy'], 4),
            'Precision': round(eval_metrics['precision'], 4),
            'Recall': round(eval_metrics['recall'], 4),
            'F1': round(eval_metrics['f1_score'], 4),
            'ROC_AUC': round(eval_metrics['roc_auc'], 4),
            'CV_F1': round(eval_metrics['cv_f1_mean'], 4)
        })

        # Build a complete sklearn Pipeline that chains preprocessing + classifier.
        # This ensures the same transformations are applied at inference time,
        # preventing train-serve skew (a common production ML bug).
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Serialize each model pipeline to disk using joblib for Django deployment
        pipeline_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(full_pipeline, pipeline_path)
        trained_pipelines[name] = full_pipeline

        # Plot ROC Curve
        if 'y_prob' in eval_metrics:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, eval_metrics['y_prob'])
            plt.plot(fpr, tpr, label=f"{name.title()} (AUC = {eval_metrics['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    # Save summary dataframe
    df_results = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    csv_report_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    df_results.to_csv(csv_report_path, index=False)
    print("\n--- Model Comparison Summary ---")
    print(df_results.to_string(index=False))

    # Identify Best Model: select the model with the highest F1 score
    # (F1 balances precision and recall, important for imbalanced loan data)
    best_row = df_results.iloc[0]
    best_name_clean = best_row['Model'].lower().replace(' ', '_')
    best_pipeline = trained_pipelines[best_name_clean]
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    # Save the champion model separately as 'best_model.pkl' for Django to load
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[BEST ML MODEL] Selected: {best_row['Model']} (F1={best_row['F1']}, ROC-AUC={best_row['ROC_AUC']})")
    print(f"Saved best model to {best_model_path}")

    # Generate Feature Importance plot for tree-based best model
    best_classifier = best_pipeline.named_steps['classifier']
    if hasattr(best_classifier, 'feature_importances_'):
        importances = best_classifier.feature_importances_
        indices = np.argsort(importances)[-12:]  # Top 12 features
        plt.figure(figsize=(10, 6))
        plt.title(f'Top Feature Importances ({best_row["Model"]})', fontsize=14, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], align='center', color='#4f46e5')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=300)
        plt.close()
        print(f"Feature importance plot saved to {FIGURES_DIR}/feature_importance.png")

    return df_results


if __name__ == '__main__':
    train_and_evaluate_all()
