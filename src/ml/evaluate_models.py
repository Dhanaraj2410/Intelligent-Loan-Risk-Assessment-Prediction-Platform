import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score


def evaluate_binary_classifier(model, X_train, y_train, X_test, y_test, model_name="Model") -> dict:
    """
    Evaluates a binary classification model on test data and cross-validation.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Check if predict_proba is supported
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # Normalize to [0, 1] range for AUC estimation
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-5)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # 5-Fold Cross Validation F1 score on training set
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        cv_f1 = float(np.mean(cv_scores))
    except Exception:
        cv_f1 = f1

    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)'])

    return {
        'model_name': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'cv_f1_mean': cv_f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_text,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
