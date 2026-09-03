from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def tune_random_forest(X_train, y_train, cv=3, n_iter=10, random_state=42):
    """
    Hyperparameter tuning for Random Forest Classifier.
    """
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='f1', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train, cv=3, n_iter=10, random_state=42):
    """
    Hyperparameter tuning for XGBoost Classifier.
    """
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=random_state, eval_metric='logloss')
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='f1', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
