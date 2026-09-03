# =============================================================================
# hyperparameter_tuning.py - Automated Hyperparameter Optimization
# =============================================================================
# Uses RandomizedSearchCV to find optimal hyperparameters for Random Forest
# and XGBoost classifiers. RandomizedSearchCV is preferred over GridSearchCV
# for efficiency when the search space is large.
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def tune_random_forest(X_train, y_train, cv=3, n_iter=10, random_state=42):
    """
    Hyperparameter tuning for Random Forest Classifier.
    Searches over tree depth, number of estimators, and leaf/split thresholds.
    """
    # Define the hyperparameter search space for Random Forest
    param_dist = {
        'n_estimators': [50, 100, 200],       # Number of trees in the forest
        'max_depth': [None, 5, 10, 15],        # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],         # Minimum samples required at a leaf node
        'bootstrap': [True, False]             # Whether to use bootstrap sampling
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    # RandomizedSearchCV samples n_iter combinations from param_dist
    # and evaluates each using cv-fold cross-validation scored by F1
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='f1', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train, cv=3, n_iter=10, random_state=42):
    """
    Hyperparameter tuning for XGBoost Classifier.
    Searches over learning rate, tree depth, and subsampling ratios.
    """
    # Define the hyperparameter search space for XGBoost
    param_dist = {
        'n_estimators': [50, 100, 200],        # Number of boosting rounds
        'max_depth': [3, 5, 7],                # Maximum depth per tree
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage (eta)
        'subsample': [0.6, 0.8, 1.0],          # Row subsampling ratio per tree
        'colsample_bytree': [0.6, 0.8, 1.0]    # Feature subsampling ratio per tree
    }
    
    xgb = XGBClassifier(random_state=random_state, eval_metric='logloss')
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='f1', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
