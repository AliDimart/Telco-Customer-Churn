import time
import mlflow
import pandas as pd

from src.features.build_features import build_preprocessor     # Feature engineering (CRITICAL for model performance)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params : dict = None) -> Pipeline:
    """
    Trains Pipeline that contains Preprocessor + XGBoost model and logs with MLflow.

    Args:
        X_train (pd.DataFrame): Feature dataset
        y_train (pd.Series): Target column
    """
    default_params = {
        # Tree structure parameters
        'n_estimators' : 301,        # Number of trees
        'learning_rate' : 0.034,     # Step size shrinkage   
        'max_depth' : 7,             # Maximum tree depth 
        "min_child_weight": 9,       # Minimum weight of childs to create a new leaf 
        
        # Regularization parameters
        'subsample' : 0.95,          # Sample ratio of training instances
        'colsample_bytree' : 0.98,   # Sample ratio of features for each tree
        "gamma": 4.36,               # Minimum loss function reduction for creating a new leaf
        "reg_alpha": 0.063,          # L1-regularization 
        "reg_lambda": 0.087          # L2-regularization   
    }

    if params is None:
        params = default_params

    # Numerical imputation + categorical imputation and one-hot encoding
    preprocessor = build_preprocessor(X_train)

    # === CRITICAL: Handle Class Imbalance ===
    # Calculate scale_pos_weight to handle imbalanced dataset
    # This tells XGBoost to give more weight to the minority class (churners)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # === STAGE 4: Model Training ===
    print("🤖 Training XGBoost model...")
    
    # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
    # In production, consider using hyperparameter optimization tools like Optuna
    model = XGBClassifier(
        **params,

        # Performance parameters
        n_jobs=-1,              # Use all CPU cores
        random_state=42,        # Reproducible results

        # ESSENTIAL: Handle class imbalance
        scale_pos_weight=scale_pos_weight,  # Weight for positive class (churners)
        eval_metric="logloss"
    )
    
    # Create pipeline to connect Preprocessor + XGBoost
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    
    # === Train Model and Track Training Time ===
    print("🤖 Training XGBoost pipeline...")

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    print(f"✅ Model trained in {train_time:.2f} seconds")

    mlflow.log_params(params)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("train_time", train_time)

    return pipeline