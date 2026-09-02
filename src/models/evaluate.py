import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)

def evaluate_model(model : Pipeline, X_test : pd.DataFrame, y_test : pd.Series, threshold : float):
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    # Generate predictions 
    proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
    
    # Apply classification threshold (default: 0.35, optimized for churn detection)
    # Lower threshold = more sensitive to churn (higher recall, lower precision)
    y_pred = (proba >= threshold).astype(int)

    # === CRITICAL: Log Evaluation Metrics to MLflow ===
    # These metrics are essential for model comparison and monitoring
    precision = precision_score(y_test, y_pred, zero_division=0)    # Of predicted churners, how many actually churned?
    recall = recall_score(y_test, y_pred, zero_division=0)          # Of actual churners, how many did we catch?
    f1 = f1_score(y_test, y_pred, zero_division=0)                  # Harmonic mean of precision and recall
    roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve (threshold-independent)
    
    # Log all metrics for experiment tracking
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall) 
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"🎯 Model Performance:")
    print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

    print(f"\n Detailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))