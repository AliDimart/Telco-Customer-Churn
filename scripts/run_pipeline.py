import sys
import argparse
import mlflow
import joblib
import json
import time
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.features.build_features import build_preprocessor     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data    # Data quality validation

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    Runs sequentially: 
        load → validate → preprocess → split → train → evaluate → log
    """
    mlflow_uri = args.mlflow_uri or f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment) # Creates experiment if doesn't exist

    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")           # Model type for comparison
        mlflow.log_param("threshold", args.threshold)   # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("🔍 Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))  # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            mlflow.log_text(json.dumps(failed, indent=4), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df)  # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Data Split and Preprocessing ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=101, stratify=y)
        
        # Numerical imputation + categorical imputation and one-hot encoding
        preprocessor = build_preprocessor(X_train)

        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 4: Model Training ===
        print("🤖 Training XGBoost model...")
        
        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model = XGBClassifier(
            # Tree structure parameters
            n_estimators=301,        # Number of trees (OPTIMIZED)
            learning_rate=0.034,     # Step size shrinkage (OPTIMIZED)  
            max_depth=7,            # Maximum tree depth (OPTIMIZED)
            
            # Regularization parameters
            subsample=0.95,         # Sample ratio of training instances
            colsample_bytree=0.98,  # Sample ratio of features for each tree
            
            # Performance parameters
            n_jobs=-1,              # Use all CPU cores
            random_state=42,        # Reproducible results
            eval_metric="logloss",  # Evaluation metric
            
            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)
        )
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        
        # === Train Model and Track Training Time ===
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("📊 Evaluating model performance...")
        
        # Generate predictions and track inference time
        t1 = time.time()
        proba = pipeline.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
        
        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

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

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            pipeline, 
            name="churn_pipeline"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")

    p.add_argument("--input", type=str, default="data/raw/Telco-Customer-Churn.csv", help="path to CSV")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)


"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py 

"""
