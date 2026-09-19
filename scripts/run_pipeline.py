import sys
import argparse
import mlflow
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_predict

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Local modules - Core pipeline components
from src.data.load_data import load_data                   # Data loading with error handling
from src.data.preprocess import preprocess_data            # Basic data cleaning
from src.utils.validate_data import validate_telco_data    # Data quality validation
from src.models.train import train_model, build_model      # Model training
from src.models.evaluate import evaluate_model             # Model evaluation
from src.models.find_threshold import find_threshold       # Threshold tunning


def load_best_params(path: Path) -> dict:
    if not path.exists():
        print("ℹ️ Optuna parameters not found. Using default model parameters.")
        return None

    with open(path, "r") as file:
        params = json.load(file)

    print("✅ Optuna parameters loaded.")
    return params


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    Runs sequentially: 
        load → validate → preprocess → split → train → choose threshold → evaluate → log
    """
    mlflow_uri = args.mlflow_uri or f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment) # Creates experiment if doesn't exist

    with mlflow.start_run():
        # === Log hyperparameters and configuration ===

        mlflow.log_param("model", "xgboost")    # Model type for comparison  

        # === STAGE 1: Data Loading & Validation ===

        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === Data Quality Validation ===
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

        # === STAGE 3: Data Split ===

        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # === STAGE 4: Creating Pipeline ===

        # Trying to read params tuned by optuna
        params_path = project_root / "artifacts" / "best_params.json"
        params = load_best_params(params_path)

        # Creating preprocessor + xgboost
        pipeline = build_model(X_train, y_train, params)

        # === STAGE 5: THRESHOLD tunning and evaluating the model ===

        out_of_bag_proba = cross_val_predict(pipeline, X_train, y_train, cv=5, method='predict_proba')[:, 1]

        threshold = find_threshold(y_train, out_of_bag_proba)
        mlflow.log_param("threshold", threshold)    # Saving threshold

        # Model training
        pipeline.fit(X_train, y_train)

        evaluate_model(pipeline, X_test, y_test, threshold)

        # === STAGE 6: Model Serialization and Logging ===

        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            pipeline,
            name="churn_pipeline",  # This creates a 'model/' folder in MLflow run artifacts
            serialization_format="cloudpickle"
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # Saving model 
        model_path = project_root / "models" / "churn_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_artifact = {
            "model": pipeline,
            "threshold": threshold
        }

        joblib.dump(model_artifact, model_path)

        print(f"✅ Model saved to {model_path}")


if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")

    p.add_argument("--input", type=str, default="data/raw/Telco-Customer-Churn.csv", help="Path to CSV")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)