import sys
import argparse
from pathlib import Path

from xgboost import XGBClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    # Data loading with error handling
from src.preprocessing.preprocess import preprocess_data            # Basic data cleaning
from src.preprocessing.build_preprocessor import build_preprocessor     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data    # Data quality validation

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    Runs sequentially: 
        load → validate → preprocess → feature engineering
    """


if __name__ == 'main':

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
