import sys
import argparse
from pathlib import Path
import json

from sklearn.model_selection import train_test_split

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Local modules - Core pipeline components
from src.data.load_data import load_data           # Data loading with error handling
from src.models.tune import tune_model             # Find best params


def main(args):
    """
    Pipeline function that Tunes XGBoost hyperparameters with Optuna

    Runs sequentially: 
        load → tune → evaluate → log
    """
    print("🔄 Loading preprocessed data...")
    df = load_data(args.input)

    target = args.target

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=args.test_size, random_state=101, stratify=y)

    print("🔎 Starting hyperparameter tuning...")
    best_params = tune_model(X_train, y_train, threshold=args.threshold, n_trials=args.n_trials)

    output_path = project_root / "artifacts" / "best_params.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print(f"✅ Best parameters saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with Optuna")

    parser.add_argument("--input", type=str, default="data/processed/telco_churn_processed.csv", help="Path to CSV")
    parser.add_argument("--target", type=str, default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials",)
    parser.add_argument("--experiment", type=str, default="Telco Churn")
    parser.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = parser.parse_args()

    main(args)