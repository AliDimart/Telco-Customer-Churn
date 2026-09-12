import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# === Fix import path for local modules ===
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.load_data import load_data   
from src.inference.inference import load_model
from src.models.find_threshold import find_threshold

def main():

    processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
    df = load_data(processed_path)

    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    
    model_path = project_root / "models" / "churn_pipeline.joblib"
    model = load_model(model_path)

    find_threshold(model, X_train, y_train)


if __name__ == '__main__':

    main()