import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path: Path):
    return joblib.load(model_path)


def predict(model, X: pd.DataFrame, threshold: float = 0.35):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities