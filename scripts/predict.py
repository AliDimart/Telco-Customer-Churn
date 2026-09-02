import sys
import pandas as pd
from pathlib import Path

# === Fix import path for local modules ===
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.inference import load_model, predict


model_path = project_root / "models" / "churn_pipeline.joblib"

model = load_model(model_path)

data = pd.read_csv(project_root / "data" / "raw" / "new_customers.csv")

predictions, probabilities = predict(model, data, threshold=0.35)

data["churn_probability"] = probabilities
data["churn_prediction"] = predictions

print(data[["churn_probability", "churn_prediction"]])