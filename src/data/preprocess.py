import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Basic cleaning for Telco Churn.
    - trim columns names
    - drom obvious ID columns
    - fix TotalCharges to numeric
    - map target column to 0/1
    - simple NA handling
    """
    df.columns = df.columns.str.strip()
    target_col = target_col.strip()

    df = df.drop(columns=['customer_ID'], errors="ignore")

    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].str.strip().str.lower().map({'yes' : 1, 'no' : 0})

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    return df