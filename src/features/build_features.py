import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Builds preprocessor, which prepares data for model:
    - removes redundant categorical information
    - fills NA with median and most_frequent
    - encodes categorical columns with OneHotEncoder  
    """
    categorical_cols = X_train.select_dtypes(include='object').columns
    numeric_cols = X_train.select_dtypes(include=['number','bool']).columns

    feature_engineering_transformer = FunctionTransformer(feature_engineering)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                SimpleImputer(strategy='median'),
                numeric_cols
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
                ]),
                categorical_cols
            )
        ]
    )

    return Pipeline([
        ("feature_engineering", feature_engineering_transformer),
        ("preprocessor", preprocessor)
    ])


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes redundant `No internet service` categories
    from features whose values are determined by InternetService.
    '''
    df = df.copy()

    # These columns contain the same information of InternetService_No
    no_internet_cols = [
        "OnlineSecurity",
        "StreamingMovies",
        "OnlineBackup",
        "TechSupport",
        "DeviceProtection",
        "StreamingTV",
    ]

    for col in no_internet_cols:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")

    return df