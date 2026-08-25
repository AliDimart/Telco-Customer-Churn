import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Builds preprocessor, which prepares data for model:
    - fills NA with median and most_frequent
    - encodes categorical columns with OneHotEncoder  
    """
    categorical_cols = X_train.select_dtypes(include='object').columns
    numeric_cols = X_train.select_dtypes(include='number').columns

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

    return preprocessor