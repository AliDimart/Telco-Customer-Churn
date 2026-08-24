import pandas as pd
from build_preprocessor import build_preprocessor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params):

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))
    ])

    model.fit(X_train, y_train)

    return model