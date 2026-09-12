import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score

from src.features.build_features import build_preprocessor

def tune_model(X_train : pd.DataFrame, y_train : pd.Series, threshold : float = 0.5, n_trials: int = 30) -> dict:
    """
    Tunes an XGBoost model using Optuna.

    Args:
        X_train (pd.DataFrame): Feature dataset.
        y_train (pd.Series): Target column.
        threshold: Classification threshold.
        n_trials: Number of Optuna trials.
    Returns: 
        Best hyperparameters found by Optuna.
    """
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def recall_with_threshold(estimator, X_val : pd.DataFrame, y_val : pd.Series) -> float:
        """
        Implements threshold to recall_score to focus on positive class.
        """
        pred = estimator.predict_proba(X_val)[:, 1]

        y_pred = (pred >= threshold).astype(int)

        return recall_score(y_val, y_pred, zero_division=0)


    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators" : trial.suggest_int("n_estimators", 200, 800),
            "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth" : trial.suggest_int("max_depth", 3, 10),
            "min_child_weight" : trial.suggest_int( "min_child_weight", 1, 10),
            "subsample" : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma" : trial.suggest_float( "gamma", 0.0, 5.0), 
            "reg_alpha" : trial.suggest_float( "reg_alpha", 1e-8, 10.0, log=True), 
            "reg_lambda" : trial.suggest_float( "reg_lambda", 1e-8, 10.0, log=True),

            "scale_pos_weight" : scale_pos_weight,
            "random_state" : 42,
            "n_jobs" : -1,
            "eval_metric" : "logloss"
        }

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", XGBClassifier(**params)),
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=recall_with_threshold, n_jobs=1)

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params