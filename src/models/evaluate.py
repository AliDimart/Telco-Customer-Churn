import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, recall_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))