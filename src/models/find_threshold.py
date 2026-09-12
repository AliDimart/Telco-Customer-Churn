import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def find_threshold(model, X_train: pd.DataFrame, y_train: pd.Series) -> float:

    best_threshold = 0.5
    best_score = 0

    print("Threshold tuning for XGBoost:")

    proba = model.predict_proba(X_train)[:,1]

    print(f"{'Thresh':<8}{'Prec_1':<8}{'Rec_1':<8}{'F1_1':<8}")
    for threshold in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        
        y_pred = (proba >= threshold).astype(int)
        
        f1 = f1_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred)
        rec = recall_score(y_train, y_pred)

        print(f"{threshold:<8}{prec:<8.3f}{rec:<8.3f}{f1:<8.3f}")

        if f1 > best_score:
            best_score = f1
            best_threshold = threshold
    
    print(f"Choosed threshold: {best_threshold}")

    return best_threshold
