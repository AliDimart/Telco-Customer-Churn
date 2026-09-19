import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

def find_threshold(y_train: pd.Series, y_proba: pd.Series) -> float:

    best_threshold = 0.5
    best_score = 0

    print("Threshold tuning for XGBoost:")

    print(f"{'Thresh':<8}{'Prec_1':<8}{'Rec_1':<8}{'F_beta':<8}")
    for threshold in np.arange(0.05, 0.96, 0.01):
        
        y_pred = (y_proba >= threshold).astype(int)

        # We give more weight to Recall score
        f_beta = fbeta_score(y_train, y_pred, beta=2)

        #precision = precision_score(y_train, y_pred)
        #recall = recall_score(y_train, y_pred)
        #print(f"{threshold:<8}{precision:<8.3f}{recall:<8.3f}{f_beta:<8.3f}")

        if f_beta > best_score:
            best_score = f_beta
            best_threshold = threshold
    
    print(f"Choosed threshold: {best_threshold}")

    return best_threshold
