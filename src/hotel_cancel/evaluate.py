"""
Functions to evaluate the model performance.
"""

import pandas as pd

import numpy as np

from hotel_cancel import config
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,average_precision_score


def get_eval_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Calculate evaluation metrics for the model at the given threshold.

    Returns a dict with keys: "f1", "precision", "recall".
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold":threshold,
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }


def get_best_threshold(y_true: pd.Series, y_prob: np.ndarray, thresholds: list[float] = config.THRESHOLD_CANDIDATES) -> float:
    """
    Find the best classification threshold on VALIDATION data: loop through the
    thresholds, call get_eval_metrics, and return the threshold with the best F1 score.
    """
    results = []
    for threshold in thresholds:
        results.append(get_eval_metrics(y_true,y_prob,threshold))
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    return best_row["threshold"]
