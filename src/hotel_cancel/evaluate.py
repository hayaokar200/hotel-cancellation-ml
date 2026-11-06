"""
Functions to evaluate the model performance.
"""

import pandas as pd

import numpy as np

from hotel_cancel import config


def get_eval_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Calculate evaluation metrics for the model.
    """
    ...


def get_best_threshold( y_true: pd.Series, y_prob: np.ndarray,thresholds: list[float] = config.THRESHOLD_CANDIDATES) -> float:
    """
    Find the best threshold for classification, loop theogh the thershold, call getEvalMetrics, and return the threshold with the best F1 score.
    """
    ...