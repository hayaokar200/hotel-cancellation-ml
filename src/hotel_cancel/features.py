"""
This module contains functions for feature engineering and preprocessing of the hotel cancellation dataset.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer

def get_numerical_features(df: pd.DataFrame) -> list[str]:
    """
    Get numerical features from the DataFrame.
    """
    ...


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    """
    Get categorical features from the DataFrame.
    """
    ...


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing, I will use the two functions above.
    """
    ...
