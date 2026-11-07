"""
This module contains functions for feature engineering and preprocessing of the hotel cancellation dataset.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

def get_numerical_features(df: pd.DataFrame) -> list[str]:
    """
    Get numerical features from the DataFrame.
    """
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    return numerical_cols


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    """
    Get categorical features from the DataFrame.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing, I will use the two functions above.
    """
    numerical_transformer = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy="constant",fill_value="Unknown")),
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numerical_transformer,get_numerical_features(X)),
            ("cat",categorical_transformer,get_categorical_features(X))
        ]
    )
    return preprocessor
