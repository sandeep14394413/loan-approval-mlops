"""
Preprocessing utilities.

Builds a scikit-learn ColumnTransformer that:
  - Imputes and scales numeric features
  - Imputes and one-hot encodes categorical features

Also exposes helpers to load data and extract X / y.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
)


def load_data(path) -> pd.DataFrame:
    """Load CSV and drop the ID column if present."""
    df = pd.read_csv(path)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the target column is integer 0/1.
    Drops rows where the target is null.
    """
    df = df.copy()
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer:
      numeric  -> SimpleImputer(median) -> StandardScaler
      category -> SimpleImputer(most_frequent) -> OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Only include columns that exist in the DataFrame at fit time
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_features_and_target(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target series y."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available]
    y = df[TARGET_COLUMN]
    return X, y
