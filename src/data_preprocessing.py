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
)


def load_data(path):
    """Load CSV dataset into a DataFrame."""
    return pd.read_csv(path)


def clean_target(df):
    """
    Encode the target column:
      Y -> 1 (Approved)
      N -> 0 (Rejected)
    Rows with missing target are dropped.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Y": 1, "N": 0})
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def build_preprocessor():
    """
    Build a ColumnTransformer that:
    - Imputes and scales numeric features
    - Imputes and one-hot encodes categorical features
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_features_and_target(df):
    """Split DataFrame into features X and target y."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
