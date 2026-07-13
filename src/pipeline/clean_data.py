"""
Stage 2 — Data Cleaning

Steps:
  - Drop duplicate rows
  - Drop rows where the target column is null
  - Strip whitespace from string columns
  - Normalize target column values to Y/N (handles lowercase)
  - Save cleaned dataset to data/processed/cleaned.csv

Outputs:
  data/processed/cleaned.csv
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS

OUT_PATH = Path("data/processed/cleaned.csv")


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    original_rows = len(df)
    print(f"Loaded {original_rows} rows from {RAW_DATA_PATH}")

    # 1. Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Normalize target values (handle lowercase like 'y', 'n')
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].str.upper()

    # 3. Drop rows with null target
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped_target = before - len(df)
    if dropped_target:
        print(f"  Dropped {dropped_target} rows with null target.")

    # 4. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before - len(df)
    if dropped_dupes:
        print(f"  Dropped {dropped_dupes} duplicate rows.")

    # 5. Keep only relevant columns
    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols]

    final_rows = len(df)
    print(f"Cleaned dataset: {final_rows} rows ({original_rows - final_rows} total removed).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data to: {OUT_PATH}")
    return df


if __name__ == "__main__":
    clean()
