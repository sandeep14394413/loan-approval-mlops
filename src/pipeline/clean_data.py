"""
Stage 2 — Data Cleaning

Steps:
  - Drop the ID column (Applicant_ID)
  - Drop rows where the target column is null
  - Drop exact duplicate rows
  - Strip whitespace from string columns
  - Ensure target column is integer 0/1
  - Keep only feature + target columns
  - Save cleaned dataset to data/processed/cleaned.csv

Outputs:
  data/processed/cleaned.csv
"""

from pathlib import Path

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMN,
)

OUT_PATH = Path("data/processed/cleaned.csv")


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    original_rows = len(df)
    print(f"Loaded {original_rows} rows from {RAW_DATA_PATH}")
    print(f"Columns: {list(df.columns)}")

    # 1. Drop ID column — not a feature
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        print(f"  Dropped ID column: {ID_COLUMN}")

    # 2. Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 3. Drop rows with null target
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped_target = before - len(df)
    if dropped_target:
        print(f"  Dropped {dropped_target} rows with null target.")

    # 4. Ensure target is integer 0/1
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # 5. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before - len(df)
    if dropped_dupes:
        print(f"  Dropped {dropped_dupes} duplicate rows.")

    # 6. Keep only relevant columns
    keep_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df = df[keep_cols]

    final_rows = len(df)
    print(f"Cleaned dataset: {final_rows} rows ({original_rows - final_rows} total removed).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data to: {OUT_PATH}")
    return df


if __name__ == "__main__":
    clean()
