"""
Stage 1 — Data Validation

Checks the RAW CSV only — engineered feature columns (income_per_exp_year,
age_minus_experience, employment_stability, residence_stability) are created
at training time by engineer_features() and must NOT be expected here.

Checks:
  - CSV file exists and is non-empty
  - Required RAW columns are present
  - Minimum row count threshold
  - Target column contains only expected values (0 and 1)
  - Null percentage per column (fails if > 50%)

Outputs:
  reports/validation_report.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN, RAW_FEATURE_COLUMNS

REPORT_PATH  = Path("reports/validation_report.json")
MIN_ROWS     = 100
MAX_NULL_PCT = 0.50


def validate() -> dict:
    report = {"passed": True, "errors": [], "warnings": [], "stats": {}}

    # 1. File existence
    if not RAW_DATA_PATH.exists():
        report["passed"] = False
        report["errors"].append(f"Dataset not found: {RAW_DATA_PATH}")
        return report

    df = pd.read_csv(RAW_DATA_PATH)

    # 2. Empty file guard
    if df.empty or len(df.columns) == 0:
        report["passed"] = False
        report["errors"].append(f"Dataset is empty: {RAW_DATA_PATH}")
        return report

    report["stats"]["total_rows"]    = len(df)
    report["stats"]["total_columns"] = len(df.columns)
    report["stats"]["columns"]       = list(df.columns)

    # 3. Required RAW columns check (no engineered columns here)
    required      = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols  = [c for c in required if c not in df.columns]
    if missing_cols:
        report["passed"] = False
        report["errors"].append(f"Missing required columns: {missing_cols}")

    # 4. Minimum row count
    if len(df) < MIN_ROWS:
        report["passed"] = False
        report["errors"].append(
            f"Dataset has only {len(df)} rows. Minimum required: {MIN_ROWS}."
        )

    # 5. Target column values
    if TARGET_COLUMN in df.columns:
        unique_vals = set(df[TARGET_COLUMN].dropna().unique())
        normalized  = {int(v) for v in unique_vals}
        unexpected  = normalized - {0, 1}
        if unexpected:
            report["passed"] = False
            report["errors"].append(
                f"Unexpected values in target '{TARGET_COLUMN}': {unexpected}. "
                f"Expected: 0 (no default) or 1 (default risk)."
            )
        vc = df[TARGET_COLUMN].value_counts().to_dict()
        report["stats"]["target_distribution"] = {str(k): int(v) for k, v in vc.items()}

        # Warn if severe imbalance
        if len(normalized) == 2:
            counts = list(vc.values())
            ratio  = max(counts) / min(counts)
            if ratio > 10:
                report["warnings"].append(
                    f"Severe class imbalance detected: {ratio:.1f}:1. "
                    "Consider SMOTE or class_weight balancing."
                )

    # 6. Null analysis
    null_pct = (df.isnull().sum() / len(df)).to_dict()
    report["stats"]["null_percentage"] = {k: round(v, 4) for k, v in null_pct.items()}
    for col, pct in null_pct.items():
        if pct > MAX_NULL_PCT:
            report["passed"] = False
            report["errors"].append(
                f"Column '{col}' has {pct:.1%} null values (threshold: {MAX_NULL_PCT:.0%})."
            )
        elif pct > 0.10:
            report["warnings"].append(
                f"Column '{col}' has {pct:.1%} null values — consider imputation."
            )

    return report


if __name__ == "__main__":
    Path("reports").mkdir(parents=True, exist_ok=True)
    result = validate()
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))

    if result["warnings"]:
        print("\nWARNINGS:")
        for w in result["warnings"]:
            print(f"  [WARN] {w}")

    if not result["passed"]:
        print("\nERRORS:")
        for e in result["errors"]:
            print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[PASS] Data validation passed.")
