"""
Stage 1 — Data Validation

Checks:
  - CSV file exists
  - Required columns are present
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

from src.config import RAW_DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS

REPORT_PATH = Path("reports/validation_report.json")
MIN_ROWS = 100
MAX_NULL_PCT = 0.50


def validate() -> dict:
    report = {"passed": True, "errors": [], "warnings": [], "stats": {}}

    # 1. File existence
    if not RAW_DATA_PATH.exists():
        report["passed"] = False
        report["errors"].append(f"Dataset not found: {RAW_DATA_PATH}")
        return report

    df = pd.read_csv(RAW_DATA_PATH)
    report["stats"]["total_rows"] = len(df)
    report["stats"]["total_columns"] = len(df.columns)
    report["stats"]["columns"] = list(df.columns)

    # 2. Required columns check
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        report["passed"] = False
        report["errors"].append(f"Missing required columns: {missing_cols}")

    # 3. Minimum row count
    if len(df) < MIN_ROWS:
        report["passed"] = False
        report["errors"].append(
            f"Dataset has only {len(df)} rows. Minimum required: {MIN_ROWS}."
        )

    # 4. Target column values
    if TARGET_COLUMN in df.columns:
        unique_vals = set(df[TARGET_COLUMN].dropna().unique())
        # Accept both int (0,1) and float (0.0,1.0)
        normalized = {int(v) for v in unique_vals}
        unexpected = normalized - {0, 1}
        if unexpected:
            report["passed"] = False
            report["errors"].append(
                f"Unexpected values in target column '{TARGET_COLUMN}': {unexpected}. "
                f"Expected: 0 (no default risk) or 1 (default risk)."
            )
        vc = df[TARGET_COLUMN].value_counts().to_dict()
        report["stats"]["target_distribution"] = {str(k): int(v) for k, v in vc.items()}

    # 5. Null analysis
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
                f"Column '{col}' has {pct:.1%} null values."
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
