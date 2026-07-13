"""
Central configuration for the loan approval MLOps project.

All column names, paths, and thresholds are defined here.
Import from this module everywhere -- never hardcode strings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset column definitions  (matches Kaggle: Applicant Details For Loan Approve)
# ---------------------------------------------------------------------------

ID_COLUMN     = "Applicant_ID"
TARGET_COLUMN = "Loan_Default_Risk"   # 0 = no default, 1 = default risk

# Raw numeric features
NUMERIC_FEATURES = [
    "Annual_Income",
    "Applicant_Age",
    "Work_Experience",
    "Years_in_Current_Employment",
    "Years_in_Current_Residence",
    # engineered features (present after engineer_features() runs)
    "income_per_exp_year",
    "age_minus_experience",
    "employment_stability",
    "residence_stability",
]

# Raw categorical features
CATEGORICAL_FEATURES = [
    "Marital_Status",
    "House_Ownership",
    "Vehicle_Ownership(car)",
    "Occupation",
    "Residence_City",
    "Residence_State",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "model_metrics.md"

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = MODEL_DIR / "best_model.joblib"
