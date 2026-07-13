from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "applicant-details-for-loan-approve.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "model_metrics.md"

# ---------------------------------------------------------------------------
# Actual dataset schema
# Columns: Applicant_ID, Annual_Income, Applicant_Age, Work_Experience,
#          Marital_Status, House_Ownership, Vehicle_Ownership(car),
#          Occupation, Residence_City, Residence_State,
#          Years_in_Current_Employment, Years_in_Current_Residence,
#          Loan_Default_Risk
# ---------------------------------------------------------------------------

# Drop this column — it is an identifier, not a feature
ID_COLUMN = "Applicant_ID"

# Target column
# Values: 0 = No default risk (loan safe), 1 = Default risk (loan risky)
TARGET_COLUMN = "Loan_Default_Risk"

# Numeric features
NUMERIC_FEATURES = [
    "Annual_Income",
    "Applicant_Age",
    "Work_Experience",
    "Years_in_Current_Employment",
    "Years_in_Current_Residence",
]

# Categorical features
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
# Target encoding map
# The target is already numeric (0/1) in this dataset.
# No Y/N mapping needed. We keep this dict for documentation.
# ---------------------------------------------------------------------------
TARGET_MAP = {0: "No Default Risk", 1: "Default Risk"}
