from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "applicant-details-for-loan-approve.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "processed.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "reports" / "model_metrics.md"

# Target column
TARGET_COLUMN = "Loan_Status"

# Feature definitions
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
