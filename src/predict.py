import pandas as pd

from src.config import MODEL_DIR
from src.utils import load_artifact

MODEL_PATH = MODEL_DIR / "best_model.joblib"


def predict(input_df: pd.DataFrame) -> dict:
    """
    Load the saved best model and return a prediction dict.

    Args:
        input_df: A single-row DataFrame with the required feature columns.

    Returns:
        dict with keys: prediction (int), loan_approved (bool),
                        approval_probability (float or None)
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run `python -m src.train` first."
        )

    model = load_artifact(MODEL_PATH)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(input_df)[0][1]), 4)

    return {
        "prediction": prediction,
        "loan_approved": bool(prediction),
        "approval_probability": probability,
    }
