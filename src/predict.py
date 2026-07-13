"""
Inference module.

Loads the best saved model and returns a prediction dict.

Prediction output:
  prediction          — 0 = No Default Risk, 1 = Default Risk
  loan_safe           — True if prediction == 0
  default_probability — probability of default (class 1)
"""

import pandas as pd

from src.config import MODEL_DIR
from src.utils import load_artifact

MODEL_PATH = MODEL_DIR / "best_model.joblib"


def predict(input_df: pd.DataFrame) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run `python -m src.pipeline.train_pipeline` first."
        )

    model = load_artifact(MODEL_PATH)
    prediction = int(model.predict(input_df)[0])

    default_prob = None
    if hasattr(model, "predict_proba"):
        default_prob = round(float(model.predict_proba(input_df)[0][1]), 4)

    return {
        "prediction": prediction,
        "loan_safe": bool(prediction == 0),
        "default_probability": default_prob,
        "label": "No Default Risk" if prediction == 0 else "Default Risk",
    }
