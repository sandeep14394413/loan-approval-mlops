"""
Flask REST API for Loan Default Risk Prediction.

Endpoints:
  GET  /health   — liveness / readiness check
  POST /predict  — predict loan default risk

The API accepts the 11 RAW feature columns from the dataset.
Feature engineering (4 derived columns) is applied automatically
inside predict() before the model sees the input — callers never
need to send engineered columns.

Required fields in POST /predict payload:
  Annual_Income, Applicant_Age, Work_Experience,
  Years_in_Current_Employment, Years_in_Current_Residence,
  Marital_Status, House_Ownership, Vehicle_Ownership(car),
  Occupation, Residence_City, Residence_State

Usage (local):
  python -m src.app
"""

from flask import Flask, jsonify, request
import pandas as pd

from src.config import RAW_FEATURE_COLUMNS
from src.predict import predict

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Kubernetes readiness and liveness probe endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def make_prediction():
    """
    Accept a JSON payload with raw applicant features and return
    a loan default risk prediction.

    Example request:
    {
      "Annual_Income": 1200000,
      "Applicant_Age": 35,
      "Work_Experience": 8,
      "Years_in_Current_Employment": 3,
      "Years_in_Current_Residence": 5,
      "Marital_Status": "married",
      "House_Ownership": "owned",
      "Vehicle_Ownership(car)": "yes",
      "Occupation": "Software_Developer",
      "Residence_City": "Bangalore",
      "Residence_State": "Karnataka"
    }

    Example response:
    {
      "prediction": 0,
      "loan_safe": true,
      "default_probability": 0.0821,
      "label": "No Default Risk"
    }
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "A JSON payload is required."}), 400

        # Validate only RAW columns — engineered features are added by predict()
        missing = [col for col in RAW_FEATURE_COLUMNS if col not in payload]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        input_df = pd.DataFrame([payload])
        result = predict(input_df)
        return jsonify(result), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
