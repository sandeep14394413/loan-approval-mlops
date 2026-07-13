"""
Flask REST API for Loan Approval Prediction.

Endpoints:
  GET  /health   — liveness / readiness check
  POST /predict  — predict loan approval

Usage (local):
  python -m src.app
"""

from flask import Flask, jsonify, request
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.predict import predict

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Kubernetes readiness and liveness probe endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def make_prediction():
    """
    Accept a JSON payload with applicant features and return a
    loan approval prediction.

    Required fields: Gender, Married, Dependents, Education,
    Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
    Loan_Amount_Term, Credit_History, Property_Area
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "A JSON payload is required."}), 400

        missing = [col for col in FEATURE_COLUMNS if col not in payload]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        input_df = pd.DataFrame([payload])
        result = predict(input_df)
        return jsonify(result), 200

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
