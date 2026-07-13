"""
API tests.

These tests do not require a trained model.
They only verify that Flask routes respond correctly.
"""

import pytest
from src.app import app

SAMPLE_PAYLOAD = {
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
    "Residence_State": "Karnataka",
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_missing_body_returns_400(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_missing_fields_returns_400(client):
    response = client.post("/predict", json={"Annual_Income": 1200000})
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]
