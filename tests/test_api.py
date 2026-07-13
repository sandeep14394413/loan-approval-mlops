"""
Basic API tests.

These tests do not require a trained model — they only check
that the Flask routes respond with the expected status codes.
"""

import pytest
from src.app import app


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
    response = client.post("/predict", json={"Gender": "Male"})
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]
