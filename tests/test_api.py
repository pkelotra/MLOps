"""
Integration tests for the /predict endpoint.

Covers:
- Valid prediction (full roundtrip)
- Response schema validation
- Invalid timestamp
- Wrong number of loads
- Empty request
- Model unavailable (503)
- Contract fixture validation
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models.pipeline_model import UnifiedServingPipeline

SAMPLE_REQUEST = {
    "client_id": "MT_200",
    "target_timestamp": "2014-07-01 00:00:00",
    "recent_24h_loads": [
        312.4, 305.1, 298.6, 280.2, 275.0, 260.4, 255.8, 250.1,
        270.3, 310.5, 340.2, 355.7, 360.1, 358.9, 345.6, 330.2,
        335.8, 342.1, 360.4, 375.2, 380.5, 365.1, 340.9, 325.0,
    ],
}


def _make_mock_pipeline():
    """Build a UnifiedServingPipeline with mock models."""
    mock_m1 = MagicMock()
    mock_m1.predict.return_value = np.array([392.31])

    mock_m2 = MagicMock()
    mock_m2.predict.return_value = np.array([0])
    mock_m2.predict_proba.return_value = np.array([[1.0, 0.0]])

    return UnifiedServingPipeline(model_1=mock_m1, model_2=mock_m2, peak_threshold=1205.4)


@pytest.fixture
def client():
    """TestClient with a mock model loaded directly."""
    from src.api.model_manager import model_manager
    from src.api.main import app

    with patch.object(model_manager, "load"):
        model_manager.pipeline = _make_mock_pipeline()
        model_manager.load_error = None
        with TestClient(app) as c:
            yield c

    # Restore
    model_manager.load()


@pytest.fixture
def client_no_model():
    """TestClient with no model available."""
    from src.api.model_manager import model_manager
    from src.api.main import app

    with patch.object(model_manager, "load"):
        model_manager.pipeline = None
        model_manager.load_error = "Test: model not available."
        with TestClient(app) as c:
            yield c

    # Restore
    model_manager.load()


# ---- Valid prediction ----

def test_predict_valid_request(client):
    """Full roundtrip prediction with valid input returns correct schema."""
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    data = response.json()

    # All required fields present
    assert "target_timestamp" in data
    assert "predicted_load_kwh" in data
    assert "is_peak" in data
    assert "peak_probability" in data
    assert "peak_threshold_kwh" in data

    # Correct types
    assert isinstance(data["target_timestamp"], str)
    assert isinstance(data["predicted_load_kwh"], (int, float))
    assert isinstance(data["is_peak"], bool)
    assert isinstance(data["peak_probability"], (int, float))
    assert isinstance(data["peak_threshold_kwh"], (int, float))

    # Timestamp echoed back
    assert data["target_timestamp"] == SAMPLE_REQUEST["target_timestamp"]

    # Peak threshold matches
    assert data["peak_threshold_kwh"] == pytest.approx(1205.4, rel=1e-3)


def test_predict_response_matches_contract_fixture(client):
    """Verify response shape matches tests/fixtures/sample_response.json."""
    response = client.post("/predict", json=SAMPLE_REQUEST)
    data = response.json()

    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_response.json")
    with open(fixture_path) as f:
        expected_keys = set(json.load(f).keys())

    assert set(data.keys()) == expected_keys


# ---- Invalid inputs ----

def test_predict_too_few_loads(client):
    """Rejects requests with fewer than 24 load values."""
    bad_request = {**SAMPLE_REQUEST, "recent_24h_loads": [100.0] * 10}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_too_many_loads(client):
    """Rejects requests with more than 24 load values."""
    bad_request = {**SAMPLE_REQUEST, "recent_24h_loads": [100.0] * 30}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_missing_fields(client):
    """Rejects requests with missing required fields."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_missing_timestamp(client):
    """Rejects requests without target_timestamp."""
    bad_request = {
        "client_id": "MT_200",
        "recent_24h_loads": [100.0] * 24,
    }
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_missing_client_id(client):
    """Rejects requests without client_id."""
    bad_request = {
        "target_timestamp": "2014-07-01 00:00:00",
        "recent_24h_loads": [100.0] * 24,
    }
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


# ---- Model unavailable ----

def test_predict_model_unavailable(client_no_model):
    """Returns 503 when model is not loaded."""
    response = client_no_model.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 503
    assert "not available" in response.json()["detail"].lower() or "not ready" in response.json()["detail"].lower()
