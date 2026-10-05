"""
Tests for FastAPI health, model-info, and reload endpoints.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models.pipeline_model import UnifiedServingPipeline


def _make_mock_pipeline(peak_threshold=1205.4, predicted_load=320.5, peak_pred=0, peak_proba=0.15):
    """Build a UnifiedServingPipeline with mock models (not pickled)."""
    mock_m1 = MagicMock()
    mock_m1.predict.return_value = np.array([predicted_load])

    mock_m2 = MagicMock()
    mock_m2.predict.return_value = np.array([peak_pred])
    mock_m2.predict_proba.return_value = np.array([[1 - peak_proba, peak_proba]])

    return UnifiedServingPipeline(model_1=mock_m1, model_2=mock_m2, peak_threshold=peak_threshold)


@pytest.fixture
def client_with_model():
    """TestClient with a mock model loaded directly (no pickle needed)."""
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
def client_without_model():
    """TestClient with no model available."""
    from src.api.model_manager import model_manager
    from src.api.main import app

    with patch.object(model_manager, "load"):
        model_manager.pipeline = None
        model_manager.load_error = "Test: no model loaded."
        with TestClient(app) as c:
            yield c

    # Restore
    model_manager.load()


# ---- Health endpoint ----

def test_health_endpoint_model_loaded(client_with_model):
    """Health returns status=ok when model is loaded."""
    response = client_with_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_endpoint_model_missing(client_without_model):
    """Health returns status=degraded when model is unavailable."""
    response = client_without_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


# ---- Model info endpoint ----

def test_model_info_loaded(client_with_model):
    """Model-info shows threshold when model is available."""
    response = client_with_model.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["pipeline_type"] == "UnifiedServingPipeline"
    assert data["peak_threshold_kwh"] == pytest.approx(1205.4, rel=1e-3)
    assert data["load_error"] is None


def test_model_info_missing(client_without_model):
    """Model-info shows error when model is unavailable."""
    response = client_without_model.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["load_error"] is not None


# ---- Reload endpoint ----

def test_reload_endpoint(client_with_model):
    """Reload endpoint triggers model reload."""
    from src.api.model_manager import model_manager

    with patch.object(model_manager, "reload") as mock_reload:
        response = client_with_model.post("/reload")
        assert response.status_code == 200
        mock_reload.assert_called_once()
        data = response.json()
        assert data["status"] == "reloaded"

