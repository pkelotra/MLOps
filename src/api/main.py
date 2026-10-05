"""
FastAPI application — Electricity Load & Peak Forecasting API.

Thin serving layer that delegates all ML logic to UnifiedServingPipeline.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import settings
from .model_manager import model_manager
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact at startup; clean up on shutdown."""
    logging.basicConfig(level=settings.log_level)
    model_manager.load()
    yield


app = FastAPI(
    title="Electricity Load & Peak Forecasting API",
    description=(
        "Predicts next-hour electricity consumption and detects peak demand "
        "events using a chained two-stage ML pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Service health check — reports model readiness."""
    return HealthResponse(
        status="ok" if model_manager.ready else "degraded",
        model_loaded=model_manager.ready,
        artifact_path=settings.model_artifact_path,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Returns metadata about the currently loaded model."""
    threshold = None
    if model_manager.ready and hasattr(model_manager.pipeline, "peak_threshold"):
        threshold = model_manager.pipeline.peak_threshold

    return ModelInfoResponse(
        pipeline_type="UnifiedServingPipeline",
        artifact_path=settings.model_artifact_path,
        model_loaded=model_manager.ready,
        peak_threshold_kwh=threshold,
        load_error=model_manager.load_error,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict next-hour load and peak status.

    Accepts exactly 24 recent hourly load readings and a target timestamp.
    Returns predicted load (kWh), peak classification, and probability.
    """
    if not model_manager.ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. {model_manager.load_error or ''}",
        )

    start = time.time()
    try:
        result = model_manager.predict(
            recent_24h_loads=request.recent_24h_loads,
            timestamp_str=request.target_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    latency_ms = round((time.time() - start) * 1000, 1)
    logger.info("Prediction served in %.1fms", latency_ms)

    return PredictionResponse(**result)


@app.post("/reload")
def reload_model():
    """Hot-reload the model artifact without restarting the server."""
    model_manager.reload()
    return {
        "status": "reloaded",
        "model_loaded": model_manager.ready,
        "error": model_manager.load_error,
    }
