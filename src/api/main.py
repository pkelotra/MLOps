from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .config import settings
from .schemas import PredictionRequest, PredictionResponse
from .model_manager import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load()
    yield

app = FastAPI(title="Electricity Load MLOps API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "models_ready": model_manager.ready, "model1_uri": settings.model1_uri, "model2_uri": settings.model2_uri}

@app.get("/model-info")
def model_info():
    return {"model1": {"uri": settings.model1_uri, "loaded": model_manager.model1 is not None, "error": model_manager.model1_error}, "model2": {"uri": settings.model2_uri, "loaded": model_manager.model2 is not None, "error": model_manager.model2_error}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predicted_load, probability, is_peak = model_manager.predict(request.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(predicted_load=predicted_load, peak_probability=probability, is_peak=is_peak, model1_uri=settings.model1_uri, model2_uri=settings.model2_uri)
