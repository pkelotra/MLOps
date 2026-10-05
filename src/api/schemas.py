from typing import Any, Dict
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features matching Person 1's production Model 1 contract.")

class PredictionResponse(BaseModel):
    predicted_load: float
    peak_probability: float
    is_peak: bool
    model1_uri: str
    model2_uri: str
