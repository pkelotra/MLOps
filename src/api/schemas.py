"""
Pydantic request/response schemas matching the documented API contract.

Request: client_id, target_timestamp, recent_24h_loads (24 floats).
Response: target_timestamp, predicted_load_kwh, is_peak, peak_probability, peak_threshold_kwh.
"""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Incoming prediction request — matches the serving contract."""
    client_id: str = Field(..., description="Client identifier (e.g. MT_200).")
    target_timestamp: str = Field(
        ...,
        description="Target timestamp in 'YYYY-MM-DD HH:MM:SS' format.",
    )
    recent_24h_loads: list[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description="Exactly 24 hourly kWh readings ordered [t-24, t-23, ..., t-1].",
    )

    @field_validator("recent_24h_loads")
    @classmethod
    def validate_loads_length(cls, v):
        if len(v) != 24:
            raise ValueError("recent_24h_loads must contain exactly 24 hourly readings.")
        return v


class PredictionResponse(BaseModel):
    """Outgoing prediction response — matches the serving contract."""
    target_timestamp: str
    predicted_load_kwh: float
    is_peak: bool
    peak_probability: float
    peak_threshold_kwh: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    artifact_path: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    pipeline_type: str
    artifact_path: str
    model_loaded: bool
    peak_threshold_kwh: float | None = None
    load_error: str | None = None
