"""
Application settings — loaded from environment variables with sensible defaults.

Supports two serving modes:
1. Local/development: loads UnifiedServingPipeline from a pickle artifact.
2. Docker/production: same pkl artifact mounted into the container.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Path to the UnifiedServingPipeline pickle artifact
    model_artifact_path: str = os.getenv(
        "MODEL_ARTIFACT_PATH", "artifacts/serving_pipeline.pkl"
    )

    # MLflow tracking (used during training, not required for serving)
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )

    # API server configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Monitoring paths
    reference_data_path: str = os.getenv(
        "REFERENCE_DATA_PATH", "monitoring/reference.json"
    )
    current_data_path: str = os.getenv(
        "CURRENT_DATA_PATH", "monitoring/current.json"
    )
    drift_threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.20"))


settings = Settings()
