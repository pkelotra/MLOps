import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model1_uri: str = os.getenv("MODEL1_URI", "models:/electricity-load-forecaster@champion")
    model2_uri: str = os.getenv("MODEL2_URI", "models:/electricity-peak-classifier@champion")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
