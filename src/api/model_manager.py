from typing import Any
import mlflow
import pandas as pd
from .config import settings

class ModelManager:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model1_error = None
        self.model2_error = None
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def load(self):
        try:
            self.model1 = mlflow.pyfunc.load_model(settings.model1_uri)
            self.model1_error = None
        except Exception as exc:
            self.model1 = None
            self.model1_error = str(exc)
        try:
            self.model2 = mlflow.pyfunc.load_model(settings.model2_uri)
            self.model2_error = None
        except Exception as exc:
            self.model2 = None
            self.model2_error = str(exc)

    @property
    def ready(self):
        return self.model1 is not None and self.model2 is not None

    def predict(self, features: dict[str, Any]):
        if not self.ready:
            raise RuntimeError(f"Models are not ready. Model1 error={self.model1_error}; Model2 error={self.model2_error}")
        model1_input = pd.DataFrame([features])
        predicted_load = float(self.model1.predict(model1_input)[0])
        model2_features = dict(features)
        model2_features["predicted_load"] = predicted_load
        result = self.model2.predict(pd.DataFrame([model2_features]))[0]
        if isinstance(result, dict):
            probability = float(result.get("peak_probability", result.get("probability", 0.0)))
            is_peak = bool(result.get("is_peak", probability >= 0.5))
        else:
            probability = float(result)
            is_peak = probability >= 0.5
        return predicted_load, probability, is_peak

model_manager = ModelManager()
