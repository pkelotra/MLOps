"""
Model manager — loads and serves the UnifiedServingPipeline artifact.

Handles graceful failure when the artifact is not yet available (e.g. before
training has been run). The /health endpoint reflects model readiness.
"""

import logging
import os
import pickle
import time
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the lifecycle of the UnifiedServingPipeline artifact."""

    def __init__(self):
        self.pipeline = None
        self.load_error: Optional[str] = None
        self.load_time: Optional[float] = None

    def load(self) -> None:
        """Load the UnifiedServingPipeline from the configured pkl path."""
        artifact_path = settings.model_artifact_path
        if not os.path.exists(artifact_path):
            self.load_error = f"Artifact not found at {artifact_path}. Run training pipeline first."
            logger.warning(self.load_error)
            return

        try:
            start = time.time()
            with open(artifact_path, "rb") as f:
                self.pipeline = pickle.load(f)
            self.load_time = round(time.time() - start, 3)
            self.load_error = None
            logger.info(
                "UnifiedServingPipeline loaded from %s in %.3fs",
                artifact_path, self.load_time,
            )
        except Exception as exc:
            self.pipeline = None
            self.load_error = f"Failed to load artifact: {exc}"
            logger.error(self.load_error)

    def reload(self) -> None:
        """Hot-reload the model artifact (e.g. after retraining/promotion)."""
        logger.info("Reloading model artifact...")
        self.pipeline = None
        self.load_error = None
        self.load()

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, recent_24h_loads: list[float], timestamp_str: str) -> dict:
        """
        Delegates to UnifiedServingPipeline.predict().

        Raises RuntimeError if models are not loaded.
        """
        if not self.ready:
            raise RuntimeError(
                f"Model is not ready. Error: {self.load_error}"
            )
        return self.pipeline.predict(recent_24h_loads, timestamp_str)


# Singleton instance used by the FastAPI application
model_manager = ModelManager()
