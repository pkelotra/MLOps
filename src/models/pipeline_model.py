"""
Unified Serving Pipeline — self-contained inference wrapper.

Bundles Model 1 (load forecaster), Model 2 (peak classifier), and the peak
threshold into a single picklable object. FastAPI loads this artifact and
calls .predict() without needing to know about lags, rolling means, or
model chaining internals.
"""

import numpy as np
import pandas as pd


class UnifiedServingPipeline:
    """
    Self-contained serving abstraction that chains Model 1 → Model 2.

    Parameters
    ----------
    model_1 : fitted estimator
        Load forecasting model with .predict(X) method.
    model_2 : fitted estimator
        Peak classification model with .predict(X) and .predict_proba(X) methods.
    peak_threshold : float
        The 90th-percentile kWh threshold used to define peak events.
    """

    def __init__(self, model_1, model_2, peak_threshold: float):
        self.model_1 = model_1
        self.model_2 = model_2
        self.peak_threshold = float(peak_threshold)

    def predict(self, recent_24h_loads: list[float], timestamp_str: str) -> dict:
        """
        End-to-end prediction from raw 24-hour history.

        Parameters
        ----------
        recent_24h_loads : list[float]
            Exactly 24 hourly kWh readings ordered [t-24, t-23, ..., t-1].
        timestamp_str : str
            Target timestamp in "YYYY-MM-DD HH:MM:SS" format.

        Returns
        -------
        dict
            {target_timestamp, predicted_load_kwh, is_peak, peak_probability,
             peak_threshold_kwh}
        """
        if len(recent_24h_loads) < 24:
            raise ValueError("recent_24h_loads must contain at least 24 values.")

        ts = pd.Timestamp(timestamp_str)

        # --- Compute Model 1 features from the 24-hour window ---
        lag_1 = recent_24h_loads[-1]          # load at t-1
        lag_24 = recent_24h_loads[0]          # load at t-24 (same hour yesterday)
        rolling_mean_24h = float(np.mean(recent_24h_loads[-24:]))

        m1_features = pd.DataFrame([{
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "rolling_mean_24h": rolling_mean_24h,
        }])

        # --- Model 1: forecast next-hour load ---
        predicted_load = float(self.model_1.predict(m1_features)[0])

        # --- Compute Model 2 features (chained) ---
        m2_features = pd.DataFrame([{
            "predicted_load": predicted_load,
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "rolling_mean_24h": rolling_mean_24h,
        }])

        # --- Model 2: peak classification ---
        peak_pred = int(self.model_2.predict(m2_features)[0])
        try:
            peak_proba = float(self.model_2.predict_proba(m2_features)[0][1])
        except (AttributeError, IndexError):
            peak_proba = float(peak_pred)

        is_peak = bool(peak_pred == 1)

        return {
            "target_timestamp": timestamp_str,
            "predicted_load_kwh": round(predicted_load, 2),
            "is_peak": is_peak,
            "peak_probability": round(peak_proba, 4),
            "peak_threshold_kwh": round(self.peak_threshold, 2),
        }
