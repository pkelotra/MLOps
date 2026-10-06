"""
Model 1 — Electricity Load Forecaster (Regression via FLAML AutoML).

Trains a regression model to predict next-hour electricity consumption (kWh).
FLAML evaluates LightGBM, XGBoost, Random Forest, and ExtraTrees candidates
and selects the best estimator by RMSE within the given time budget.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flaml import AutoML


def train_forecaster(X_train, y_train, time_budget: int = 30, seed: int = 42) -> AutoML:
    """
    Trains Model 1 (load forecasting) using FLAML AutoML.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features: [hour, day_of_week, lag_1, lag_24, rolling_mean_24h].
    y_train : pd.Series
        Training target: continuous load in kWh.
    time_budget : int
        Maximum seconds for FLAML's search.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    AutoML
        Fitted FLAML AutoML object (exposes .predict(), .best_estimator, etc.).
    """
    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task="regression",
        metric="rmse",
        time_budget=time_budget,
        seed=seed,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        verbose=0,
    )
    return automl


def evaluate_forecaster(model: AutoML, X_val, y_val) -> tuple:
    """
    Evaluates Model 1 on a held-out validation set.

    Returns
    -------
    tuple[dict, np.ndarray]
        (metrics_dict with 'rmse' and 'mae', raw predictions array)
    """
    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
    return metrics, preds
