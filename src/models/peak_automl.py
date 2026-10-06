"""
Model 2 — Peak Demand Classifier (Classification via FLAML AutoML).

Trains a binary classifier to detect peak demand events. Uses Model 1's
predicted load as a chained feature alongside calendar and rolling statistics.
FLAML optimises for F1-score to handle the inherent class imbalance
(only ~10% of hours are peaks at the 90th percentile threshold).
"""

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from flaml import AutoML


def train_peak_classifier(X_train, y_train, time_budget: int = 30, seed: int = 42) -> AutoML:
    """
    Trains Model 2 (peak classifier) using FLAML AutoML.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features: [predicted_load, hour, day_of_week, rolling_mean_24h].
    y_train : pd.Series
        Binary labels (1 = peak, 0 = normal).
    time_budget : int
        Maximum seconds for FLAML's search.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    AutoML
        Fitted FLAML AutoML object.
    """
    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task="classification",
        metric="f1",
        time_budget=time_budget,
        seed=seed,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        verbose=0,
    )
    return automl


def evaluate_peak_classifier(model: AutoML, X_val, y_val) -> tuple:
    """
    Evaluates Model 2 on a held-out validation set.

    Returns
    -------
    tuple[dict, np.ndarray, np.ndarray]
        (metrics_dict, binary predictions, probability predictions)
    """
    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)

    f1 = float(f1_score(y_val, preds, zero_division=0))
    recall = float(recall_score(y_val, preds, zero_division=0))
    precision = float(precision_score(y_val, preds, zero_division=0))

    # ROC-AUC requires probability of positive class
    try:
        roc_auc = float(roc_auc_score(y_val, probas[:, 1]))
    except (ValueError, IndexError):
        roc_auc = 0.0

    metrics = {
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "roc_auc": round(roc_auc, 4),
    }
    return metrics, preds, probas
