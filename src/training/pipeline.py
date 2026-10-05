import os
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import TimeSeriesSplit

from src.data.load import load_client_data
from src.data.validate import validate_data
from src.features.engineer import build_features
from src.models.forecast_automl import train_forecaster, evaluate_forecaster
from src.models.peak_automl import train_peak_classifier, evaluate_peak_classifier
from src.models.pipeline_model import UnifiedServingPipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads configuration settings from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "configs/config.yaml"):
    """
    Executes the end-to-end MLOps training pipeline:
    Data loading -> Validation -> Feature Prep -> Chronological Split
    -> Model 1 AutoML -> Out-of-sample Model 2 AutoML -> MLflow Logging -> Artifact Export
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    automl_cfg = config["automl"]
    peak_cfg = config["peak_detection"]
    mlflow_cfg = config["mlflow"]
    
    # 1. Setup MLflow Tracking
    if "tracking_uri" in mlflow_cfg:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    
    print("\n--- [Step 1/6] Loading & Ingesting Dataset ---")
    df_hourly = load_client_data(
        filepath=data_cfg["raw_path"],
        client_id=data_cfg["client_id"],
        start_date=data_cfg["start_date"]
    )
    print(f"Loaded {len(df_hourly)} hourly records for client {data_cfg['client_id']}.")
    
    print("\n--- [Step 2/6] Validating Data Integrity ---")
    validate_data(df_hourly, target_col="load_kwh")
    print("Data validation passed successfully: zero nulls, monotonic timestamps, positive load.")
    
    print("\n--- [Step 3/6] Engineering Features & Splitting Chronologically ---")
    df_features = build_features(df_hourly, target_col="load_kwh")
    
    # Strict temporal partitions (no random shuffle)
    train_mask = (df_features.index <= data_cfg["train_end"])
    val_mask = (df_features.index > data_cfg["train_end"]) & (df_features.index <= data_cfg["val_end"])
    test_mask = (df_features.index > data_cfg["val_end"]) & (df_features.index <= data_cfg["test_end"])
    
    feature_cols = ["hour", "day_of_week", "lag_1", "lag_24", "rolling_mean_24h"]
    target_col = "load_kwh"
    
    X_train = df_features.loc[train_mask, feature_cols]
    y_train = df_features.loc[train_mask, target_col]
    
    X_val = df_features.loc[val_mask, feature_cols]
    y_val = df_features.loc[val_mask, target_col]
    
    X_test = df_features.loc[test_mask, feature_cols]
    y_test = df_features.loc[test_mask, target_col]
    
    # Calculate peak threshold strictly from the training set
    peak_threshold = float(np.quantile(y_train, peak_cfg["percentile_threshold"]))
    y_peak_train = (y_train >= peak_threshold).astype(int)
    y_peak_val = (y_val >= peak_threshold).astype(int)
    y_peak_test = (y_test >= peak_threshold).astype(int)
    
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print(f"Peak load threshold (90th percentile of train): {peak_threshold:.2f} kWh")
    
    with mlflow.start_run(run_name=f"run_{data_cfg['client_id']}_automl") as run:
        # Log metadata parameters
        mlflow.log_params({
            "client_id": data_cfg["client_id"],
            "resample_freq": data_cfg["resample_freq"],
            "peak_threshold_kwh": round(peak_threshold, 2),
            "automl_time_budget_sec": automl_cfg["time_budget_seconds"]
        })
        
        # --- [Step 4/6] Train Model 1 (Forecasting AutoML) ---
        print("\n--- [Step 4/6] Training Model 1 (Load Forecaster) via AutoML ---")
        model_1 = train_forecaster(
            X_train, y_train,
            time_budget=automl_cfg["time_budget_seconds"],
            seed=automl_cfg["seed"]
        )
        val_metrics_m1, val_preds_m1 = evaluate_forecaster(model_1, X_val, y_val)
        
        mlflow.log_param("model1_best_estimator", model_1.best_estimator)
        mlflow.log_metrics({f"m1_val_{k}": v for k, v in val_metrics_m1.items()})
        print(f"Model 1 Best Algorithm: {model_1.best_estimator}")
        print(f"Model 1 Val Metrics: RMSE={val_metrics_m1['rmse']}, MAE={val_metrics_m1['mae']}")
        
        # --- [Step 5/6] Train Model 2 (Peak Classifier) using Model 1 Predictions ---
        print("\n--- [Step 5/6] Training Model 2 (Peak Classifier) via AutoML ---")
        # Generate out-of-sample predictions on train using TimeSeriesSplit to prevent leakage
        oof_train_preds = np.zeros(len(X_train))
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X_train):
            proxy_m1 = train_forecaster(X_train.iloc[train_idx], y_train.iloc[train_idx], time_budget=10)
            oof_train_preds[test_idx] = proxy_m1.predict(X_train.iloc[test_idx])
            
        # For initial indices before first split, backfill with proxy prediction
        unpredicted_mask = (oof_train_preds == 0)
        if unpredicted_mask.any():
            oof_train_preds[unpredicted_mask] = model_1.predict(X_train[unpredicted_mask])
            
        # Construct Model 2 feature matrices
        X_train_m2 = pd.DataFrame({
            "predicted_load": oof_train_preds,
            "hour": X_train["hour"].values,
            "day_of_week": X_train["day_of_week"].values,
            "rolling_mean_24h": X_train["rolling_mean_24h"].values
        }, index=X_train.index)
        
        X_val_m2 = pd.DataFrame({
            "predicted_load": val_preds_m1,
            "hour": X_val["hour"].values,
            "day_of_week": X_val["day_of_week"].values,
            "rolling_mean_24h": X_val["rolling_mean_24h"].values
        }, index=X_val.index)
        
        model_2 = train_peak_classifier(
            X_train_m2, y_peak_train,
            time_budget=automl_cfg["time_budget_seconds"],
            seed=automl_cfg["seed"]
        )
        val_metrics_m2, _, _ = evaluate_peak_classifier(model_2, X_val_m2, y_peak_val)
        
        mlflow.log_param("model2_best_estimator", model_2.best_estimator)
        mlflow.log_metrics({f"m2_val_{k}": v for k, v in val_metrics_m2.items()})
        print(f"Model 2 Best Algorithm: {model_2.best_estimator}")
        print(f"Model 2 Val Metrics: F1={val_metrics_m2['f1']}, Recall={val_metrics_m2['recall']}, Precision={val_metrics_m2['precision']}")
        
        # --- [Step 6/6] Bundle Unified Serving Pipeline & Export ---
        print("\n--- [Step 6/6] Packaging Unified Serving Pipeline for Person 2 ---")
        os.makedirs("artifacts", exist_ok=True)
        serving_pipeline = UnifiedServingPipeline(
            model_1=model_1,
            model_2=model_2,
            peak_threshold=peak_threshold
        )
        artifact_path = "artifacts/serving_pipeline.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(serving_pipeline, f)
            
        mlflow.log_artifact(artifact_path, artifact_path="model")
        print(f"Serving pipeline packaged and saved to {artifact_path} & logged to MLflow.")
        print(f"MLflow Run ID: {run.info.run_id}")
        
    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Electricity Load Forecasting & Peak Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()
    
    run_pipeline(args.config)
