"""
Register trained models in MLflow Model Registry and assign aliases.

This script is run after training to register the UnifiedServingPipeline
artifact as a versioned model in MLflow's registry, enabling candidate/champion
lifecycle management.

Usage:
    python scripts/register_model.py [--run-id <MLFLOW_RUN_ID>]

If --run-id is omitted, uses the latest run in the experiment.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient


def get_latest_run_id(experiment_name: str = "electricity_mlops_pipeline") -> str:
    """Find the most recent completed run in the experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    return runs.iloc[0]["run_id"]


def register_model(run_id: str, model_name: str = "electricity-serving-pipeline"):
    """Register the serving_pipeline.pkl artifact from an MLflow run."""
    client = MlflowClient()

    # The training pipeline logs the artifact under "model/serving_pipeline.pkl"
    model_uri = f"runs:/{run_id}/model"

    print(f"Registering model from run {run_id}...")
    print(f"  Model URI: {model_uri}")
    print(f"  Registry name: {model_name}")

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version

    print(f"  Registered as version: {version}")
    return version


def promote_to_champion(model_name: str, version: str):
    """Assign the 'champion' alias to the given version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"  Version {version} promoted to 'champion'.")


def main():
    parser = argparse.ArgumentParser(description="Register and promote MLflow model")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID")
    parser.add_argument("--model-name", type=str, default="electricity-serving-pipeline")
    parser.add_argument("--promote", action="store_true", help="Also assign champion alias")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    run_id = args.run_id or get_latest_run_id()
    version = register_model(run_id, args.model_name)

    if args.promote:
        promote_to_champion(args.model_name, version)

    print("\nDone.")


if __name__ == "__main__":
    main()
