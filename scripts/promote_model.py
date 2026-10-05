"""
Promote or rollback models in MLflow Model Registry.

Promotion: Move the 'champion' alias to a new version.
Rollback:  Move the 'champion' alias back to a previous version.

Usage:
    python scripts/promote_model.py --version 3
    python scripts/promote_model.py --rollback --version 2
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from mlflow.tracking import MlflowClient


def get_champion_version(client: MlflowClient, model_name: str) -> str | None:
    """Get the current champion version number, or None."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        return mv.version
    except Exception:
        return None


def promote(client: MlflowClient, model_name: str, version: str):
    """Promote a version to champion. Tags the previous champion as 'previous_champion'."""
    current_champion = get_champion_version(client, model_name)

    if current_champion == version:
        print(f"Version {version} is already the champion. Nothing to do.")
        return

    # Tag previous champion for rollback
    if current_champion is not None:
        print(f"Tagging current champion (v{current_champion}) as 'previous_champion'.")
        client.set_registered_model_alias(model_name, "previous_champion", current_champion)

    # Promote new version
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"Version {version} promoted to 'champion'.")


def rollback(client: MlflowClient, model_name: str, version: str | None = None):
    """
    Rollback to a specific version, or to the 'previous_champion' alias if
    no version is specified.
    """
    if version is None:
        # Try to find previous_champion alias
        try:
            mv = client.get_model_version_by_alias(model_name, "previous_champion")
            version = mv.version
            print(f"Rolling back to previous_champion (v{version}).")
        except Exception:
            print("ERROR: No 'previous_champion' alias found and no --version specified.")
            sys.exit(1)

    current_champion = get_champion_version(client, model_name)
    if current_champion == version:
        print(f"Version {version} is already the champion. Nothing to do.")
        return

    client.set_registered_model_alias(model_name, "champion", version)
    print(f"Rolled back to version {version} as 'champion'.")


def main():
    parser = argparse.ArgumentParser(description="Promote or rollback model versions")
    parser.add_argument("--model-name", type=str, default="electricity-serving-pipeline")
    parser.add_argument("--version", type=str, default=None, help="Target version number")
    parser.add_argument("--rollback", action="store_true", help="Rollback instead of promote")
    parser.add_argument("--list", action="store_true", dest="list_versions", help="List all versions")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if args.list_versions:
        print(f"Model: {args.model_name}")
        try:
            versions = client.search_model_versions(f"name='{args.model_name}'")
            for v in versions:
                aliases = v.aliases if hasattr(v, "aliases") else []
                print(f"  v{v.version}: stage={v.current_stage}, aliases={aliases}, run_id={v.run_id}")
        except Exception as e:
            print(f"  Error: {e}")
        champion = get_champion_version(client, args.model_name)
        print(f"  Current champion: v{champion}" if champion else "  No champion set.")
        return

    if args.rollback:
        rollback(client, args.model_name, args.version)
    else:
        if args.version is None:
            print("ERROR: --version is required for promotion.")
            sys.exit(1)
        promote(client, args.model_name, args.version)


if __name__ == "__main__":
    main()
