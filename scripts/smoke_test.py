"""
Smoke test for the live FastAPI service.
Can test against an in-process TestClient or a running HTTP URL.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_smoke_test(url: str = None):
    payload = {
        "client_id": "MT_200",
        "target_timestamp": "2014-07-01 00:00:00",
        "recent_24h_loads": [
            312.4, 305.1, 298.6, 280.2, 275.0, 260.4, 255.8, 250.1,
            270.3, 310.5, 340.2, 355.7, 360.1, 358.9, 345.6, 330.2,
            335.8, 342.1, 360.4, 375.2, 380.5, 365.1, 340.9, 325.0,
        ],
    }

    if url:
        import httpx
        with httpx.Client(base_url=url, timeout=10.0) as client:
            _execute_checks(client, payload)
    else:
        from fastapi.testclient import TestClient
        from src.api.main import app
        with TestClient(app) as client:
            _execute_checks(client, payload)


def _execute_checks(client, payload):

    print("Checking /health...")
    res = client.get("/health")
    print(f"  Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    assert res.json()["status"] == "ok", f"Health status degraded: {res.text}"

    print("\nChecking /model-info...")
    res = client.get("/model-info")
    print(f"  Status: {res.status_code}, Body: {res.json()}")
    assert res.status_code == 200, f"Model info failed: {res.text}"
    assert res.json()["model_loaded"] is True, "Model not loaded"

    print("\nChecking /predict...")
    res = client.post("/predict", json=payload)
    print(f"  Status: {res.status_code}, Body: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200, f"Prediction failed: {res.text}"

    body = res.json()
    assert "predicted_load_kwh" in body
    assert "is_peak" in body
    assert "peak_probability" in body
    assert body["target_timestamp"] == payload["target_timestamp"]

    print("\nAll smoke tests passed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Smoke Test")
    parser.add_argument("--url", type=str, default=None, help="Base URL of running API server")
    args = parser.parse_args()
    run_smoke_test(args.url)
