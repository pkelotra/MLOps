# Electricity Load Forecasting & Peak Detection MLOps Pipeline

[![CI](https://github.com/pkelotra/MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/pkelotra/MLOps/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passing-brightgreen.svg)](tests/)

An enterprise-grade, end-to-end MLOps pipeline for **next-hour electricity consumption forecasting** and **peak demand event classification**. Operating on the UCI Electricity Load Diagrams dataset, the system implements an automated cascading ML architecture with full CI/CD automation, model governance, drift monitoring, containerization, and zero-downtime serving.

---

## Table of Contents

- [1. Problem Formulation & Architecture](#1-problem-formulation--architecture)
- [2. Key System Features](#2-key-system-features)
- [3. Quickstart Guide](#3-quickstart-guide)
- [4. Master Pipeline Runner](#4-master-pipeline-runner)
- [5. API Specification & Contract](#5-api-specification--contract)
- [6. Drift Detection & Retraining Lifecycle](#6-drift-detection--retraining-lifecycle)
- [7. Containerization & Docker Compose](#7-containerization--docker-compose)
- [8. CI/CD Workflows](#8-cicd-workflows)
- [9. Automated Test Suite](#9-automated-test-suite)
- [10. Project Directory Structure](#10-project-directory-structure)

---

## 1. Problem Formulation & Architecture

The system solves two interconnected, cascading machine learning tasks for power grid management:

1. **Model 1 (Regression — Forecasting)**: Predicts continuous electricity consumption in kilowatt-hours (kWh) for the next hour ($t$).
   * **Selected Champion**: FLAML-tuned **XGBoost Regressor** (Validation RMSE: `28.238 kWh`, MAE: `20.732 kWh`).
2. **Model 2 (Classification — Peak Detection)**: Classifies whether the upcoming hour is an anomalous high-demand "peak" event ($1$) or normal demand ($0$). A peak is defined as consumption exceeding the 90th percentile of baseline training load (`1205.4 kWh`).
   * **Selected Champion**: FLAML-tuned **ExtraTrees Classifier** (Validation F1: `0.692`, Recall: `0.763`, ROC-AUC: `0.941`).
3. **Cascading Dependency**: Model 2 consumes **Model 1's forecasted load** as its primary input feature, simulating real-world decision pipelines (e.g. demand response dispatching).

### Architecture Diagram

```
UCI Raw Dataset (LD2011_2014.txt)
          │
          ▼
   [src/data/load.py] ────────► Targeted client ingestion, DST cleaning, resample to 1H kWh
          │
          ▼
 [src/data/validate.py] ──────► Invariant Gate: Zero NaNs, Non-negative (kWh >= 0), Monotonic
          │
          ▼
[src/features/engineer.py] ───► 5 Lean Features: hour, day_of_week, lag_1, lag_24, rolling_mean_24h
          │                     Chronological Splits: Train (2012-13), Val (2014-H1), Test (2014-H2)
          ├──► Model 1 (XGBoost) ──────► Predicts continuous next-hour load (kWh)
          │                                      │
          └──► Model 2 (ExtraTrees) ◄────────────┘ (Out-of-sample TimeSeriesSplit chaining)
                    │
                    ▼
[src/models/pipeline_model.py] ──► Bundles M1 + M2 + Threshold into UnifiedServingPipeline
                    │             Saved to artifacts/serving_pipeline.pkl
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 [sqlite:///mlflow.db]     [FastAPI Serving Layer (src/api/)]
  Model Registry & Runs    POST /predict (sub-50ms latency)
                           POST /reload  (Zero-downtime hot-reload)
```

---

## 2. Key System Features

- **Leakage-Safe Feature Engineering**: Rolling 24-hour averages are strictly shifted by 1 hour (`.shift(1)`), mathematically preventing current or future target leakage into features.
- **Unified Serving Abstraction (`UnifiedServingPipeline`)**: A self-contained, picklable wrapper that takes raw 24-hour load arrays, derives lags, runs Model 1, cascades to Model 2, and formats the response dictionary—freeing the serving layer from feature engineering dependencies.
- **FastAPI Microservice**: Thin serving layer with strict Pydantic validation, `/health` and `/model-info` observability probes, and sub-50ms inference latency.
- **Zero-Downtime Hot-Reloading (`POST /reload`)**: Models are swapped in memory without stopping the container or dropping active client requests.
- **Population Stability Index (PSI) Drift Detection**: Statistical monitoring detecting distribution shifts between reference and production traffic. Actionable drift ($\text{PSI} \ge 0.20$) triggers automated alerting.
- **Automated Quality Gates & MLflow Registry**: Candidate models are validated against contract and metric gates (`evaluate_candidate.py`) before promotion to the `champion` alias. One-command rollback (`promote_model.py --rollback`) protects production.
- **Multi-Container Stack**: Docker Compose orchestrating FastAPI (`:8000`) and the MLflow Tracking Server (`:5000`) with built-in Docker healthcheck probes.

---

## 3. Quickstart Guide

### 3.1 Setup Environment

```bash
# Clone the repository
git clone git@github.com:pkelotra/MLOps.git
cd MLOps

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Run Automated Tests (19 Tests)

```bash
pytest -v
```

### 3.3 Start the FastAPI Server Locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)
- **Model Info**: [http://localhost:8000/model-info](http://localhost:8000/model-info)

### 3.4 Run the Live Smoke Test

In a second terminal:

```bash
python scripts/smoke_test.py --url http://localhost:8000
```

---

## 4. Master Pipeline Runner

To execute and verify the **entire end-to-end MLOps lifecycle** in one command, run:

```bash
python scripts/run_pipeline.py
```

This master runner executes and validates:
1. Artifact and contract integrity verification.
2. The full 19-test automated test suite.
3. Quality gates evaluation against the candidate model.
4. Serving layer smoke testing (`POST /predict`).
5. PSI baseline drift monitoring check.
6. Full drift detection, alerting, and hot-reload lifecycle simulation.

---

## 5. API Specification & Contract

The API strictly implements the contract documented in [`contracts/model_contract.json`](contracts/model_contract.json).

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness & model readiness check (`status: ok / degraded`, `model_loaded`) |
| `GET` | `/model-info` | Metadata about loaded pipeline, artifact path, and peak threshold |
| `POST` | `/predict` | Chained Model 1 $\to$ Model 2 inference |
| `POST` | `/reload` | Zero-downtime hot-reload of model artifact from disk |

### Request Payload (`POST /predict`)

```json
{
  "client_id": "MT_200",
  "target_timestamp": "2014-07-01 00:00:00",
  "recent_24h_loads": [
    312.4, 305.1, 298.6, 280.2, 275.0, 260.4, 255.8, 250.1,
    270.3, 310.5, 340.2, 355.7, 360.1, 358.9, 345.6, 330.2,
    335.8, 342.1, 360.4, 375.2, 380.5, 365.1, 340.9, 325.0
  ]
}
```

> **Validation**: `recent_24h_loads` must contain **exactly 24 float values**. Inputs with fewer or more values are rejected with `HTTP 422 Unprocessable Entity`.

### Response Payload

```json
{
  "target_timestamp": "2014-07-01 00:00:00",
  "predicted_load_kwh": 392.31,
  "is_peak": false,
  "peak_probability": 0.0,
  "peak_threshold_kwh": 1205.4
}
```

### Curl Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "MT_200",
    "target_timestamp": "2014-07-01 00:00:00",
    "recent_24h_loads": [
      312.4, 305.1, 298.6, 280.2, 275.0, 260.4, 255.8, 250.1,
      270.3, 310.5, 340.2, 355.7, 360.1, 358.9, 345.6, 330.2,
      335.8, 342.1, 360.4, 375.2, 380.5, 365.1, 340.9, 325.0
    ]
  }'
```

---

## 6. Drift Detection & Retraining Lifecycle

Production data streams are monitored using the **Population Stability Index (PSI)** against baseline distributions (`monitoring/reference.json`):

$$\text{PSI} < 0.10 \implies \text{Stable} \quad|\quad 0.10 \le \text{PSI} < 0.20 \implies \text{Warning} \quad|\quad \text{PSI} \ge 0.20 \implies \text{Actionable Drift}$$

### Run the Interactive Drift Demonstration

```bash
python scripts/simulate_drift.py --demo-all
```

This simulates the full lifecycle:
1. **Checks Baseline**: Verifies $\text{PSI} = 0.0$ on identical distributions.
2. **Injects Simulated Drift**: Shifts production loads from $\approx 6\text{ kWh}$ to $\approx 126\text{ kWh}$.
3. **Alerts on Drift**: Detects $\text{PSI} = 11.54$, reports drifted feature `load`, and exits with code `2`.
4. **Evaluates Quality Gates**: Runs `scripts/evaluate_candidate.py` to ensure candidate models satisfy quality criteria.
5. **Hot-Reloads Live API**: Invokes `POST /reload` to swap models in memory without dropping connections.
6. **Cleans Up**: Restores baseline distributions automatically.

### Manual Drift Commands

```bash
# Check current drift status
python scripts/check_drift.py

# Inject simulated drift into current.json
python scripts/simulate_drift.py --inject

# Restore current.json back to baseline
python scripts/simulate_drift.py --restore
```

---

## 7. Containerization & Docker Compose

The production stack runs as a coordinated multi-container service:

```bash
# Build and launch all services in the background
docker compose up -d --build

# Inspect running container health status
docker compose ps

# Follow container logs
docker compose logs -f api
```

### Services

| Service | Port | Description | Healthcheck |
|---|---|---|---|
| **`api`** | `8000` | FastAPI Uvicorn Serving Service | `curl -f http://localhost:8000/health` (every 30s) |
| **`mlflow`** | `5000` | MLflow Tracking Server & Model Registry | Port listener |

---

## 8. CI/CD Workflows

Automated with GitHub Actions in [`.github/workflows/`](.github/workflows/):

- **[`ci.yml`](.github/workflows/ci.yml)**: Triggered on PRs and pushes to `main`. Runs code checks, the 19-test pytest suite, validates contract JSON formatting, and tests Docker image compilation.
- **[`train.yml`](.github/workflows/train.yml)**: Retraining pipeline triggered manually or by drift events (`repository_dispatch`). Retrains models via FLAML AutoML, evaluates candidates against quality gates, and conditionally promotes candidates to `champion`.
- **[`deploy.yml`](.github/workflows/deploy.yml)**: Triggered on version tags (`v*`). Builds the production image, spins up a temporary container, runs a healthcheck smoke test, and prepares release tags.
- **[`monitor.yml`](.github/workflows/monitor.yml)**: Scheduled cron job (every 6 hours). Executes `check_drift.py` and dispatches retraining events when drift occurs.

---

## 9. Automated Test Suite

The test suite runs in under 4 seconds and covers the entire stack:

```bash
pytest -v
```

```text
tests/test_api.py::test_predict_valid_request PASSED                     [  5%]
tests/test_api.py::test_predict_response_matches_contract_fixture PASSED [ 10%]
tests/test_api.py::test_predict_too_few_loads PASSED                     [ 15%]
tests/test_api.py::test_predict_too_many_loads PASSED                    [ 21%]
tests/test_api.py::test_predict_missing_fields PASSED                    [ 26%]
tests/test_api.py::test_predict_missing_timestamp PASSED                 [ 31%]
tests/test_api.py::test_predict_missing_client_id PASSED                 [ 36%]
tests/test_api.py::test_predict_model_unavailable PASSED                 [ 42%]
tests/test_drift.py::test_no_drift_for_identical_distributions PASSED    [ 47%]
tests/test_health.py::test_health_endpoint_model_loaded PASSED           [ 52%]
tests/test_health.py::test_health_endpoint_model_missing PASSED          [ 57%]
tests/test_health.py::test_model_info_loaded PASSED                      [ 63%]
tests/test_health.py::test_model_info_missing PASSED                     [ 68%]
tests/test_health.py::test_reload_endpoint PASSED                        [ 73%]
tests/test_pipeline.py::test_data_validation_valid PASSED                [ 78%]
tests/test_pipeline.py::test_data_validation_catches_negative_values PASSED [ 84%]
tests/test_pipeline.py::test_data_validation_catches_nans PASSED         [ 89%]
tests/test_pipeline.py::test_feature_engineering_no_leakage PASSED       [ 94%]
tests/test_pipeline.py::test_unified_serving_pipeline_contract PASSED    [100%]
======================== 19 passed, 1 warning in 3.04s =========================
```

---

## 10. Project Directory Structure

```text
electricity-mlops-person2/
├── .github/workflows/          # CI/CD Workflows (CI, Train, Deploy, Monitor)
├── artifacts/
│   └── serving_pipeline.pkl    # Serialized UnifiedServingPipeline artifact
├── configs/
│   ├── config.yaml             # Core training & MLflow configuration
│   └── serving.yaml            # Serving & quality gates configuration
├── contracts/
│   └── model_contract.json     # Immutable API schema & endpoint contract
├── docker/
│   ├── Dockerfile.inference    # Serving image (Python 3.11-slim, libgomp1, Uvicorn)
│   └── Dockerfile.training     # Retraining pipeline container
├── monitoring/
│   ├── current.json            # Live production sliding window
│   └── reference.json          # Baseline training distribution
├── scripts/
│   ├── check_drift.py          # PSI drift evaluator with exit code triggers
│   ├── evaluate_candidate.py   # Candidate quality gate validation
│   ├── promote_model.py        # MLflow champion promotion & rollback
│   ├── register_model.py       # MLflow model registration
│   ├── run_pipeline.py         # Master all-in-one pipeline runner
│   ├── simulate_drift.py       # Interactive drift simulation & demonstration
│   └── smoke_test.py           # In-process and HTTP smoke testing
├── src/
│   ├── api/                    # FastAPI application, schemas, model manager
│   ├── data/                   # Data ingestion, DST cleaning, validation
│   ├── features/               # Leakage-safe feature engineering
│   ├── models/                 # Model 1, Model 2, and UnifiedServingPipeline
│   ├── monitoring/             # Population Stability Index (PSI) logic
│   └── training/               # End-to-end training orchestrator
├── tests/                      # Automated test suite (19 tests)
├── docker-compose.yml          # Multi-service stack (FastAPI + MLflow)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Pinned dependencies
└── README.md                   # Project documentation
```

---

## License

This project is licensed under the MIT License.
