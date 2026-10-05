# Electricity Load MLOps — Person 2

MLOps/infrastructure side of the ElectricityLoadDiagrams20112014 project.

## Architecture

UCI data -> Person 1 training pipeline -> MLflow Registry -> Model 1 forecast -> Model 2 peak classifier -> FastAPI -> Docker -> monitoring -> retraining trigger.

## Person 2 owns
- GitHub Actions CI/CD
- Docker
- MLflow serving/registry integration
- FastAPI inference API
- Integration tests
- Monitoring/drift detection
- Retraining orchestration
- Deployment configuration

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

API: http://localhost:8000/docs

MLflow: http://localhost:5000

Until Person 1 registers real models, the API will report that models are unavailable rather than generating fake predictions.

See `contracts/model_contract.json` and `docs/INTEGRATION_CHECKLIST.md`.
