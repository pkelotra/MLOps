# Operational Workflow

PR -> GitHub Actions -> tests + contract checks + Docker build -> merge.

Training -> Person 1 pipeline -> Model 1 -> Model 2 -> evaluation -> MLflow Registry.

Deployment -> approved model -> Docker/FastAPI -> production.

Monitoring -> reference/current data -> drift/performance -> alert/retraining.

Retraining -> candidate models -> evaluation -> MLflow -> controlled promotion -> deployment.

A new model is never promoted merely because it exists; it must pass agreed quality gates.
