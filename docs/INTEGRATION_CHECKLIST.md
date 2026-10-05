# Person 1 ↔ Person 2 Integration Checklist

## Person 1 must provide
- Exact Model 1 feature names/types
- Exact Model 2 feature names/types
- Model 1 target definition
- Peak definition
- Registered model names and aliases
- Model 1 output format
- Model 2 output format
- Inference preprocessing/feature logic
- Known-good request/response example
- Quality thresholds

## Person 2 provides
- MLflow registry integration
- FastAPI
- Docker
- CI/CD
- Deployment
- Monitoring/drift
- Retraining trigger
- Rollback strategy

## Final E2E test
1. Register Model 1 and Model 2.
2. Assign champion aliases.
3. Start MLflow and API.
4. Verify `/health` and `/model-info`.
5. Send known prediction request.
6. Verify Model 1 -> Model 2 chaining.
7. Run CI.
8. Build/deploy Docker image.
9. Run monitoring.
10. Simulate drift.
11. Trigger retraining.
12. Evaluate candidate.
13. Promote only if quality gates pass.
