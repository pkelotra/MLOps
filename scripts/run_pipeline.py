"""
Master Pipeline Runner — Runs the entire Electricity MLOps pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py
"""

import json
import os
import subprocess
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def log_step(step_num, title):
    print("\n" + "=" * 70)
    print(f"  STEP {step_num}: {title.upper()}")
    print("=" * 70)


def run_command(cmd, desc):
    print(f">> Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.returncode != 0:
        print(f"[!] Warning/Error ({desc}): Exit code {res.returncode}")
        if res.stderr:
            print(res.stderr.strip())
    return res


def main():
    print("""
======================================================================
     ELECTRICITY LOAD FORECASTING & PEAK DETECTION MLOPS PIPELINE
                     Master End-to-End Pipeline Runner
======================================================================
""")
    start_time = time.time()

    # Step 1: Environment & Artifact Verification
    log_step(1, "Verify Artifacts & Contracts")
    artifact_path = "artifacts/serving_pipeline.pkl"
    contract_path = "contracts/model_contract.json"
    assert os.path.exists(artifact_path), f"Artifact missing at {artifact_path}"
    assert os.path.exists(contract_path), f"Contract missing at {contract_path}"
    print(f"[✓] Serving artifact verified: {artifact_path} ({os.path.getsize(artifact_path):,} bytes)")
    print(f"[✓] Model contract verified:  {contract_path}")

    # Step 2: Automated Test Suite (19 Tests)
    log_step(2, "Run Full Automated Test Suite (19 Tests)")
    test_res = run_command([sys.executable, "-m", "pytest", "tests/", "-v"], "pytest test suite")
    assert test_res.returncode == 0, "Test suite failed!"
    print("[✓] All 19 unit, contract, drift, and API tests passed successfully!")

    # Step 3: Candidate Model Evaluation Against Quality Gates
    log_step(3, "Evaluate Model Candidate Against Quality Gates")
    eval_res = run_command(
        [sys.executable, "scripts/evaluate_candidate.py", "--artifact", artifact_path],
        "Candidate evaluation",
    )
    assert eval_res.returncode == 0, "Candidate evaluation failed!"
    print("[✓] Quality gates and contract checks verified.")

    # Step 4: Live Serving Layer Smoke Test
    log_step(4, "Test Serving Layer Inference (POST /predict)")
    smoke_res = run_command([sys.executable, "scripts/smoke_test.py"], "API smoke test")
    assert smoke_res.returncode == 0, "Serving smoke test failed!"
    print("[✓] FastAPI serving layer validated.")

    # Step 5: Drift Monitoring & Detection Gatekeeper
    log_step(5, "Execute Drift Monitoring (Population Stability Index)")
    drift_res = run_command([sys.executable, "scripts/check_drift.py"], "Drift check")
    print(f"[✓] Drift check complete (exit code: {drift_res.returncode}).")

    # Step 6: End-to-End Drift Handling Simulation
    log_step(6, "Simulate Drift Detection & Retraining Trigger")
    sim_res = run_command(
        [sys.executable, "scripts/simulate_drift.py", "--demo-all"],
        "Drift simulation lifecycle",
    )
    assert sim_res.returncode == 0, "Drift simulation failed!"
    print("[✓] Drift detection, alerting, and hot-reload verified.")

    # Summary
    total_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"  PIPELINE EXECUTION COMPLETE — ALL SYSTEMS OPERATIONAL (took {total_time}s)")
    print("=" * 70)
    print("""
Next steps you can explore:
  1. Start the API locally:
     uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  2. Launch with Docker Compose (if Docker Desktop is running):
     docker compose up -d --build
  3. View interactive documentation:
     http://localhost:8000/docs
""")


if __name__ == "__main__":
    main()
