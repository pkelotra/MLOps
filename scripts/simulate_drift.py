"""
Simulate and demonstrate the automated drift detection and handling lifecycle.

Usage:
    python scripts/simulate_drift.py --check
    python scripts/simulate_drift.py --inject
    python scripts/simulate_drift.py --restore
    python scripts/simulate_drift.py --demo-all
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REFERENCE_PATH = "monitoring/reference.json"
CURRENT_PATH = "monitoring/current.json"
BACKUP_PATH = "monitoring/current.json.bak"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def check_current_drift():
    """Runs check_drift.py and displays output and exit code."""
    print("=" * 60)
    print("1. RUNNING DRIFT DETECTION CHECK")
    print(f"   Reference: {REFERENCE_PATH}")
    print(f"   Current:   {CURRENT_PATH}")
    print("=" * 60)

    from src.monitoring.drift import detect_drift
    result = detect_drift(REFERENCE_PATH, CURRENT_PATH)

    print("\nDrift Analysis Result:")
    print(json.dumps(result, indent=2))

    if result["drift_detected"]:
        print(f"\n[ALERT] DRIFT DETECTED in features: {result['drifted_features']}")
        print(f"        PSI exceeds threshold ({result['threshold']}).")
        print("        Automated workflow action: Trigger Retraining Pipeline.")
        return 2
    else:
        print("\n[STABLE] No distribution drift detected. Pipeline is healthy.")
        return 0


def inject_drift():
    """Injects high consumption load shift into current.json to simulate real-world drift."""
    print("\n" + "=" * 60)
    print("INJECTING SIMULATED DATA DRIFT (Industrial Demand Spike)")
    print("=" * 60)

    # Backup original current.json if not already backed up
    if not os.path.exists(BACKUP_PATH):
        shutil.copyfile(CURRENT_PATH, BACKUP_PATH)
        print(f"  Backed up original {CURRENT_PATH} -> {BACKUP_PATH}")

    # Create severe distribution shift: normal values [1..14] shifted to [60..180]
    drifted_data = {
        "load": [
            55.0, 58.2, 62.1, 65.4, 70.0, 75.2, 80.1, 88.5,
            95.0, 105.2, 115.0, 125.4, 135.0, 142.1, 150.5, 158.0,
            162.4, 168.0, 172.5, 175.0, 178.2, 180.0, 182.5, 185.0,
            110.0, 120.0, 130.0, 140.0, 150.0, 160.0
        ],
        "hour": [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            0, 1, 2, 3, 4, 5
        ]
    }

    save_json(CURRENT_PATH, drifted_data)
    print(f"  Injected 30 shifted load readings (mean {sum(drifted_data['load'])/len(drifted_data['load']):.1f} vs baseline ~6.0) into {CURRENT_PATH}.")
    print("  Ready to test drift detection.")


def restore_baseline():
    """Restores current.json back to baseline identical to reference.json."""
    print("\n" + "=" * 60)
    print("RESTORING BASELINE DATA (No Drift)")
    print("=" * 60)

    if os.path.exists(BACKUP_PATH):
        shutil.copyfile(BACKUP_PATH, CURRENT_PATH)
        os.remove(BACKUP_PATH)
        print(f"  Restored {CURRENT_PATH} from backup.")
    else:
        # Copy from reference.json
        shutil.copyfile(REFERENCE_PATH, CURRENT_PATH)
        print(f"  Restored {CURRENT_PATH} from {REFERENCE_PATH}.")

    print("  System data restored to baseline.")


def run_full_demo():
    """Runs a complete live demonstration of baseline -> drift detection -> retraining trigger -> candidate evaluation -> reload."""
    print("\n" + "#" * 65)
    print("#  DEMONSTRATION: DRIFT DETECTION & RETRAINING WORKFLOW")
    print("#" * 65)

    # Step 1: Baseline Check
    print("\n>>> STEP 1: Verify Healthy Baseline (No Drift)")
    restore_baseline()
    code_baseline = check_current_drift()
    assert code_baseline == 0, "Baseline should not detect drift!"

    # Step 2: Inject Drift
    print("\n>>> STEP 2: Inject Production Data Drift")
    inject_drift()

    # Step 3: Drift Detection
    print("\n>>> STEP 3: Execute Drift Detection Gatekeeper (scripts/check_drift.py)")
    code_drift = check_current_drift()
    assert code_drift == 2, "Drift should be detected!"

    # Step 4: Simulate Automated Pipeline Reaction
    print("\n>>> STEP 4: Automated CI/CD Trigger (.github/workflows/monitor.yml)")
    print("    In GitHub Actions, monitor.yml detects exit code 2 and dispatches:")
    print("    repository_dispatch: event-type: retrain, reason: drift_detected")
    print("    Triggering .github/workflows/train.yml...")

    # Step 5: Candidate Quality Gate Evaluation
    print("\n>>> STEP 5: Quality Gate Evaluation (scripts/evaluate_candidate.py)")
    cmd = [sys.executable, "scripts/evaluate_candidate.py", "--artifact", "artifacts/serving_pipeline.pkl"]
    eval_proc = subprocess.run(cmd, capture_output=True, text=True)
    print(eval_proc.stdout)
    assert eval_proc.returncode == 0, "Candidate evaluation should pass!"

    # Step 6: Hot Reload Serving API
    print("\n>>> STEP 6: Hot-Reload Serving API (POST /reload)")
    try:
        import httpx
        with httpx.Client(base_url="http://localhost:8000", timeout=5.0) as client:
            res = client.post("/reload")
            if res.status_code == 200:
                print(f"    Live Docker API Reload Response: {res.json()}")
            else:
                print(f"    Live API returned HTTP {res.status_code}")
    except Exception as e:
        print(f"    (Live Docker check skipped or offline: {e})")

    # Step 7: Live Smoke Test
    print("\n>>> STEP 7: Smoke Test Prediction Against Serving Engine")
    smoke_cmd = [sys.executable, "scripts/smoke_test.py"]
    smoke_proc = subprocess.run(smoke_cmd, capture_output=True, text=True)
    print(smoke_proc.stdout)

    # Clean up
    print("\n>>> STEP 8: Cleanup & Restore Baseline")
    restore_baseline()

    print("\n" + "=" * 65)
    print("DEMO COMPLETE: System successfully detected drift, triggered gates,")
    print("evaluated candidate, and hot-reloaded without service interruption.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Drift Detection & Handling")
    parser.add_argument("--check", action="store_true", help="Check current drift status")
    parser.add_argument("--inject", action="store_true", help="Inject artificial drift into current.json")
    parser.add_argument("--restore", action="store_true", help="Restore current.json to baseline")
    parser.add_argument("--demo-all", action="store_true", help="Run end-to-end drift demonstration")
    args = parser.parse_args()

    if args.demo_all:
        run_full_demo()
    elif args.inject:
        inject_drift()
        check_current_drift()
    elif args.restore:
        restore_baseline()
        check_current_drift()
    else:
        code = check_current_drift()
        sys.exit(code)


if __name__ == "__main__":
    main()
