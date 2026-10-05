import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring.drift import detect_drift

r = detect_drift(
    os.getenv("REFERENCE_DATA_PATH", "monitoring/reference.json"),
    os.getenv("CURRENT_DATA_PATH", "monitoring/current.json"),
)
print(json.dumps(r, indent=2))
raise SystemExit(2 if r["drift_detected"] else 0)
