import json, os
import numpy as np

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.20"))

def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _psi(expected, actual, bins=10):
    if len(expected) < 20 or len(actual) < 20:
        raise ValueError("At least 20 reference/current observations are recommended")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)
    ep = np.clip(e / len(expected), 1e-6, None)
    ap = np.clip(a / len(actual), 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))

def detect_drift(reference_path, current_path):
    reference, current = _load(reference_path), _load(current_path)
    features, drifted = {}, []
    for name, ref in reference.items():
        if name not in current: continue
        score = _psi(ref, current[name])
        flag = score >= DRIFT_THRESHOLD
        features[name] = {"psi": score, "drifted": flag}
        if flag: drifted.append(name)
    return {"drift_detected": bool(drifted), "drifted_features": drifted, "threshold": DRIFT_THRESHOLD, "features": features}
