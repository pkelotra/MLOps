from src.monitoring.drift import detect_drift

def test_no_drift_for_identical_distributions(tmp_path):
    p = '{"x": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}'
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    a.write_text(p); b.write_text(p)
    assert detect_drift(str(a), str(b))["drift_detected"] is False
