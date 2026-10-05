from fastapi.testclient import TestClient
from src.api.main import app

def test_health_endpoint():
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert 'models_ready' in response.json()
