from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    # Check that 'status' is 'healthy', regardless of what else is in the JSON
    assert response.json()["status"] == "healthy"

def test_metrics_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text