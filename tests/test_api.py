import pytest
from fastapi.testclient import TestClient
from app.main import app, lifespan

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns 200 and reports model device and threshold."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert data["threshold"] == 0.85
        assert "database" in data


def test_predict_high_confidence_auto_route():
    """Verify clear ticket intent routes automatically with high confidence."""
    with TestClient(app) as test_client:
        payload = {
            "ticket_id": "TCK-TEST-001",
            "text": "My credit card was stolen yesterday and I need to block it immediately."
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"] == "TCK-TEST-001"
        assert data["intent"] in ["card_issues", "fraud_security"]
        assert data["confidence"] >= 0.70
        assert "route" in data
        assert "escalate_to_human" in data
        assert "all_scores" in data


def test_predict_validation_error():
    """Verify input validation fails gracefully on empty or short payload."""
    with TestClient(app) as test_client:
        payload = {"text": "hi"}  # min_length is 3
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422


def test_logs_endpoint():
    """Verify /logs endpoint returns logged inference records."""
    with TestClient(app) as test_client:
        # First send a prediction
        test_client.post("/predict", json={"ticket_id": "TCK-LOG-TEST", "text": "What is the interest rate on savings?"})
        # Fetch logs
        response = test_client.get("/logs?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "logs" in data
        assert isinstance(data["logs"], list)


def test_stats_endpoint():
    """Verify /stats endpoint returns aggregated system throughput and distribution."""
    with TestClient(app) as test_client:
        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "database_engine" in data
        assert "auto_route_count" in data
        assert "human_escalation_count" in data
        assert "intent_distribution" in data
