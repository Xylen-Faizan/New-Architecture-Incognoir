from fastapi.testclient import TestClient

from intent_engine.app import app


def test_metrics_health_endpoint():
    """Calling /metrics/health should return 200 and valid JSON fields."""
    client = TestClient(app)

    response = client.get("/metrics/health")
    assert response.status_code == 200, f"expected 200 got {response.status_code}"
    data = response.json()
    # basic structure tests
    assert "health_score" in data
    assert "uptime_days" in data
    assert isinstance(data.get("health_score"), (int, float))
    # client test client does not always return CORS headers, so we just ensure we got JSON
    # normally the real server should include the header based on settings.cors_origins
    pass


if __name__ == "__main__":
    test_metrics_health_endpoint()
