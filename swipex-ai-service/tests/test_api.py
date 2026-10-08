"""Tests for health, security, middleware, and exception handling."""

from app.core.config import settings


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "X-Request-ID" in response.headers


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded", "reindex_required")
    assert "embedding_model_loaded" in data
    assert "spacy_loaded" in data
    assert "llm_enabled" in data


def test_request_id_middleware(client):
    custom_id = "test-request-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_api_key_security(client, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "secret_key_123")

    # Missing API Key -> 401
    res_no_key = client.get("/api/v1/ready")
    assert res_no_key.status_code == 401
    err = res_no_key.json()["error"]
    assert err["code"] == "UNAUTHORIZED"
    assert "request_id" in err

    # Invalid API Key -> 401
    res_bad_key = client.get("/api/v1/ready", headers={"X-Internal-Api-Key": "wrong_key"})
    assert res_bad_key.status_code == 401
    assert res_bad_key.json()["error"]["code"] == "UNAUTHORIZED"

    # Valid API Key -> 200
    res_valid = client.get("/api/v1/ready", headers={"X-Internal-Api-Key": "secret_key_123"})
    assert res_valid.status_code == 200
