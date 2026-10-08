"""Pytest configuration and global test fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client():
    """TestClient instance for API endpoints."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    """Default headers with internal API key when configured."""
    if settings.INTERNAL_API_KEY:
        return {"X-Internal-Api-Key": settings.INTERNAL_API_KEY}
    return {}
