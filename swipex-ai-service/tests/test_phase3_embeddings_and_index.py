"""Tests for Phase 3: EmbeddingService, VectorStore, and /index/* endpoints."""

import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import VectorStoreUnavailableError
from app.core.task_registry import TaskRegistry, TaskStatus
from app.services.ai.embeddings import FakeEmbeddingService, set_embedding_service

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def inject_fake_embedding():
    """Inject FakeEmbeddingService before every test; restore after."""
    svc = FakeEmbeddingService()
    set_embedding_service(svc)
    yield svc


@pytest.fixture()
def fake_svc() -> FakeEmbeddingService:
    return FakeEmbeddingService()


# ---------------------------------------------------------------------------
# FakeEmbeddingService: shape, dtype, norm, determinism
# ---------------------------------------------------------------------------


class TestFakeEmbeddingService:
    def test_shape_and_dtype(self, fake_svc):
        texts = ["python developer", "machine learning engineer"]
        vecs = fake_svc.encode(texts)
        assert vecs.shape == (2, settings.EMBEDDING_DIM)
        assert vecs.dtype == np.float32

    def test_empty_input(self, fake_svc):
        vecs = fake_svc.encode([])
        assert vecs.shape == (0, settings.EMBEDDING_DIM)

    def test_unit_norm(self, fake_svc):
        texts = ["hello world", "fast api", "chromadb"]
        vecs = fake_svc.encode(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self, fake_svc):
        texts = ["react", "typescript"]
        vecs1 = fake_svc.encode(texts)
        vecs2 = fake_svc.encode(texts)
        np.testing.assert_array_equal(vecs1, vecs2)

    def test_different_texts_different_vecs(self, fake_svc):
        a = fake_svc.encode(["python"])
        b = fake_svc.encode(["java"])
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# EmbeddingService: LRU cache correctness (using FakeEmbeddingService)
# ---------------------------------------------------------------------------


class TestEmbeddingLRUCache:
    def test_cache_hit_returns_same_vector(self):
        """Second encode of the same text must return the cached vector."""
        from app.services.ai.embeddings import get_embedding_service

        svc = get_embedding_service()
        texts = ["python is great"]
        v1 = svc.encode(texts)
        v2 = svc.encode(texts)
        np.testing.assert_array_equal(v1, v2)

    def test_partial_cache_miss(self):
        """Mix of cached and new texts should return the correct shape."""
        from app.services.ai.embeddings import get_embedding_service

        svc = get_embedding_service()
        first = ["cached text one", "cached text two"]
        svc.encode(first)
        second = ["cached text one", "brand new text"]
        vecs = svc.encode(second)
        assert vecs.shape == (2, settings.EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# BaseEmbeddingService helpers: cosine_matrix, top_k
# ---------------------------------------------------------------------------


class TestEmbeddingHelpers:
    def test_cosine_matrix_self_similarity(self, fake_svc):
        texts = ["python", "java", "rust"]
        vecs = fake_svc.encode(texts)
        sim = fake_svc.cosine_matrix(vecs, vecs)
        # Diagonal must be ≈ 1.0 (unit vectors dot self)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_top_k_returns_correct_count(self, fake_svc):
        vecs = fake_svc.encode(["a", "b", "c", "d", "e"])
        query = vecs[0:1]
        sim = fake_svc.cosine_matrix(query, vecs)[0]
        indices = fake_svc.top_k(sim, k=3)
        assert len(indices) == 3

    def test_top_k_clamps_to_available(self, fake_svc):
        vecs = fake_svc.encode(["x", "y"])
        sim = fake_svc.cosine_matrix(vecs[0:1], vecs)[0]
        indices = fake_svc.top_k(sim, k=100)
        assert len(indices) == 2


# ---------------------------------------------------------------------------
# Aencode (async wrapper) — runs in thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aencode_returns_correct_shape(fake_svc):
    texts = ["async encode test", "another text"]
    vecs = await fake_svc.aencode(texts)
    assert vecs.shape == (2, settings.EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_create_and_get(self):
        reg = TaskRegistry()
        rec = reg.create("task-123")
        assert rec.task_id == "task-123"
        assert rec.status == TaskStatus.QUEUED

        retrieved = reg.get("task-123")
        assert retrieved is not None
        assert retrieved.task_id == "task-123"

    def test_get_missing(self):
        reg = TaskRegistry()
        assert reg.get("nonexistent") is None

    def test_update(self):
        reg = TaskRegistry()
        rec = reg.create("task-456")
        rec.status = TaskStatus.RUNNING
        rec.indexed = 42
        reg.update(rec)

        updated = reg.get("task-456")
        assert updated.status == TaskStatus.RUNNING
        assert updated.indexed == 42

    def test_duration_before_finish_is_none(self):
        reg = TaskRegistry()
        rec = reg.create("task-789")
        rec.started_at = time.time()
        assert rec.duration_s is None

    def test_duration_after_finish(self):
        reg = TaskRegistry()
        rec = reg.create("task-000")
        rec.started_at = time.time() - 1.5
        rec.finished_at = time.time()
        assert rec.duration_s is not None
        assert rec.duration_s >= 1.0

    def test_list_recent_ordering(self):
        reg = TaskRegistry()
        for i in range(5):
            reg.create(f"task-{i}")
            time.sleep(0.01)
        recent = reg.list_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].created_at >= recent[1].created_at

    def test_eviction_of_expired_records(self):
        reg = TaskRegistry()
        reg._TTL = 0.05  # 50 ms TTL for fast test
        reg.create("expire-me")
        time.sleep(0.1)
        # Creating a new record triggers eviction
        reg.create("new-one")
        assert reg.get("expire-me") is None


# ---------------------------------------------------------------------------
# VectorStore — using an in-memory mock Chroma collection
# ---------------------------------------------------------------------------


def _make_mock_chroma_collection(count: int = 0):
    """Build a MagicMock that mimics a chromadb.Collection."""
    coll = MagicMock()
    coll.count.return_value = count
    coll.metadata = {
        "hnsw:space": "cosine",
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": settings.EMBEDDING_DIM,
    }
    # get() used for dedup check — return empty by default
    coll.get.return_value = {"ids": [], "metadatas": []}
    # upsert returns None
    coll.upsert.return_value = None
    # query returns synthetic results
    coll.query.return_value = {
        "ids": [["job-001", "job-002"]],
        "distances": [[0.15, 0.35]],
    }
    return coll


def _make_job_input(**kwargs):
    from app.schemas.job import JobInput

    defaults = {
        "job_id": str(uuid.uuid4()),
        "title": "Software Engineer",
        "description": "Build scalable APIs using Python and FastAPI.",
        "required_skills": ["python", "fastapi"],
        "skills": ["python", "fastapi", "sql"],
        "keywords": ["backend", "api"],
    }
    defaults.update(kwargs)
    return JobInput(**defaults)


@pytest.fixture()
def patched_vector_store():
    """VectorStore with a mocked ChromaDB client/collection."""
    from app.services.ai.vector_store import VectorStore

    store = VectorStore()
    store._collection = _make_mock_chroma_collection(count=5)
    store._status = "ok"
    store._client = MagicMock()
    return store


@pytest.mark.asyncio
async def test_vector_store_upsert_returns_indexed_count(patched_vector_store):
    jobs = [_make_job_input(job_id=f"job-{i}") for i in range(3)]
    n = await patched_vector_store.upsert_jobs(jobs)
    assert n == 3  # all are cache misses (empty existing)


@pytest.mark.asyncio
async def test_vector_store_upsert_skips_unchanged(patched_vector_store):
    """If dedup check returns same content hash for a job, it should be skipped."""
    from app.services.ai.vector_store import _content_hash, _make_job_text

    job = _make_job_input(job_id="job-stable")
    text = _make_job_text(job)
    chash = _content_hash(text)
    # Simulate existing metadata with same hash
    patched_vector_store._collection.get.return_value = {
        "ids": ["job-stable"],
        "metadatas": [{"content_hash": chash}],
    }
    n = await patched_vector_store.upsert_jobs([job])
    assert n == 0  # unchanged → skip


@pytest.mark.asyncio
async def test_vector_store_query(patched_vector_store):
    from app.services.ai.embeddings import get_embedding_service

    svc = get_embedding_service()
    vecs = svc.encode(["python developer"])
    results = await patched_vector_store.query(vecs[0], k=5)
    assert len(results) == 2
    job_id, sim = results[0]
    assert job_id == "job-001"
    assert pytest.approx(sim, abs=1e-3) == 0.85  # 1.0 - 0.15


@pytest.mark.asyncio
async def test_vector_store_degraded_query_returns_empty():
    from app.services.ai.vector_store import VectorStore

    store = VectorStore()
    store._status = "degraded"
    vec = np.ones(settings.EMBEDDING_DIM, dtype=np.float32) / np.sqrt(settings.EMBEDDING_DIM)
    results = await store.query(vec, k=5)
    assert results == []


@pytest.mark.asyncio
async def test_vector_store_count(patched_vector_store):
    count = await patched_vector_store.count()
    assert count == 5


@pytest.mark.asyncio
async def test_vector_store_unavailable_error_on_degraded_upsert():
    from app.services.ai.vector_store import VectorStore

    store = VectorStore()
    store._status = "degraded"
    with pytest.raises(VectorStoreUnavailableError):
        store._require_ok()


# ---------------------------------------------------------------------------
# /index/* API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def index_client(patched_vector_store):
    """Test client with FakeEmbeddingService + patched VectorStore injected."""
    from fastapi.testclient import TestClient

    from app.main import create_application
    from app.services.ai.embeddings import set_embedding_service
    from app.services.ai.vector_store import set_vector_store

    test_app = create_application()
    set_embedding_service(FakeEmbeddingService())
    set_vector_store(patched_vector_store)

    with TestClient(test_app, raise_server_exceptions=True) as client:
        yield client


def _auth_headers():
    return {"X-Internal-API-Key": "test-secret"}


def test_bulk_index_returns_202(index_client):
    payload = {
        "jobs": [
            {
                "job_id": "job-test-1",
                "title": "ML Engineer",
                "description": "Build ML models",
                "required_skills": ["python", "tensorflow"],
                "skills": ["python", "tensorflow"],
            }
        ]
    }
    resp = index_client.post("/api/v1/index/jobs", json=payload, headers=_auth_headers())
    assert resp.status_code == 202
    body = resp.json()
    assert "task_id" in body
    assert body["job_count"] == 1


def test_bulk_index_requires_auth(index_client):
    payload = {"jobs": [{"job_id": "j1", "title": "Dev", "required_skills": []}]}
    resp = index_client.post("/api/v1/index/jobs", json=payload)
    assert resp.status_code in (401, 403)


def test_task_status_not_found(index_client):
    resp = index_client.get("/api/v1/index/tasks/nonexistent-id", headers=_auth_headers())
    assert resp.status_code == 422  # InvalidInputError maps to 422


def test_task_status_found(index_client):
    """Queue a task and then immediately poll its status."""
    payload = {
        "jobs": [
            {"job_id": "job-poll-1", "title": "Data Scientist", "required_skills": ["python", "spark"]}
        ]
    }
    resp = index_client.post("/api/v1/index/jobs", json=payload, headers=_auth_headers())
    task_id = resp.json()["task_id"]

    status_resp = index_client.get(f"/api/v1/index/tasks/{task_id}", headers=_auth_headers())
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["task_id"] == task_id
    assert body["status"] in ("queued", "running", "done", "failed")


def test_reindex_endpoint(index_client):
    resp = index_client.post("/api/v1/index/reindex", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert "reset" in body["message"].lower()
