"""ChromaDB vector store wrapper with model/dim guards and graceful degradation."""

import asyncio
import hashlib
import logging

import numpy as np

from app.core.config import settings
from app.core.exceptions import VectorStoreUnavailableError
from app.schemas.job import JobInput
from app.services.ai import embeddings as emb_module

logger = logging.getLogger("swipex.ai.vector_store")

# Chroma status values exposed to /ready
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_REINDEX = "reindex_required"


def _make_job_text(job: "JobInput") -> str:
    """Build the canonical text blob for a job that gets embedded."""
    skills_str = ", ".join(job.required_skills + job.skills)
    return f"{job.title}. {job.description} Skills: {skills_str}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class VectorStore:
    """Thin async wrapper around a ChromaDB collection with model/dim guards."""

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._status: str = STATUS_DEGRADED
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Initialisation (called in lifespan, blocking — run via to_thread)
    # ------------------------------------------------------------------

    def _init_sync(self) -> None:
        """Connect to ChromaDB and validate model/dim metadata."""
        try:
            import chromadb

            if settings.CHROMA_MODE == "http":
                self._client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                )
            else:
                self._client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

            self._collection = self._client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "embedding_dim": settings.EMBEDDING_DIM,
                },
            )

            # Guard: if existing collection has different model/dim → reindex required
            stored_meta = self._collection.metadata or {}
            stored_model = stored_meta.get("embedding_model", settings.EMBEDDING_MODEL)
            stored_dim = stored_meta.get("embedding_dim", settings.EMBEDDING_DIM)

            if stored_model != settings.EMBEDDING_MODEL or stored_dim != settings.EMBEDDING_DIM:
                self._status = STATUS_REINDEX
                self._error = (
                    f"Stored model={stored_model}/dim={stored_dim} does not match "
                    f"configured model={settings.EMBEDDING_MODEL}/dim={settings.EMBEDDING_DIM}. "
                    "Run POST /index/reindex to rebuild."
                )
                logger.error(self._error)
                return

            self._status = STATUS_OK
            logger.info(f"VectorStore ready — collection={settings.CHROMA_COLLECTION}, count={self._collection.count()}")

        except Exception as exc:
            self._status = STATUS_DEGRADED
            self._error = str(exc)
            logger.warning(f"ChromaDB unavailable (degraded mode): {exc}")

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    def _require_ok(self) -> None:
        """Raise VectorStoreUnavailableError unless status is ok."""
        if self._status == STATUS_REINDEX:
            raise VectorStoreUnavailableError(
                f"Vector store requires reindexing before queries can be served. {self._error}"
            )
        if self._status == STATUS_DEGRADED or self._collection is None:
            raise VectorStoreUnavailableError(
                f"Vector store is currently unavailable. {self._error}"
            )

    # ------------------------------------------------------------------
    # Upsert jobs
    # ------------------------------------------------------------------

    def _upsert_sync(self, jobs: list) -> int:
        """Synchronous batch upsert with content-hash deduplication."""
        self._require_ok()
        svc = emb_module.get_embedding_service()

        # Determine which jobs need re-embedding
        texts: list[str] = []
        hashes: list[str] = []
        for job in jobs:
            text = _make_job_text(job)
            texts.append(text)
            hashes.append(_content_hash(text))

        # Check existing metadata to skip unchanged jobs
        ids = [job.job_id for job in jobs]
        try:
            existing = self._collection.get(ids=ids, include=["metadatas"])
            existing_hash_map: dict[str, str] = {
                eid: (emeta.get("content_hash", "") if emeta else "")
                for eid, emeta in zip(existing["ids"], existing["metadatas"])
            }
        except Exception:
            existing_hash_map = {}

        new_ids, new_texts, new_meta, need_embed_indices = [], [], [], []
        for i, (job, text, chash) in enumerate(zip(jobs, texts, hashes)):
            if existing_hash_map.get(job.job_id) == chash:
                continue  # Skip unchanged
            new_ids.append(job.job_id)
            new_texts.append(text)
            posted_ts = job.posted_at.isoformat() if job.posted_at else ""
            new_meta.append({
                "content_hash": chash,
                "company_type": job.company_type or "",
                "work_mode": job.work_mode or "",
                "competition_level": job.competition_level or "",
                "posted_at": posted_ts,
                "applicant_count": job.applicant_count or 0,
                "title": job.title,
            })
            need_embed_indices.append(i)

        if not new_ids:
            return 0

        # Embed in batches
        vecs = svc.encode(new_texts)
        self._collection.upsert(
            ids=new_ids,
            embeddings=vecs.tolist(),
            metadatas=new_meta,
        )
        return len(new_ids)

    async def upsert_jobs(self, jobs: list) -> int:
        return await asyncio.to_thread(self._upsert_sync, jobs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _query_sync(
        self,
        vector: np.ndarray,
        k: int,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Query the collection; returns list of (job_id, similarity) tuples."""
        self._require_ok()
        params: dict = {
            "query_embeddings": [vector.tolist()],
            "n_results": min(k, max(1, self._collection.count())),
            "include": ["distances"],
        }
        if where:
            params["where"] = where

        results = self._collection.query(**params)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Chroma cosine: distance = 1 - cosine_similarity → similarity = 1 - distance
        return [(jid, round(1.0 - float(dist), 4)) for jid, dist in zip(ids, distances)]

    async def query(
        self,
        vector: np.ndarray,
        k: int,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Async query — falls back to empty list when degraded (caller handles)."""
        if self._status != STATUS_OK:
            return []
        return await asyncio.to_thread(self._query_sync, vector, k, where)

    # ------------------------------------------------------------------
    # Delete / count / reset
    # ------------------------------------------------------------------

    def _delete_sync(self, ids: list[str]) -> None:
        self._require_ok()
        self._collection.delete(ids=ids)

    async def delete_jobs(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._delete_sync, ids)

    def _count_sync(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _reset_sync(self) -> None:
        """Drop and recreate the collection (dev / reindex only)."""
        if self._client is None:
            raise VectorStoreUnavailableError("ChromaDB client not initialised.")
        self._client.delete_collection(settings.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dim": settings.EMBEDDING_DIM,
            },
        )
        self._status = STATUS_OK
        logger.info("VectorStore collection reset.")

    async def reset(self) -> None:
        await asyncio.to_thread(self._reset_sync)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def set_vector_store(store: VectorStore) -> None:
    """Inject a custom store (used in tests)."""
    global _vector_store
    _vector_store = store
