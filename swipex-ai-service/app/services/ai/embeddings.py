"""Sentence-Transformers embedding singleton with LRU cache, dimension guards, and async wrapper."""

import asyncio
import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from app.core.cache import get_embed_cache, get_from_cache, make_embed_cache_key, set_in_cache
from app.core.config import settings
from app.core.exceptions import EmbeddingDimensionError

logger = logging.getLogger("swipex.ai.embeddings")

_ai_semaphore: asyncio.Semaphore | None = None


def get_ai_semaphore() -> asyncio.Semaphore:
    """Return a process-level semaphore for AI task concurrency. Created on first use."""
    global _ai_semaphore
    if _ai_semaphore is None:
        _ai_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AI_TASKS)
    return _ai_semaphore


# ---------------------------------------------------------------------------
# Abstract base — lets tests swap in FakeEmbeddingService
# ---------------------------------------------------------------------------


class BaseEmbeddingService(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        ...

    async def aencode(self, texts: list[str]) -> np.ndarray:
        sem = get_ai_semaphore()
        async with sem:
            return await asyncio.to_thread(self.encode, texts)

    # ------------------------------------------------------------------
    # Utility helpers (shared for all implementations)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between rows of a and b.

        Both arrays must be L2-normalised (so cosine = dot product).
        Returns shape (len(a), len(b)).
        """
        return np.matmul(a, b.T)

    @staticmethod
    def top_k(sim_vector: np.ndarray, k: int) -> np.ndarray:
        """Return indices of top-k scores (descending) from a 1-D similarity vector."""
        k = min(k, len(sim_vector))
        idx = np.argpartition(sim_vector, -k)[-k:]
        return idx[np.argsort(sim_vector[idx])[::-1]]


# ---------------------------------------------------------------------------
# Production embedding service
# ---------------------------------------------------------------------------


class EmbeddingService(BaseEmbeddingService):
    """Sentence-transformers MiniLM-L6-v2 singleton, warm-started in lifespan."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._model_lock = threading.Lock()
        self._cache, self._cache_lock = get_embed_cache()

        # Warm-up encode so model weights are loaded before first request
        dummy = self._raw_encode(["warm-up"])
        self._validate(dummy)
        logger.info(
            f"EmbeddingService ready — model={settings.EMBEDDING_MODEL} dim={settings.EMBEDDING_DIM}"
        )

    def _raw_encode(self, texts: list[str]) -> np.ndarray:
        with self._model_lock:
            vecs = self._model.encode(
                texts,
                batch_size=settings.EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vecs.astype(np.float32)

    def _validate(self, vecs: np.ndarray) -> None:
        if vecs.shape[-1] != settings.EMBEDDING_DIM:
            raise EmbeddingDimensionError(
                f"Model returned dim={vecs.shape[-1]}, expected {settings.EMBEDDING_DIM}."
            )
        if vecs.dtype != np.float32:
            raise EmbeddingDimensionError(
                f"Expected float32 embeddings, got {vecs.dtype}."
            )
        if not np.all(np.isfinite(vecs)):
            raise EmbeddingDimensionError("Embedding output contains NaN or Inf values.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts, using LRU cache for previously seen strings.

        Returns shape (n, EMBEDDING_DIM), dtype float32, L2-normalised.
        """
        if not texts:
            return np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)

        cache, lock = self._cache, self._cache_lock
        keys = [make_embed_cache_key(t) for t in texts]

        # Collect cache hits and misses
        result: dict[int, np.ndarray] = {}
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, key in enumerate(keys):
            cached = get_from_cache(cache, key, lock)
            if cached is not None:
                result[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(texts[i])

        # Encode only cache misses
        if miss_texts:
            new_vecs = self._raw_encode(miss_texts)
            self._validate(new_vecs)
            for j, idx in enumerate(miss_indices):
                vec = new_vecs[j]
                set_in_cache(cache, keys[idx], vec, lock)
                result[idx] = vec

        # Assemble in original order
        out = np.stack([result[i] for i in range(len(texts))], axis=0)
        return out


# ---------------------------------------------------------------------------
# Fake embedding service for tests — deterministic, no model download
# ---------------------------------------------------------------------------


class FakeEmbeddingService(BaseEmbeddingService):
    """Deterministic hash-seeded unit-vector embeddings for tests (no torch required)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(settings.EMBEDDING_DIM).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v)
        return np.stack(vecs, axis=0) if vecs else np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)


# Need hashlib in FakeEmbeddingService
import hashlib

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedding_service: BaseEmbeddingService | None = None
_embedding_loaded: bool = False
_embedding_error: str | None = None


def get_embedding_service() -> BaseEmbeddingService:
    return _embedding_service  # type: ignore[return-value]


def is_embedding_loaded() -> bool:
    return _embedding_loaded


def get_embedding_error() -> str | None:
    return _embedding_error


def load_embedding_service() -> None:
    """Load the production embedding service. Call once in the FastAPI lifespan."""
    global _embedding_service, _embedding_loaded, _embedding_error
    try:
        _embedding_service = EmbeddingService()
        _embedding_loaded = True
        _embedding_error = None
        logger.info("Embedding service loaded successfully.")
    except Exception as exc:
        _embedding_error = str(exc)
        _embedding_loaded = False
        logger.error(f"Failed to load embedding service: {exc}")


def set_embedding_service(svc: BaseEmbeddingService) -> None:
    """Inject a custom embedding service (used in tests for DI override)."""
    global _embedding_service, _embedding_loaded
    _embedding_service = svc
    _embedding_loaded = True
