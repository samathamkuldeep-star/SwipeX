"""TTL/LRU caches and cache key helpers."""

import hashlib
import threading
from typing import Any

import cachetools

from app.core.config import settings


def make_embed_cache_key(text: str) -> str:
    """SHA-256 cache key for an embedding entry (model + text)."""
    payload = settings.EMBEDDING_MODEL + "\n" + text.strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_llm_cache_key(prompt: str) -> str:
    """SHA-256 cache key for an LLM suggestion entry."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# Module-level embedding LRU cache (shared across the process)
_embed_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=settings.EMBED_CACHE_SIZE)
_embed_cache_lock: threading.Lock = threading.Lock()


def get_embed_cache() -> tuple[cachetools.LRUCache, threading.Lock]:
    """Return the global embedding LRU cache and its lock."""
    return _embed_cache, _embed_cache_lock


# Module-level TTL cache for LLM suggestions
_llm_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=512,
    ttl=settings.LLM_CACHE_TTL_SECONDS,
)
_llm_cache_lock: threading.Lock = threading.Lock()


def get_llm_cache() -> tuple[cachetools.TTLCache, threading.Lock]:
    """Return the global LLM TTL cache and its lock."""
    return _llm_cache, _llm_cache_lock


def get_from_cache(cache: Any, key: str, lock: threading.Lock) -> Any | None:
    """Thread-safe cache get."""
    with lock:
        return cache.get(key)


def set_in_cache(cache: Any, key: str, value: Any, lock: threading.Lock) -> None:
    """Thread-safe cache set."""
    with lock:
        cache[key] = value
