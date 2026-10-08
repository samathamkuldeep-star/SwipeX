"""In-memory background task registry with TTL expiry and thread-safe access."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # Duration (seconds), filled on finish
    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 3)
        return None


class TaskRegistry:
    """Thread-safe in-memory store for background task records.

    Old records are evicted once they exceed TTL (default 3600 s).
    """

    _TTL: float = 3600.0  # 1 hour

    def __init__(self) -> None:
        self._store: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str) -> TaskRecord:
        rec = TaskRecord(task_id=task_id)
        with self._lock:
            self._evict_expired()
            self._store[task_id] = rec
        return rec

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._store.get(task_id)

    def update(self, rec: TaskRecord) -> None:
        """Re-write the record (caller must hold a reference to it)."""
        with self._lock:
            self._store[rec.task_id] = rec

    def _evict_expired(self) -> None:
        now = time.time()
        stale = [tid for tid, r in self._store.items() if now - r.created_at > self._TTL]
        for tid in stale:
            del self._store[tid]

    def list_recent(self, limit: int = 50) -> list[TaskRecord]:
        with self._lock:
            records = sorted(self._store.values(), key=lambda r: r.created_at, reverse=True)
        return records[:limit]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_task_registry: TaskRegistry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    return _task_registry
