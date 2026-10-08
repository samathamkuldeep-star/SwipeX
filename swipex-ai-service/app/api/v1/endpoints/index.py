"""Index API endpoints: POST /index/jobs, GET /index/tasks/{task_id}, POST /index/reindex."""

import asyncio
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import verify_api_key
from app.core.task_registry import TaskRecord, TaskStatus, get_task_registry
from app.schemas.job import JobInput
from app.services.ai.embeddings import is_embedding_loaded
from app.services.ai.vector_store import STATUS_OK, VectorStore, get_vector_store

router = APIRouter(prefix="/index", tags=["indexing"])
logger = logging.getLogger("swipex.api.index")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class BulkIndexResponse(BaseModel):
    task_id: str = Field(..., description="Async task ID to poll for progress")
    message: str = "Indexing task accepted"
    job_count: int = Field(..., description="Number of jobs submitted")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    total: int
    indexed: int
    skipped: int
    duration_s: float | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class ReindexResponse(BaseModel):
    message: str
    previous_count: int
    task_id: str


class IndexBulkRequest(BaseModel):
    jobs: list[JobInput] = Field(..., min_length=1, max_length=10_000)


# ---------------------------------------------------------------------------
# Background indexing worker
# ---------------------------------------------------------------------------


async def _run_index_worker(
    rec: TaskRecord,
    jobs: list[JobInput],
    store: VectorStore,
) -> None:
    """Batch-embed and upsert jobs into ChromaDB. Updates TaskRecord throughout."""
    registry = get_task_registry()
    rec.status = TaskStatus.RUNNING
    rec.started_at = time.time()
    rec.total = len(jobs)
    registry.update(rec)

    try:
        total_indexed = 0
        total_skipped = 0
        batch_size = settings.INDEX_BATCH_SIZE

        for batch_start in range(0, len(jobs), batch_size):
            batch = jobs[batch_start : batch_start + batch_size]
            n_indexed = await store.upsert_jobs(batch)
            n_skipped = len(batch) - n_indexed
            total_indexed += n_indexed
            total_skipped += n_skipped

            # Update progress incrementally
            rec.indexed = total_indexed
            rec.skipped = total_skipped
            registry.update(rec)

            # Yield control after each batch so the event loop stays responsive
            await asyncio.sleep(0)

        rec.status = TaskStatus.DONE
    except Exception as exc:
        logger.exception(f"Indexing task {rec.task_id} failed")
        rec.status = TaskStatus.FAILED
        rec.error = str(exc)
    finally:
        rec.finished_at = time.time()
        registry.update(rec)
        logger.info(
            f"Index task {rec.task_id}: status={rec.status} indexed={rec.indexed} "
            f"skipped={rec.skipped} duration={rec.duration_s}s"
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=BulkIndexResponse,
    summary="Bulk-index jobs into the vector store",
    responses={
        202: {"description": "Indexing task queued"},
        503: {"description": "Embedding model or vector store unavailable"},
    },
)
async def bulk_index_jobs(
    payload: IndexBulkRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(verify_api_key)],
) -> BulkIndexResponse:
    """Accept up to 10 000 jobs, embed them in batches, and upsert into ChromaDB.

    Returns a `task_id` immediately; use `GET /index/tasks/{task_id}` to poll progress.
    Returns 503 if the embedding model or vector store is not ready.
    """
    if not is_embedding_loaded():
        from app.core.exceptions import VectorStoreUnavailableError

        raise VectorStoreUnavailableError("Embedding model is not loaded yet. Retry after /ready returns ok.")

    store = get_vector_store()
    if store.status != STATUS_OK:
        from app.core.exceptions import VectorStoreUnavailableError

        raise VectorStoreUnavailableError(
            f"Vector store is not ready (status={store.status}). {store.error or ''}"
        )

    task_id = str(uuid.uuid4())
    registry = get_task_registry()
    rec = registry.create(task_id)

    background_tasks.add_task(_run_index_worker, rec, payload.jobs, store)
    logger.info(f"Index task {task_id} queued — {len(payload.jobs)} jobs")

    return BulkIndexResponse(task_id=task_id, job_count=len(payload.jobs))


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll the status of a background indexing task",
    responses={
        200: {"description": "Task status"},
        404: {"description": "Task not found"},
    },
)
async def get_task_status(
    task_id: str,
    _: Annotated[None, Depends(verify_api_key)],
) -> TaskStatusResponse:
    """Return the current progress of a background indexing task."""
    registry = get_task_registry()
    rec = registry.get(task_id)
    if rec is None:
        from app.core.exceptions import InvalidInputError

        raise InvalidInputError(f"Task '{task_id}' not found (may have expired or never existed).")

    return TaskStatusResponse(
        task_id=rec.task_id,
        status=rec.status.value,
        total=rec.total,
        indexed=rec.indexed,
        skipped=rec.skipped,
        duration_s=rec.duration_s,
        error=rec.error,
        created_at=rec.created_at,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
    )


@router.post(
    "/reindex",
    status_code=status.HTTP_200_OK,
    response_model=ReindexResponse,
    summary="Drop and recreate the ChromaDB collection (admin only)",
    responses={
        200: {"description": "Collection reset; submit /index/jobs to repopulate"},
        503: {"description": "Vector store unavailable"},
    },
)
async def reindex(
    background_tasks: BackgroundTasks,
    _: Annotated[None, Depends(verify_api_key)],
) -> ReindexResponse:
    """Drop the existing collection and recreate it empty.

    Use this when the embedding model has been changed (model/dim mismatch).
    After calling this endpoint, re-submit all jobs via POST /index/jobs.
    """
    store = get_vector_store()
    previous_count = await store.count()
    await store.reset()
    logger.warning(f"ChromaDB collection reset — previous document count: {previous_count}")

    return ReindexResponse(
        message="Vector store collection reset. Submit jobs via POST /index/jobs to repopulate.",
        previous_count=previous_count,
        task_id="n/a",
    )
