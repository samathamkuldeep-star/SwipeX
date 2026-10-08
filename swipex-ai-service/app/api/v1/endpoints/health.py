"""Health check and readiness endpoints."""

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.common import HealthStatus, ReadinessStatus

router = APIRouter()


@router.get("/health", response_model=HealthStatus, summary="Liveness check")
async def health_check() -> HealthStatus:
    """Basic liveness check."""
    return HealthStatus(
        status="ok",
        environment=settings.ENVIRONMENT,
    )


@router.get("/ready", response_model=ReadinessStatus, summary="Readiness check")
async def readiness_check() -> ReadinessStatus:
    """Readiness probe — reflects real status of all loaded AI components."""
    from app.services.ai.embeddings import is_embedding_loaded
    from app.services.ai.skill_extraction import is_spacy_loaded
    from app.services.ai.vector_store import get_vector_store

    embedding_model_loaded = is_embedding_loaded()
    spacy_loaded = is_spacy_loaded()
    chroma_status = get_vector_store().status

    if not embedding_model_loaded or not spacy_loaded:
        overall_status = "degraded"
    elif chroma_status == "reindex_required":
        overall_status = "reindex_required"
    elif chroma_status == "degraded":
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return ReadinessStatus(
        status=overall_status,
        embedding_model_loaded=embedding_model_loaded,
        spacy_loaded=spacy_loaded,
        chroma=chroma_status,
        llm_enabled=settings.LLM_ENABLED,
    )
