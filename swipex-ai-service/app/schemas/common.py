"""Common DTOs and API envelope schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error summary")
    details: Any | None = Field(default=None, description="Optional structured error details")
    request_id: str = Field(..., description="Unique request tracing ID")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthStatus(BaseModel):
    status: str = "ok"
    environment: str
    version: str = "1.0.0"


class ReadinessStatus(BaseModel):
    status: str  # ok, degraded, reindex_required
    embedding_model_loaded: bool
    spacy_loaded: bool
    chroma: str  # ok, degraded, reindex_required
    llm_enabled: bool
