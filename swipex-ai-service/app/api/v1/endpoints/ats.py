"""ATS scoring endpoints.

Provides two scoring surfaces:
  POST /ats/score         — score a pre-parsed resume payload against a job
  POST /ats/score-upload  — upload a raw PDF/DOCX, extract inline, then score
"""

import logging
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.ats import ATSResult
from app.schemas.job import JobInput
from app.services.ai.ats_scorer import ScoringContext, get_ats_scorer

router = APIRouter(prefix="/ats", tags=["ATS Scoring"])
logger = logging.getLogger("swipex.api.ats")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ATSScoreRequest(BaseModel):
    """Payload for scoring an already-extracted resume against a job."""

    resume_text: str = Field(
        ...,
        min_length=20,
        max_length=settings.MAX_TEXT_CHARS,
        description="Plain-text resume content (PII already redacted by caller or extracted endpoint).",
    )
    resume_skills: list[str] = Field(
        default_factory=list,
        description="Skills already extracted from the resume (canonical names).",
    )
    resume_keywords: list[str] = Field(
        default_factory=list,
        description="Broad keywords extracted from the resume.",
    )
    embed_resume: bool = Field(
        default=True,
        description=(
            "If True, embed the resume text and use semantic scoring. "
            "Set False to force keyword-only heuristic mode (faster)."
        ),
    )
    job: JobInput


class ATSScoreResponse(BaseModel):
    result: ATSResult
    threshold: float = Field(
        default_factory=lambda: settings.ATS_THRESHOLD,
        description="ATS score threshold below which below_threshold is True.",
    )
    suggestions_available: bool = Field(
        default=False,
        description="True when below_threshold and the LLM suggestion endpoint can be called.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_resume(text: str) -> np.ndarray | None:
    """Embed resume text; return None gracefully when embeddings not loaded."""
    from app.services.ai.embeddings import (
        get_embedding_service,
        is_embedding_loaded,
    )

    if not is_embedding_loaded():
        return None
    try:
        svc = get_embedding_service()
        return svc.encode([text])[0]
    except Exception:
        logger.warning("Resume embedding failed; falling back to keyword-only ATS.")
        return None


def _extract_from_bytes(content: bytes, filename: str) -> tuple[str, list[str], list[str]]:
    """Run inline extraction pipeline on raw uploaded file bytes."""
    from app.services.ai.keyword_analysis import match_keywords
    from app.services.ai.preprocessing import normalize_text, redact_pii
    from app.services.ai.skill_extraction import extract_skills_from_text
    from app.services.ai.text_extraction import extract_text

    raw_text = extract_text(content, filename).text
    clean_text, _ = normalize_text(raw_text)
    redacted_text = redact_pii(clean_text)

    extracted_skills = extract_skills_from_text(redacted_text)
    skill_names = [s.name for s in extracted_skills]

    kw_result = match_keywords(
        resume_text=redacted_text,
        resume_skills=skill_names,
        required_skills=[],
        job_keywords=[],
    )
    keywords = kw_result.get("matched_keywords", [])

    return redacted_text, skill_names, keywords


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/score",
    response_model=ATSScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a pre-parsed resume against a job description",
    responses={
        200: {"description": "ATS score computed successfully"},
        422: {"description": "Invalid input"},
    },
)
async def score_parsed(
    payload: ATSScoreRequest,
    _: Annotated[None, Depends(verify_api_key)],
) -> ATSScoreResponse:
    """Compute the blended ATS match score for a pre-extracted resume + job pair.

    - If `embed_resume=True` (default) **and** the embedding service is loaded,
      returns a blended keyword + semantic score (`generated_by=blended-v1`).
    - Falls back to keyword-only (`generated_by=heuristic-v0`) when embeddings
      are unavailable or `embed_resume=False`.
    - Returns `below_threshold=True` when `match_score < ATS_THRESHOLD` (default 80).
    """
    import asyncio

    resume_vector: np.ndarray | None = None
    if payload.embed_resume:
        resume_vector = await asyncio.to_thread(_embed_resume, payload.resume_text)

    scorer = get_ats_scorer()
    ctx = ScoringContext(
        resume_skills=payload.resume_skills,
        resume_keywords=payload.resume_keywords,
        resume_text=payload.resume_text,
        job=payload.job,
        resume_vector=resume_vector,
    )
    result = await asyncio.to_thread(scorer.score, ctx)

    return ATSScoreResponse(
        result=result,
        threshold=settings.ATS_THRESHOLD,
        suggestions_available=result.below_threshold and settings.LLM_ENABLED,
    )


@router.post(
    "/score-upload",
    response_model=ATSScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a resume file and score it against a job (inline extraction)",
    responses={
        200: {"description": "ATS score computed successfully"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "No extractable text / invalid job payload"},
    },
)
async def score_upload(
    file: Annotated[UploadFile, File(description="Resume PDF or DOCX file")],
    job_id: Annotated[str, Form(description="Unique job ID")],
    job_title: Annotated[str, Form(description="Job title")],
    job_description: Annotated[str, Form(description="Full job description")] = "",
    required_skills: Annotated[str, Form(description="Comma-separated required skills")] = "",
    job_keywords: Annotated[str, Form(description="Comma-separated job keywords")] = "",
    embed_resume: Annotated[bool, Form(description="Enable semantic scoring")] = True,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> ATSScoreResponse:
    """Accept a raw resume file upload, run the full extraction pipeline inline,
    then compute the blended ATS score — all in one call.

    Use this endpoint when the caller does not want to run a separate
    `POST /extract` call before scoring.
    """
    import asyncio

    from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError

    # Validate file size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise FileTooLargeError(
            f"File size {len(content) / 1024 / 1024:.1f} MB exceeds the {settings.MAX_FILE_SIZE_MB} MB limit."
        )

    filename = file.filename or "upload.pdf"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in {"pdf", "docx"}:
        raise UnsupportedFileTypeError(f"Unsupported file extension '.{ext}'. Only PDF and DOCX are accepted.")

    # Build JobInput from form fields
    r_skills = [s.strip() for s in required_skills.split(",") if s.strip()]
    kw_list = [k.strip() for k in job_keywords.split(",") if k.strip()]
    job = JobInput(
        job_id=job_id,
        title=job_title,
        description=job_description,
        required_skills=r_skills,
        skills=r_skills,
        keywords=kw_list,
    )

    # Extract resume inline (CPU-bound — run in thread)
    resume_text, resume_skills, resume_keywords = await asyncio.to_thread(
        _extract_from_bytes, content, filename
    )

    resume_vector: np.ndarray | None = None
    if embed_resume:
        resume_vector = await asyncio.to_thread(_embed_resume, resume_text)

    scorer = get_ats_scorer()
    ctx = ScoringContext(
        resume_skills=resume_skills,
        resume_keywords=resume_keywords,
        resume_text=resume_text,
        job=job,
        resume_vector=resume_vector,
    )
    result = await asyncio.to_thread(scorer.score, ctx)

    return ATSScoreResponse(
        result=result,
        threshold=settings.ATS_THRESHOLD,
        suggestions_available=result.below_threshold and settings.LLM_ENABLED,
    )
