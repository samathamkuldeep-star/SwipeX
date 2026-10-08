"""Extraction API endpoints for file parsing, skill extraction, and resume processing."""

import asyncio

from fastapi import APIRouter, File, UploadFile, status

from app.core.config import settings
from app.schemas.extraction import (
    ExtractedDocument,
    ResumeExtractionResponse,
    SkillExtractionRequest,
    SkillExtractionResponse,
)
from app.services.ai.experience_extraction import extract_experience
from app.services.ai.preprocessing import normalize_text, split_sections
from app.services.ai.skill_extraction import extract_skills_from_text
from app.services.ai.text_extraction import extract_text

router = APIRouter()
_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AI_TASKS)


@router.post(
    "/extract/text",
    response_model=ExtractedDocument,
    status_code=status.HTTP_200_OK,
    summary="Extract raw text from PDF or DOCX file",
)
async def extract_text_endpoint(
    file: UploadFile = File(..., description="PDF or DOCX resume document"),
) -> ExtractedDocument:
    """Extract and normalize text from an uploaded resume document."""
    file_bytes = await file.read()

    async with _semaphore:
        doc = await asyncio.to_thread(
            extract_text,
            data=file_bytes,
            content_type=file.content_type,
            filename=file.filename,
        )
        normalized_text, norm_warnings = normalize_text(doc.text)
        doc.text = normalized_text
        doc.warnings.extend(norm_warnings)
        return doc


@router.post(
    "/extract/skills",
    response_model=SkillExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract skills, experience, and total years from text",
)
async def extract_skills_endpoint(
    payload: SkillExtractionRequest,
) -> SkillExtractionResponse:
    """Extract structured skills, experience items, and total duration from resume text."""
    async with _semaphore:
        normalized_text, _ = await asyncio.to_thread(normalize_text, payload.text)

        skills = await asyncio.to_thread(extract_skills_from_text, normalized_text)
        experience, total_years = await asyncio.to_thread(extract_experience, normalized_text)

        return SkillExtractionResponse(
            skills=skills,
            experience=experience,
            total_years_experience=total_years,
        )


@router.post(
    "/extract/resume",
    response_model=ResumeExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-end file text parsing, section splitting, and skill extraction",
)
async def extract_resume_endpoint(
    file: UploadFile = File(..., description="PDF or DOCX resume document"),
) -> ResumeExtractionResponse:
    """Parse document file, extract text, split sections, and perform skill/experience extraction."""
    file_bytes = await file.read()

    async with _semaphore:
        doc = await asyncio.to_thread(
            extract_text,
            data=file_bytes,
            content_type=file.content_type,
            filename=file.filename,
        )

        normalized_text, norm_warnings = normalize_text(doc.text)
        doc.text = normalized_text
        doc.warnings.extend(norm_warnings)

        sections = await asyncio.to_thread(split_sections, normalized_text)
        skills = await asyncio.to_thread(extract_skills_from_text, normalized_text)
        experience, total_years = await asyncio.to_thread(extract_experience, normalized_text)

        return ResumeExtractionResponse(
            document=doc,
            skills=skills,
            experience=experience,
            total_years_experience=total_years,
            sections=sections,
        )
