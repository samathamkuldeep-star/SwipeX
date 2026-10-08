"""API endpoint for AI/LLM resume optimization suggestions."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.security import verify_api_key
from app.schemas.suggestions import SuggestionsRequest, SuggestionsResponse
from app.services.ai.suggestions import generate_resume_suggestions

router = APIRouter(prefix="", dependencies=[Depends(verify_api_key)])


@router.post(
    "/suggestions",
    response_model=SuggestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI resume improvement suggestions",
    responses={
        200: {"description": "Suggestions generated successfully"},
        400: {"description": "Invalid payload"},
        401: {"description": "Unauthorized — missing or invalid API key"},
    },
)
async def get_resume_suggestions(
    payload: SuggestionsRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> SuggestionsResponse:
    """Analyze resume text against job requirements and generate recommendations

    (Gemini-powered when configured, heuristic fallback when LLM is unavailable).
    """
    return await asyncio.to_thread(generate_resume_suggestions, payload)
