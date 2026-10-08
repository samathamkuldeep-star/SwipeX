"""ATS scoring DTOs."""

from typing import Literal

from pydantic import BaseModel, Field


class ATSResult(BaseModel):
    match_score: float = Field(..., ge=0.0, le=100.0, description="Blended ATS match score 0–100")
    keyword_score: float = Field(..., ge=0.0, le=100.0, description="Keyword/skill overlap score")
    semantic_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Semantic similarity score (None when embeddings unavailable)"
    )
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    below_threshold: bool = Field(..., description="True when match_score < ATS_THRESHOLD")
    generated_by: Literal["heuristic-v0", "blended-v1"] = Field(
        ..., description="Scoring mode used"
    )
