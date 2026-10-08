"""DTO schemas for resume improvement suggestions."""

from typing import Literal

from pydantic import BaseModel, Field


class ResumeSuggestion(BaseModel):
    category: Literal["missing_skills", "formatting", "impact_phrasing", "keywords", "general"] = Field(
        ..., description="Suggestion focus category"
    )
    suggestion: str = Field(..., description="Actionable recommendation text")
    impact: Literal["high", "medium", "low"] = Field(..., description="Expected impact on ATS / recruiter score")
    example: str | None = Field(default=None, description="Concrete example bullet or phrasing")


class SuggestionsRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Resume text to analyze")
    job_description: str | None = Field(default=None, description="Target job description text")
    missing_skills: list[str] = Field(default_factory=list, description="Skills missing from ATS check")
    missing_keywords: list[str] = Field(default_factory=list, description="Keywords missing from ATS check")
    target_role: str | None = Field(default=None, description="Target job title or role name")


class SuggestionsResponse(BaseModel):
    suggestions: list[ResumeSuggestion] = Field(default_factory=list)
    generated_by: Literal["gemini-flash", "heuristic-rule-engine"] = Field(
        ..., description="Generation engine used"
    )
