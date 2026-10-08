"""DTO schemas for M4 career intelligence & insights endpoints."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.job import JobInput


class SkillGapItem(BaseModel):
    skill: str = Field(..., description="Canonical skill name")
    market_frequency: int = Field(..., ge=1, description="Number of candidate jobs requiring this skill")
    importance_score: float = Field(..., ge=0.0, le=100.0, description="Relative demand score")
    category: str | None = Field(default=None)


class SkillGapRequest(BaseModel):
    user_skills: list[str] = Field(default_factory=list, description="Skills candidate currently possesses")
    target_role: str | None = Field(default=None, description="Optional target role for filtered demand")
    jobs: list[JobInput] = Field(default_factory=list, description="Benchmark job postings for skill gap analysis")


class SkillGapResponse(BaseModel):
    missing_critical_skills: list[SkillGapItem] = Field(default_factory=list)
    missing_optional_skills: list[SkillGapItem] = Field(default_factory=list)
    match_rate: float = Field(..., ge=0.0, le=100.0, description="Overall skill coverage percentage across benchmark jobs")


class ResumePerformanceRequest(BaseModel):
    job_scores: list[float] = Field(..., min_length=1, description="List of ATS match scores recorded for candidate resume")


class ResumePerformanceResponse(BaseModel):
    average_score: float = Field(..., ge=0.0, le=100.0)
    median_score: float = Field(..., ge=0.0, le=100.0)
    min_score: float = Field(..., ge=0.0, le=100.0)
    max_score: float = Field(..., ge=0.0, le=100.0)
    score_distribution: dict[str, int] = Field(
        ..., description="Histogram counts for ranges e.g. '0-40', '41-60', '61-80', '81-100'"
    )


class HiringTrendsRequest(BaseModel):
    jobs: list[JobInput] = Field(default_factory=list, description="Job postings dataset (uses vector store if empty)")


class HiringTrendsResponse(BaseModel):
    top_demanded_skills: list[dict[str, Any]] = Field(default_factory=list)
    average_years_required: float = Field(default=0.0, ge=0.0)
    work_mode_distribution: dict[str, int] = Field(default_factory=dict)
    experience_level_distribution: dict[str, int] = Field(default_factory=dict)
