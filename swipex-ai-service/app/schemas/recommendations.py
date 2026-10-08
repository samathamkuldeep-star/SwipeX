"""DTO schemas for semantic recommendations and job ranking."""

from pydantic import BaseModel, Field

from app.schemas.job import JobInput


class RankedJob(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    score: float = Field(..., ge=0.0, le=100.0, description="Blended match score 0-100")
    semantic_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Semantic similarity score 0-100"
    )
    skill_score: float = Field(..., ge=0.0, le=100.0, description="Skill overlap score 0-100")
    swipe_bias_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Preference signal boost score 0-100"
    )
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class RankJobsRequest(BaseModel):
    resume_text: str | None = Field(default=None, description="Resume body text for semantic similarity")
    resume_skills: list[str] = Field(default_factory=list, description="Candidate skills list")
    swiped_job_ids: list[str] = Field(
        default_factory=list, description="IDs of previously liked/saved jobs to compute preference bias"
    )
    candidate_jobs: list[JobInput] = Field(..., min_length=1, description="Jobs to evaluate and rank")
    top_k: int = Field(default=20, ge=1, le=100, description="Maximum ranked items to return")


class RankJobsResponse(BaseModel):
    ranked_jobs: list[RankedJob] = Field(default_factory=list)


class JobFilterParams(BaseModel):
    location: str | None = Field(default=None)
    work_mode: str | None = Field(default=None)
    experience_level: str | None = Field(default=None)
    min_experience_years: float | None = Field(default=None, ge=0.0)
    company_type: str | None = Field(default=None)


class FeedJobsRequest(BaseModel):
    resume_text: str | None = Field(default=None, description="Resume text for query embedding")
    resume_skills: list[str] = Field(default_factory=list, description="Candidate skills list")
    filters: JobFilterParams | None = Field(default=None, description="Filtering criteria")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of recommendations")


class FeedJobsResponse(BaseModel):
    jobs: list[RankedJob] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
