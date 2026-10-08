"""JobInput shared DTO — used by ATS, recommender, vector store, and insights endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class JobInput(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    description: str = Field(default="", description="Full job description text")
    required_skills: list[str] = Field(default_factory=list, description="Skills required for the role")
    keywords: list[str] = Field(default_factory=list, description="Additional keywords from the job")
    skills: list[str] = Field(default_factory=list, description="All skills listed (union of required + nice-to-have)")
    location: str | None = Field(default=None, description="Job location string")
    work_mode: str | None = Field(default=None, description="remote|hybrid|onsite")
    experience_level: str | None = Field(default=None, description="entry|junior|mid|senior|lead")
    min_experience_years: float | None = Field(default=None, ge=0.0)
    company_type: str | None = Field(default=None, description="startup|mid-size|enterprise|newly_founded")
    competition_level: str | None = Field(default=None, description="low|medium|high")
    posted_at: datetime | None = Field(default=None)
    applicant_count: int | None = Field(default=None, ge=0)
