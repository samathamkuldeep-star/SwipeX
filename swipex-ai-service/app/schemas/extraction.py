"""Extraction DTO schemas for text, skills, experience, and sections."""

from pydantic import BaseModel, Field


class ExtractedSkill(BaseModel):
    name: str = Field(..., description="Original extracted skill string")
    normalized_name: str = Field(..., description="Canonical skill name")
    category: str | None = Field(default=None, description="Skill classification category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score 0.0-1.0")
    years: float | None = Field(default=None, ge=0.0, description="Estimated years of experience with skill")


class ExperienceItem(BaseModel):
    title: str | None = Field(default=None, description="Role title")
    company: str | None = Field(default=None, description="Company name")
    start: str | None = Field(default=None, description="Start date string (e.g. Jan 2021)")
    end: str | None = Field(default=None, description="End date string (e.g. Present or Dec 2022)")
    is_current: bool = Field(default=False, description="Whether position is ongoing")
    years: float | None = Field(default=None, ge=0.0, description="Duration in years")
    snippet: str = Field(default="", description="Snippet or bullet text")


class ExtractedDocument(BaseModel):
    text: str = Field(..., description="Extracted clean text")
    pages: int = Field(default=1, description="Number of pages processed")
    method: str = Field(..., description="Extraction tool used (e.g. pdfplumber, pypdf, python-docx)")
    warnings: list[str] = Field(default_factory=list, description="Parsing warnings")


class SkillExtractionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Resume or text to extract skills/experience from")


class SkillExtractionResponse(BaseModel):
    skills: list[ExtractedSkill] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    total_years_experience: float = Field(default=0.0, ge=0.0)


class ResumeExtractionResponse(BaseModel):
    document: ExtractedDocument
    skills: list[ExtractedSkill] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    total_years_experience: float = Field(default=0.0, ge=0.0)
    sections: dict[str, str] = Field(default_factory=dict, description="Parsed section text mapping")
