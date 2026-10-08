"""Tests for keyword matching and missing skill diffing."""

from app.services.ai.keyword_analysis import match_keywords
from app.services.ai.skill_extraction import extract_skills_from_text


def test_keyword_matching_exact_and_alias():
    resume_text = """
    Software Engineer with experience in Python, JS, Postgres, and Docker.
    Built machine learning models and CI/CD pipelines.
    """
    resume_skills = extract_skills_from_text(resume_text)

    required_skills = ["Python", "JavaScript", "PostgreSQL", "React.js", "Kubernetes"]
    job_keywords = ["machine learning", "ci/cd", "microservices", "aws"]

    res = match_keywords(
        resume_text=resume_text,
        resume_skills=resume_skills,
        required_skills=required_skills,
        job_keywords=job_keywords,
    )

    assert "Python" in res["matched_skills"]
    assert "JavaScript" in res["matched_skills"]
    assert "PostgreSQL" in res["matched_skills"]
    assert "React.js" in res["missing_skills"]
    assert "Kubernetes" in res["missing_skills"]

    assert "machine learning" in res["matched_keywords"]
    assert "ci/cd" in res["matched_keywords"]
    assert "microservices" in res["missing_keywords"]
    assert "aws" in res["missing_keywords"]
