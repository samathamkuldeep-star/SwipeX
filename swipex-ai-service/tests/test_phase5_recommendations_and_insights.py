"""Tests for Phase 5: Suggestions, Semantic Recommendations, and M4 Insights endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.insights import HiringTrendsRequest, ResumePerformanceRequest, SkillGapRequest
from app.schemas.job import JobInput
from app.schemas.recommendations import RankJobsRequest
from app.schemas.suggestions import SuggestionsRequest
from app.services.ai.insights import (
    analyze_skill_gap,
    compute_hiring_trends,
    compute_resume_performance,
)
from app.services.ai.recommendations import rank_candidate_jobs
from app.services.ai.suggestions import generate_resume_suggestions


@pytest.fixture
def api_client():
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


def _auth():
    return {"X-Internal-API-Key": settings.INTERNAL_API_KEY or "test-secret"}


def _make_job(job_id: str, title: str, req_skills: list[str]) -> JobInput:
    return JobInput(
        job_id=job_id,
        title=title,
        description=f"{title} position building scalable systems with {', '.join(req_skills)}.",
        required_skills=req_skills,
        skills=req_skills + ["git"],
        keywords=["backend", "engineering"],
        work_mode="remote",
        experience_level="mid",
        min_experience_years=3.0,
    )


# ---------------------------------------------------------------------------
# Suggestions Tests
# ---------------------------------------------------------------------------


def test_suggestions_heuristic_fallback():
    req = SuggestionsRequest(
        resume_text="Experienced engineer who was responsible for managing databases and worked on Python apps.",
        missing_skills=["FastAPI", "Docker", "PostgreSQL"],
        missing_keywords=["microservices", "CI/CD"],
        target_role="Backend Engineer",
    )
    res = generate_resume_suggestions(req)
    assert res.generated_by == "heuristic-rule-engine"
    assert len(res.suggestions) >= 3

    categories = {s.category for s in res.suggestions}
    assert "missing_skills" in categories
    assert "keywords" in categories
    assert "impact_phrasing" in categories


def test_suggestions_api_endpoint(api_client):
    payload = {
        "resume_text": "Experienced Python developer with 3 years of building REST APIs.",
        "missing_skills": ["Docker", "Kubernetes"],
        "target_role": "DevOps Engineer",
    }
    resp = api_client.post("/api/v1/suggestions", json=payload, headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "suggestions" in body
    assert len(body["suggestions"]) > 0
    assert body["generated_by"] in ["gemini-flash", "heuristic-rule-engine"]


# ---------------------------------------------------------------------------
# Recommendations Tests
# ---------------------------------------------------------------------------


def test_rank_candidate_jobs_logic():
    job1 = _make_job("j1", "Python Lead", ["python", "fastapi", "docker"])
    job2 = _make_job("j2", "Java Engineer", ["java", "spring", "oracle"])
    req = RankJobsRequest(
        resume_text="Senior Python Developer with strong FastAPI and Docker expertise.",
        resume_skills=["python", "fastapi", "docker"],
        candidate_jobs=[job1, job2],
    )
    res = rank_candidate_jobs(req)
    assert len(res.ranked_jobs) == 2
    assert res.ranked_jobs[0].job_id == "j1"
    assert res.ranked_jobs[0].score > res.ranked_jobs[1].score


def test_rank_jobs_api_endpoint(api_client):
    job1 = _make_job("j1", "Backend Dev", ["python", "fastapi"]).model_dump(mode="json")
    job2 = _make_job("j2", "Frontend Dev", ["react", "typescript"]).model_dump(mode="json")

    payload = {
        "resume_text": "Full stack engineer proficient in Python and FastAPI.",
        "resume_skills": ["python", "fastapi"],
        "candidate_jobs": [job1, job2],
        "top_k": 5,
    }
    resp = api_client.post("/api/v1/recommendations/rank", json=payload, headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["ranked_jobs"]) == 2
    assert body["ranked_jobs"][0]["job_id"] == "j1"


def test_feed_jobs_api_endpoint(api_client):
    payload = {
        "resume_text": "Backend developer specializing in Python APIs.",
        "resume_skills": ["python", "fastapi"],
        "top_k": 5,
    }
    resp = api_client.post("/api/v1/recommendations/jobs", json=payload, headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "jobs" in body
    assert "total_found" in body


# ---------------------------------------------------------------------------
# Insights Tests
# ---------------------------------------------------------------------------


def test_skill_gap_analysis_logic():
    job1 = _make_job("j1", "Dev 1", ["python", "fastapi", "docker"])
    job2 = _make_job("j2", "Dev 2", ["python", "fastapi", "kubernetes"])
    req = SkillGapRequest(
        user_skills=["python"],
        jobs=[job1, job2],
    )
    res = analyze_skill_gap(req)
    assert res.match_rate < 100.0
    missing_critical_names = [item.skill for item in res.missing_critical_skills]
    assert "fastapi" in missing_critical_names


def test_resume_performance_logic():
    req = ResumePerformanceRequest(job_scores=[35.0, 55.0, 75.0, 92.0, 88.0])
    res = compute_resume_performance(req)
    assert res.average_score == 69.0
    assert res.min_score == 35.0
    assert res.max_score == 92.0
    assert res.score_distribution["0-40"] == 1
    assert res.score_distribution["81-100"] == 2


def test_hiring_trends_logic():
    job1 = _make_job("j1", "Dev 1", ["python", "fastapi"])
    job2 = _make_job("j2", "Dev 2", ["python", "docker"])
    req = HiringTrendsRequest(jobs=[job1, job2])
    res = compute_hiring_trends(req)
    assert res.average_years_required == 3.0
    assert len(res.top_demanded_skills) >= 2
    top_skills_set = {item["skill"] for item in res.top_demanded_skills}
    assert "python" in top_skills_set


def test_insights_endpoints(api_client):
    job1 = _make_job("j1", "Dev 1", ["python", "fastapi"]).model_dump(mode="json")
    
    # 1. skill-gap
    sg_resp = api_client.post(
        "/api/v1/insights/skill-gap",
        json={"user_skills": ["python"], "jobs": [job1]},
        headers=_auth(),
    )
    assert sg_resp.status_code == 200
    assert "missing_critical_skills" in sg_resp.json()

    # 2. resume-performance
    rp_resp = api_client.post(
        "/api/v1/insights/resume-performance",
        json={"job_scores": [70.0, 85.0, 90.0]},
        headers=_auth(),
    )
    assert rp_resp.status_code == 200
    assert rp_resp.json()["average_score"] == 81.67

    # 3. hiring-trends
    ht_resp = api_client.post(
        "/api/v1/insights/hiring-trends",
        json={"jobs": [job1]},
        headers=_auth(),
    )
    assert ht_resp.status_code == 200
    assert "top_demanded_skills" in ht_resp.json()
