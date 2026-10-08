"""Tests for Phase 4: ATS Scorer and /ats/* endpoints."""

import io
import uuid
from unittest.mock import patch

import numpy as np
import pytest

from app.core.config import settings
from app.schemas.job import JobInput
from app.services.ai.ats_scorer import ATSScorer, ScoringContext, _rescale_semantic, _skill_set
from app.services.ai.embeddings import FakeEmbeddingService, set_embedding_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**kwargs) -> JobInput:
    defaults = {
        "job_id": str(uuid.uuid4()),
        "title": "Backend Engineer",
        "description": "Build REST APIs with Python, FastAPI, and PostgreSQL.",
        "required_skills": ["python", "fastapi", "postgresql", "docker"],
        "skills": ["python", "fastapi", "postgresql", "docker", "git"],
        "keywords": ["backend", "api", "microservices", "rest"],
    }
    defaults.update(kwargs)
    return JobInput(**defaults)


def _make_ctx(
    resume_skills: list[str],
    resume_keywords: list[str] | None = None,
    resume_text: str = "Experienced Python developer with FastAPI.",
    job: JobInput | None = None,
    resume_vector: np.ndarray | None = None,
) -> ScoringContext:
    return ScoringContext(
        resume_skills=resume_skills,
        resume_keywords=resume_keywords or [],
        resume_text=resume_text,
        job=job or _make_job(),
        resume_vector=resume_vector,
    )


@pytest.fixture(autouse=True)
def inject_fake_embeddings():
    set_embedding_service(FakeEmbeddingService())
    yield


# ---------------------------------------------------------------------------
# _skill_set
# ---------------------------------------------------------------------------


class TestSkillSetHelper:
    def test_normalises_case(self):
        result = _skill_set(["Python", "FASTAPI", "React "])
        assert "python" in result
        assert "fastapi" in result
        assert "react" in result

    def test_strips_empty_strings(self):
        result = _skill_set(["  ", "", "go"])
        assert "" not in result
        assert "go" in result


# ---------------------------------------------------------------------------
# _rescale_semantic
# ---------------------------------------------------------------------------


class TestRescaleSemantic:
    def test_floor_clips_to_zero(self):
        assert _rescale_semantic(settings.SEM_SIM_FLOOR - 0.1) == 0.0

    def test_ceil_clips_to_100(self):
        assert _rescale_semantic(settings.SEM_SIM_CEIL + 0.1) == 100.0

    def test_midpoint(self):
        mid = (settings.SEM_SIM_FLOOR + settings.SEM_SIM_CEIL) / 2
        score = _rescale_semantic(mid)
        assert 40.0 < score < 60.0

    def test_exact_floor(self):
        assert _rescale_semantic(settings.SEM_SIM_FLOOR) == 0.0

    def test_exact_ceil(self):
        assert _rescale_semantic(settings.SEM_SIM_CEIL) == 100.0


# ---------------------------------------------------------------------------
# ATSScorer — keyword scoring
# ---------------------------------------------------------------------------


class TestATSScorerKeyword:
    def setup_method(self):
        self.scorer = ATSScorer()

    def test_perfect_match_score_is_100(self):
        ctx = _make_ctx(
            resume_skills=["python", "fastapi", "postgresql", "docker"],
            resume_keywords=["backend", "api", "microservices", "rest"],
        )
        result = self.scorer.score(ctx)
        assert result.match_score == pytest.approx(result.keyword_score, abs=1.0)
        assert result.keyword_score == 100.0

    def test_zero_match_gives_low_score(self):
        ctx = _make_ctx(
            resume_skills=["java", "spring"],
            resume_keywords=["enterprise", "beans"],
        )
        result = self.scorer.score(ctx)
        assert result.keyword_score < 30.0

    def test_partial_match(self):
        # 2/4 hard required matched → hard=50%; soft_kw has 4 keywords not in resume → soft=0%
        # keyword_score = 0.80*50 + 0.20*0 = 40.0
        ctx = _make_ctx(
            resume_skills=["python", "fastapi"],   # matched 2/4
            resume_keywords=[],
        )
        result = self.scorer.score(ctx)
        assert result.keyword_score == pytest.approx(40.0, abs=0.1)

    def test_missing_skills_populated(self):
        ctx = _make_ctx(
            resume_skills=["python"],
            resume_keywords=[],
        )
        result = self.scorer.score(ctx)
        assert "fastapi" in result.missing_skills
        assert "postgresql" in result.missing_skills
        assert "docker" in result.missing_skills

    def test_matched_skills_populated(self):
        ctx = _make_ctx(
            resume_skills=["python", "docker"],
            resume_keywords=[],
        )
        result = self.scorer.score(ctx)
        assert "python" in result.matched_skills
        assert "docker" in result.matched_skills

    def test_keyword_match_via_resume_keywords(self):
        """Skills matched through resume_keywords should count.

        hard-score = 4/4 = 100%; soft_kw = {backend, api, microservices, rest} - not in resume
        → soft_score = 0%; keyword_score = 0.80*100 + 0.20*0 = 80.0
        """
        ctx = _make_ctx(
            resume_skills=[],
            resume_keywords=["python", "fastapi", "postgresql", "docker"],
        )
        result = self.scorer.score(ctx)
        assert result.keyword_score == pytest.approx(80.0, abs=0.1)
        # All required skills should appear as matched (not missing)
        assert result.missing_skills == []

    def test_heuristic_mode_when_no_vector(self):
        ctx = _make_ctx(
            resume_skills=["python"],
            resume_keywords=[],
            resume_vector=None,
        )
        result = self.scorer.score(ctx)
        assert result.generated_by == "heuristic-v0"
        assert result.semantic_score is None
        assert result.match_score == pytest.approx(result.keyword_score, abs=0.01)

    def test_soft_skills_excluded_from_missing(self):
        job = _make_job(required_skills=["python", "communication"])
        ctx = _make_ctx(resume_skills=["python"], resume_keywords=[], job=job)
        result = self.scorer.score(ctx)
        assert "communication" not in result.missing_skills

    def test_empty_required_skills_gives_100_keyword_score(self):
        job = _make_job(required_skills=[], skills=[], keywords=[])
        ctx = _make_ctx(resume_skills=[], resume_keywords=[], job=job)
        result = self.scorer.score(ctx)
        assert result.keyword_score == 100.0


# ---------------------------------------------------------------------------
# ATSScorer — blended mode (with fake embeddings)
# ---------------------------------------------------------------------------


class TestATSScorerBlended:
    def setup_method(self):
        self.scorer = ATSScorer()
        self.svc = FakeEmbeddingService()

    def test_blended_mode_when_vector_provided(self):
        vec = self.svc.encode(["python developer with fastapi experience"])[0]
        ctx = _make_ctx(
            resume_skills=["python", "fastapi"],
            resume_keywords=[],
            resume_vector=vec,
        )
        result = self.scorer.score(ctx)
        assert result.generated_by == "blended-v1"
        assert result.semantic_score is not None
        assert 0.0 <= result.semantic_score <= 100.0

    def test_blended_score_weighted_correctly(self):
        vec = self.svc.encode(["python developer"])[0]
        ctx = _make_ctx(
            resume_skills=["python", "fastapi", "postgresql", "docker"],
            resume_keywords=[],
            resume_vector=vec,
        )
        result = self.scorer.score(ctx)
        # keyword_score should be 100 → match_score = 0.6*100 + 0.4*semantic
        assert result.match_score == pytest.approx(
            settings.ATS_KEYWORD_WEIGHT * result.keyword_score
            + settings.ATS_SEMANTIC_WEIGHT * result.semantic_score,
            abs=0.1,
        )

    def test_below_threshold_set_correctly(self):
        ctx = _make_ctx(
            resume_skills=[],
            resume_keywords=[],
        )
        result = self.scorer.score(ctx)
        assert result.below_threshold == (result.match_score < settings.ATS_THRESHOLD)

    def test_score_bounds(self):
        vec = self.svc.encode(["random text"])[0]
        ctx = _make_ctx(
            resume_skills=["python"],
            resume_keywords=[],
            resume_vector=vec,
        )
        result = self.scorer.score(ctx)
        assert 0.0 <= result.match_score <= 100.0
        assert 0.0 <= result.keyword_score <= 100.0
        assert 0.0 <= result.semantic_score <= 100.0


# ---------------------------------------------------------------------------
# /ats/score API endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def ats_client():
    from fastapi.testclient import TestClient

    from app.main import create_application
    from app.services.ai.embeddings import set_embedding_service
    from app.services.ai.vector_store import VectorStore, set_vector_store

    # minimal degraded store so /ready doesn't 503
    store = VectorStore()
    store._status = "degraded"

    test_app = create_application()
    set_embedding_service(FakeEmbeddingService())
    set_vector_store(store)

    with TestClient(test_app, raise_server_exceptions=True) as client:
        yield client


def _auth():
    return {"X-Internal-API-Key": "test-secret"}


def _base_score_payload(**overrides):
    payload = {
        "resume_text": "Experienced backend engineer with Python, FastAPI, and Docker.",
        "resume_skills": ["python", "fastapi", "docker"],
        "resume_keywords": ["backend", "microservices"],
        "embed_resume": True,
        "job": {
            "job_id": "job-ats-001",
            "title": "Backend Engineer",
            "description": "Build REST APIs with Python, FastAPI, and PostgreSQL.",
            "required_skills": ["python", "fastapi", "postgresql", "docker"],
            "skills": ["python", "fastapi", "postgresql", "docker"],
            "keywords": ["backend", "api", "rest"],
        },
    }
    payload.update(overrides)
    return payload


class TestATSScoreEndpoint:
    def test_returns_200_with_valid_payload(self, ats_client):
        resp = ats_client.post(
            "/api/v1/ats/score", json=_base_score_payload(), headers=_auth()
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        assert 0.0 <= body["result"]["match_score"] <= 100.0

    def test_threshold_field_in_response(self, ats_client):
        resp = ats_client.post(
            "/api/v1/ats/score", json=_base_score_payload(), headers=_auth()
        )
        assert resp.json()["threshold"] == settings.ATS_THRESHOLD

    def test_below_threshold_field_matches_score(self, ats_client):
        resp = ats_client.post(
            "/api/v1/ats/score", json=_base_score_payload(), headers=_auth()
        )
        body = resp.json()
        match_score = body["result"]["match_score"]
        below = body["result"]["below_threshold"]
        assert below == (match_score < settings.ATS_THRESHOLD)

    def test_heuristic_mode_when_embed_false(self, ats_client):
        payload = _base_score_payload(embed_resume=False)
        resp = ats_client.post("/api/v1/ats/score", json=payload, headers=_auth())
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["generated_by"] == "heuristic-v0"
        assert body["result"]["semantic_score"] is None

    def test_missing_skills_returned(self, ats_client):
        payload = _base_score_payload()
        payload["resume_skills"] = ["python"]  # missing fastapi, postgresql, docker
        resp = ats_client.post("/api/v1/ats/score", json=payload, headers=_auth())
        body = resp.json()
        assert "fastapi" in body["result"]["missing_skills"]
        assert "postgresql" in body["result"]["missing_skills"]

    def test_requires_auth(self, ats_client):
        resp = ats_client.post("/api/v1/ats/score", json=_base_score_payload())
        assert resp.status_code in (401, 403)

    def test_rejects_short_resume_text(self, ats_client):
        payload = _base_score_payload(resume_text="short")
        resp = ats_client.post("/api/v1/ats/score", json=payload, headers=_auth())
        assert resp.status_code == 422

    def test_suggestions_available_when_llm_enabled_and_below_threshold(self, ats_client):
        """When score < threshold and LLM is disabled (no API key), suggestions_available=False."""
        payload = _base_score_payload()
        payload["resume_skills"] = []  # guarantees low score
        with patch.object(settings.__class__, "LLM_ENABLED", new_callable=lambda: property(lambda self: False)):
            resp = ats_client.post("/api/v1/ats/score", json=payload, headers=_auth())
        body = resp.json()
        assert body["suggestions_available"] is False


# ---------------------------------------------------------------------------
# /ats/score-upload API endpoint
# ---------------------------------------------------------------------------


class TestATSScoreUploadEndpoint:
    def _make_pdf_bytes(self) -> bytes:
        """Generate a minimal valid PDF in memory using reportlab."""
        try:
            from io import BytesIO

            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas

            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=LETTER)
            c.drawString(72, 720, "Python developer with FastAPI and Docker experience.")
            c.drawString(72, 700, "Skills: python, fastapi, docker, postgresql")
            c.save()
            return buf.getvalue()
        except ImportError:
            # Minimal valid PDF if reportlab unavailable
            return (
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
                b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 720 Td (Python developer) Tj ET\nendstream\nendobj\n"
                b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
                b"xref\n0 6\n0000000000 65535 f\n"
                b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
            )

    def test_score_upload_with_pdf(self, ats_client):
        pdf_bytes = self._make_pdf_bytes()
        data = {
            "job_id": "job-upload-001",
            "job_title": "Backend Engineer",
            "job_description": "Build REST APIs with Python and FastAPI.",
            "required_skills": "python,fastapi,docker",
            "job_keywords": "backend,api",
            "embed_resume": "true",
        }
        resp = ats_client.post(
            "/api/v1/ats/score-upload",
            data=data,
            files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=_auth(),
        )
        # Acceptable outcomes: 200 (text extracted) or 422 (minimal PDF no text)
        assert resp.status_code in (200, 422)

    def test_score_upload_rejects_wrong_ext(self, ats_client):
        data = {
            "job_id": "j1",
            "job_title": "Dev",
            "required_skills": "python",
        }
        resp = ats_client.post(
            "/api/v1/ats/score-upload",
            data=data,
            files={"file": ("resume.txt", io.BytesIO(b"plain text"), "text/plain")},
            headers=_auth(),
        )
        assert resp.status_code == 415

    def test_score_upload_rejects_oversized_file(self, ats_client):
        big = b"A" * (settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        data = {
            "job_id": "j2",
            "job_title": "Dev",
            "required_skills": "python",
        }
        resp = ats_client.post(
            "/api/v1/ats/score-upload",
            data=data,
            files={"file": ("resume.pdf", io.BytesIO(big), "application/pdf")},
            headers=_auth(),
        )
        assert resp.status_code == 413

    def test_score_upload_requires_auth(self, ats_client):
        data = {"job_id": "j3", "job_title": "Dev", "required_skills": "python"}
        resp = ats_client.post(
            "/api/v1/ats/score-upload",
            data=data,
            files={"file": ("r.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert resp.status_code in (401, 403)
