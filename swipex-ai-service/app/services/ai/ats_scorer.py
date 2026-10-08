"""Blended ATS scoring engine: keyword/skill overlap + optional semantic similarity.

Scoring algorithm (v1):
  keyword_score  = (matched_skills / required_skills) × 100  (Jaccard-style, weighted by importance)
  semantic_score = cosine_sim(resume_vec, job_vec) rescaled to 0–100 via SEM_SIM_FLOOR/CEIL clamp
  match_score    = ATS_KEYWORD_WEIGHT × keyword_score + ATS_SEMANTIC_WEIGHT × semantic_score

When embeddings are unavailable the scorer falls back to heuristic-v0 (keyword only, semantic_score=None).
"""

import logging
import re
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.schemas.ats import ATSResult
from app.schemas.job import JobInput

logger = logging.getLogger("swipex.ai.ats_scorer")

# Required-skill importance multipliers (based on taxonomy / seniority signals)
_SENIOR_MARKERS = re.compile(
    r"\b(senior|lead|principal|staff|architect|head of|director)\b",
    re.IGNORECASE,
)

# Skills that are nearly universal and should not inflate the penalty for missing them
_SOFT_SKILLS = frozenset(
    {
        "communication",
        "teamwork",
        "collaboration",
        "leadership",
        "problem solving",
        "critical thinking",
        "time management",
        "attention to detail",
        "adaptability",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(skill: str) -> str:
    """Lower-case and strip extra whitespace from a skill string."""
    return skill.strip().lower()


def _skill_set(skills: list[str]) -> set[str]:
    return {_normalize(s) for s in skills if s.strip()}


def _rescale_semantic(cosine_sim: float) -> float:
    """Map cosine similarity [FLOOR, CEIL] → [0, 100], clamp outside range."""
    floor = settings.SEM_SIM_FLOOR
    ceil_ = settings.SEM_SIM_CEIL
    clipped = max(floor, min(ceil_, cosine_sim))
    return round((clipped - floor) / (ceil_ - floor) * 100.0, 2)


# ---------------------------------------------------------------------------
# Public scoring dataclass (internal to scorer, not the API schema)
# ---------------------------------------------------------------------------


@dataclass
class ScoringContext:
    """All inputs needed for one ATS scoring call."""

    resume_skills: list[str]
    resume_keywords: list[str]
    resume_text: str
    job: JobInput
    resume_vector: np.ndarray | None = None  # pre-computed by caller if available


# ---------------------------------------------------------------------------
# ATS Scorer
# ---------------------------------------------------------------------------


class ATSScorer:
    """Deterministic ATS scoring with optional semantic augmentation."""

    # ---------------------------------------------------------
    # Keyword / skill scoring
    # ---------------------------------------------------------

    def _compute_keyword_score(
        self,
        resume_skills: set[str],
        resume_keywords: set[str],
        required_skills: set[str],
        job_keywords: set[str],
    ) -> tuple[float, list[str], list[str]]:
        """Return (keyword_score 0–100, matched_skills, missing_skills).

        Strategy:
        1. Hard-skill overlap: required_skills vs resume (skills + keywords)
        2. Soft-skill / keyword overlap: job_keywords vs resume keywords
        3. Weighted blend: hard 80% + soft 20% (each 0–100)
        """
        resume_combined = resume_skills | resume_keywords

        # Hard-skill overlap (required by job)
        hard_required = required_skills - _SOFT_SKILLS
        if hard_required:
            hard_matched = hard_required & resume_combined
            hard_score = len(hard_matched) / len(hard_required) * 100.0
        else:
            hard_matched = set()
            hard_score = 100.0  # no hard requirements → perfect on hard axis

        # Soft / keyword overlap
        soft_kw = job_keywords - hard_required - _SOFT_SKILLS
        if soft_kw:
            soft_matched = soft_kw & resume_combined
            soft_score = len(soft_matched) / len(soft_kw) * 100.0
        else:
            soft_score = 100.0

        keyword_score = round(0.80 * hard_score + 0.20 * soft_score, 2)

        # All matched and missing skills (from the combined required pool)
        all_required = required_skills
        matched_skills = sorted(all_required & resume_combined)
        missing_skills = sorted(all_required - resume_combined - _SOFT_SKILLS)
        missing_keywords = sorted(job_keywords - resume_combined - _SOFT_SKILLS - all_required)

        return keyword_score, matched_skills, missing_skills, missing_keywords  # type: ignore[return-value]

    # ---------------------------------------------------------
    # Semantic scoring
    # ---------------------------------------------------------

    def _compute_semantic_score(
        self,
        resume_vector: np.ndarray,
        job: JobInput,
    ) -> float | None:
        """Return semantic similarity rescaled to 0–100, or None on failure."""
        from app.services.ai.embeddings import (
            get_embedding_service,
            is_embedding_loaded,
        )

        if not is_embedding_loaded():
            return None

        try:
            svc = get_embedding_service()
            job_text = f"{job.title}. {job.description} Skills: {', '.join(job.required_skills + job.skills)}"
            job_vec = svc.encode([job_text])[0]

            # Both vectors should already be L2-normalised → cosine = dot product
            cosine = float(np.dot(resume_vector, job_vec))
            return _rescale_semantic(cosine)
        except Exception:
            logger.warning("Semantic scoring failed; falling back to keyword-only.", exc_info=False)
            return None

    # ---------------------------------------------------------
    # Public score method
    # ---------------------------------------------------------

    def score(self, ctx: ScoringContext) -> ATSResult:
        """Compute the blended ATS score for one resume ↔ job pair."""
        resume_skills = _skill_set(ctx.resume_skills)
        resume_keywords = _skill_set(ctx.resume_keywords)
        required_skills = _skill_set(ctx.job.required_skills)
        job_keywords = _skill_set(ctx.job.keywords)

        keyword_score, matched_skills, missing_skills, missing_keywords = self._compute_keyword_score(
            resume_skills, resume_keywords, required_skills, job_keywords
        )

        # Semantic scoring (only when a resume embedding is provided)
        semantic_score: float | None = None
        if ctx.resume_vector is not None and ctx.resume_vector.size > 0:
            semantic_score = self._compute_semantic_score(ctx.resume_vector, ctx.job)

        # Blend
        if semantic_score is not None:
            match_score = round(
                settings.ATS_KEYWORD_WEIGHT * keyword_score
                + settings.ATS_SEMANTIC_WEIGHT * semantic_score,
                2,
            )
            generated_by = "blended-v1"
        else:
            match_score = round(keyword_score, 2)
            generated_by = "heuristic-v0"

        below_threshold = match_score < settings.ATS_THRESHOLD

        logger.debug(
            f"ATS score — job={ctx.job.job_id!r} "
            f"keyword={keyword_score} semantic={semantic_score} "
            f"blended={match_score} below_threshold={below_threshold}"
        )

        return ATSResult(
            match_score=match_score,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            missing_keywords=missing_keywords,
            below_threshold=below_threshold,
            generated_by=generated_by,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ats_scorer: ATSScorer | None = None


def get_ats_scorer() -> ATSScorer:
    global _ats_scorer
    if _ats_scorer is None:
        _ats_scorer = ATSScorer()
    return _ats_scorer
