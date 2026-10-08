"""Keyword analysis and missing keyword/skill diffing."""

import logging
import re

from app.schemas.extraction import ExtractedSkill
from app.services.ai.skill_extraction import get_nlp, get_skill_matcher_service

logger = logging.getLogger("swipex.ai.keyword_analysis")


def normalize_keyword(term: str) -> str:
    """Normalize keyword string for comparison."""
    term = term.lower().strip()
    term = re.sub(r"[^\w\s+#.-]", "", term)
    term = re.sub(r"\s+", " ", term)
    return term


def match_keywords(
    resume_text: str,
    resume_skills: list[ExtractedSkill] | list[str],
    required_skills: list[str],
    job_keywords: list[str],
) -> dict[str, list[str]]:
    """Match job required skills and keywords against resume skills and raw text."""
    matcher_service = get_skill_matcher_service()
    alias_map = matcher_service.alias_to_canonical

    # Build set of normalized resume skills (both original names & canonicals)
    resume_skill_set: set[str] = set()
    for s in resume_skills:
        if isinstance(s, ExtractedSkill):
            resume_skill_set.add(normalize_keyword(s.name))
            resume_skill_set.add(normalize_keyword(s.normalized_name))
        elif isinstance(s, str):
            resume_skill_set.add(normalize_keyword(s))
            canonical = alias_map.get(s.lower(), s)
            resume_skill_set.add(normalize_keyword(canonical))

    # Normalize raw resume text for phrase searching
    normalized_resume_text = normalize_keyword(resume_text)

    # Lemmatize text if spaCy model is available
    nlp = get_nlp()
    if hasattr(nlp, "vocab") and nlp.meta.get("name") == "core_web_sm":
        doc = nlp(resume_text.lower())
        lemmatized_resume_text = " ".join(token.lemma_ for token in doc)
    else:
        lemmatized_resume_text = normalized_resume_text

    matched_skills: list[str] = []
    missing_skills: list[str] = []

    for req_skill in required_skills:
        norm_req = normalize_keyword(req_skill)
        canonical = alias_map.get(norm_req, req_skill)
        norm_canonical = normalize_keyword(canonical)

        # Check skill matches
        is_matched = (
            norm_req in resume_skill_set
            or norm_canonical in resume_skill_set
            or norm_req in normalized_resume_text
            or norm_canonical in normalized_resume_text
            or norm_req in lemmatized_resume_text
        )

        if is_matched:
            if canonical not in matched_skills:
                matched_skills.append(canonical)
        else:
            if canonical not in missing_skills:
                missing_skills.append(canonical)

    matched_keywords: list[str] = []
    missing_keywords: list[str] = []

    for kw in job_keywords:
        norm_kw = normalize_keyword(kw)
        canonical_kw = alias_map.get(norm_kw, kw)

        is_matched = (
            norm_kw in resume_skill_set
            or normalize_keyword(canonical_kw) in resume_skill_set
            or norm_kw in normalized_resume_text
            or norm_kw in lemmatized_resume_text
        )

        if is_matched:
            if kw not in matched_keywords:
                matched_keywords.append(kw)
        else:
            if kw not in missing_keywords:
                missing_keywords.append(kw)

    return {
        "matched_skills": sorted(set(matched_skills)),
        "missing_skills": sorted(set(missing_skills)),
        "matched_keywords": sorted(set(matched_keywords)),
        "missing_keywords": sorted(set(missing_keywords)),
    }
