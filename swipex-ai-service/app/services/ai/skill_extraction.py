"""Skill extraction using spaCy PhraseMatcher, skills.json dictionary, and context confidence rules."""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

from app.schemas.extraction import ExtractedSkill
from app.services.ai.preprocessing import split_sections

logger = logging.getLogger("swipex.ai.skill_extraction")

# Single-letter/ambiguous skills requiring context
AMBIGUOUS_SKILLS = {"c", "r"}
PROGRAMMING_CONTEXT_WORDS = {"programming", "developer", "engineer", "language", "code", "development", "experience", "script"}

# Pattern for detecting years near a skill match
YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of\s+|\s+in\s+|\s+with\s+)?([A-Za-z0-9#+.-]+(?:\s+[A-Za-z0-9#+.-]+){0,2})",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy model once with fallback to spacy.blank('en')."""
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model 'en_core_web_sm'.")
        return nlp
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Falling back to spacy.blank('en').")
        return spacy.blank("en")


class SkillMatcherService:
    """Service wrapping spaCy PhraseMatcher over skills.json taxonomy."""

    def __init__(self, skills_json_path: Path | None = None):
        if skills_json_path is None:
            skills_json_path = Path(__file__).parent.parent.parent / "data" / "skills.json"

        self.skills_json_path = skills_json_path
        self.taxonomy: dict[str, dict] = {}
        self.alias_to_canonical: dict[str, str] = {}
        self._load_taxonomy()

        self.nlp = get_nlp()
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_phrase_matcher()

    def _load_taxonomy(self) -> None:
        if not self.skills_json_path.exists():
            raise FileNotFoundError(f"skills.json missing at {self.skills_json_path}")

        with open(self.skills_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data or not isinstance(data, list):
            raise ValueError("skills.json must contain a non-empty list of skill definitions.")

        seen_canonicals = set()
        for item in data:
            canonical = item["canonical"]
            if canonical in seen_canonicals:
                raise ValueError(f"Duplicate canonical skill in skills.json: {canonical}")
            seen_canonicals.add(canonical)

            aliases = item.get("aliases", [])
            category = item.get("category")

            self.taxonomy[canonical] = {
                "canonical": canonical,
                "aliases": aliases,
                "category": category,
            }

            self.alias_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical

        logger.info(f"Loaded {len(self.taxonomy)} canonical skills into taxonomy.")

    def _build_phrase_matcher(self) -> None:
        """Register canonicals and aliases with spaCy PhraseMatcher."""
        patterns_by_canonical: dict[str, list] = {}

        for canonical, info in self.taxonomy.items():
            terms = [canonical] + info["aliases"]
            patterns = [self.nlp.make_doc(t) for t in terms]
            patterns_by_canonical[canonical] = patterns

        for canonical, docs in patterns_by_canonical.items():
            self.matcher.add(canonical, docs)

    def extract_skills(self, text: str) -> list[ExtractedSkill]:
        """Extract skills from text with confidence scoring and duration matching."""
        if not text.strip():
            return []

        doc = self.nlp(text)
        matches = self.matcher(doc)

        sections = split_sections(text)
        skills_section_text = sections.get("skills", "").lower()

        # Extract duration mappings ("X years of Y", "Y: X years")
        years_map: dict[str, float] = {}
        # Forward: "5+ years of Python", "3.5 years in Kubernetes"
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+(?:of|in|with))?\s*([A-Za-z0-9#+.-]+)", text, re.IGNORECASE):
            try:
                yrs = float(match.group(1))
                target = match.group(2).lower()
                canonical = self.alias_to_canonical.get(target)
                if canonical:
                    years_map[canonical] = yrs
            except ValueError:
                pass

        # Reverse: "Python (5+ years)", "Kubernetes - 3.5 yrs"
        for match in re.finditer(r"([A-Za-z0-9#+.-]+)\s*[\(:=-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text, re.IGNORECASE):
            try:
                target = match.group(1).lower()
                canonical = self.alias_to_canonical.get(target)
                if canonical:
                    yrs = float(match.group(2))
                    years_map[canonical] = yrs
            except ValueError:
                pass

        matched_counts: dict[str, int] = {}
        first_occurrence_token: dict[str, int] = {}
        matched_tokens: dict[str, list[str]] = {}

        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Canonical name
            matched_span = doc[start:end]
            matched_str = matched_span.text

            # Boundary guard for ambiguous single-character skills ('C', 'R')
            if string_id.lower() in AMBIGUOUS_SKILLS and not matched_str.isupper():
                prev_token = doc[start - 1].text.lower() if start > 0 else ""
                next_token = doc[end].text.lower() if end < len(doc) else ""
                if prev_token not in PROGRAMMING_CONTEXT_WORDS and next_token not in PROGRAMMING_CONTEXT_WORDS:
                    continue

            matched_counts[string_id] = matched_counts.get(string_id, 0) + 1
            if string_id not in first_occurrence_token:
                first_occurrence_token[string_id] = start
            if string_id not in matched_tokens:
                matched_tokens[string_id] = []
            matched_tokens[string_id].append(matched_str)

        extracted_skills: list[ExtractedSkill] = []

        for canonical, count in matched_counts.items():
            info = self.taxonomy[canonical]
            category = info.get("category")
            orig_name = matched_tokens[canonical][0]

            # Confidence calculation
            confidence = 0.5  # Base confidence

            # Boost if found in Skills section
            if canonical.lower() in skills_section_text or orig_name.lower() in skills_section_text:
                confidence += 0.25

            # Boost per repetition
            if count > 1:
                confidence += min(0.15, (count - 1) * 0.05)

            # Boost if near a duration expression
            years = years_map.get(canonical)
            if years is not None:
                confidence += 0.10

            confidence = round(min(0.95, confidence), 2)

            extracted_skills.append(
                ExtractedSkill(
                    name=orig_name,
                    normalized_name=canonical,
                    category=category,
                    confidence=confidence,
                    years=years,
                )
            )

        # Sort by confidence desc, then canonical
        extracted_skills.sort(key=lambda s: (-s.confidence, s.normalized_name))
        return extracted_skills


_matcher_service: SkillMatcherService | None = None
_spacy_loaded: bool = False


def load_spacy_model() -> None:
    """Eagerly load spaCy and the SkillMatcherService. Call once during lifespan."""
    global _matcher_service, _spacy_loaded
    try:
        get_nlp()  # triggers lru_cache; also validates en_core_web_sm is available
        _matcher_service = SkillMatcherService()
        _spacy_loaded = True
        logger.info("spaCy model and SkillMatcherService loaded.")
    except Exception as exc:
        _spacy_loaded = False
        logger.error(f"Failed to load spaCy model: {exc}")


def is_spacy_loaded() -> bool:
    return _spacy_loaded


def get_skill_matcher_service() -> SkillMatcherService:
    global _matcher_service
    if _matcher_service is None:
        _matcher_service = SkillMatcherService()
    return _matcher_service


def extract_skills_from_text(text: str) -> list[ExtractedSkill]:
    """Convenience helper for skill extraction."""
    service = get_skill_matcher_service()
    return service.extract_skills(text)
