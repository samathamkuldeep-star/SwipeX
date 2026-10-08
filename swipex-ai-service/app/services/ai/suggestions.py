"""AI/LLM & Heuristic Resume Optimization Suggestions Service."""

import logging
import re

from app.core.config import settings
from app.schemas.suggestions import ResumeSuggestion, SuggestionsRequest, SuggestionsResponse

logger = logging.getLogger(__name__)


def generate_resume_suggestions(req: SuggestionsRequest) -> SuggestionsResponse:
    """Generate resume optimization suggestions via Gemini (if key configured) or fallback rule engine."""
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip():
        try:
            suggestions = _generate_with_gemini(req)
            if suggestions:
                return SuggestionsResponse(suggestions=suggestions, generated_by="gemini-flash")
        except Exception as err:
            logger.warning("Gemini suggestion generation failed, falling back to rule engine: %s", err)

    rule_suggestions = _generate_heuristic_suggestions(req)
    return SuggestionsResponse(suggestions=rule_suggestions, generated_by="heuristic-rule-engine")


def _generate_with_gemini(req: SuggestionsRequest) -> list[ResumeSuggestion] | None:
    """Attempt suggestion generation via Google Gemini API."""
    
    # Try importing google-genai or google.generativeai
    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        prompt = _build_gemini_prompt(req)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        raw_text = response.text or ""
        return _parse_gemini_response(raw_text)
    except Exception as e1:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=settings.GEMINI_API_KEY)
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            prompt = _build_gemini_prompt(req)
            res = model.generate_content(prompt)
            return _parse_gemini_response(res.text or "")
        except Exception as e2:
            logger.debug("Gemini SDK error: %s / %s", e1, e2)
            raise e1


def _build_gemini_prompt(req: SuggestionsRequest) -> str:
    missing_skills_str = ", ".join(req.missing_skills) if req.missing_skills else "None noted"
    missing_kw_str = ", ".join(req.missing_keywords) if req.missing_keywords else "None noted"
    target_role = req.target_role or "Target Role"

    return f"""You are an expert ATS resume optimizer and career coach.
Analyze the candidate's resume for the role: '{target_role}'.

Target Job Description:
{req.job_description or "N/A"}

Identified Missing Skills: {missing_skills_str}
Identified Missing Industry Keywords: {missing_kw_str}

Candidate Resume Text:
{req.resume_text[:3000]}

Provide 3 to 6 actionable recommendations as a JSON array.
Each recommendation object MUST have:
- category: one of ["missing_skills", "formatting", "impact_phrasing", "keywords", "general"]
- suggestion: concise actionable instruction
- impact: "high" | "medium" | "low"
- example: optional concrete example bullet string

Return ONLY valid JSON array with no markdown codeblocks or extra commentary.
"""


def _parse_gemini_response(text: str) -> list[ResumeSuggestion] | None:
    import json

    clean = text.strip()
    clean = clean.removeprefix("```json")
    clean = clean.removeprefix("```")
    clean = clean.removesuffix("```")
    clean = clean.strip()

    try:
        data = json.loads(clean)
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, dict) and "category" in item and "suggestion" in item:
                    cat = item.get("category", "general")
                    if cat not in ["missing_skills", "formatting", "impact_phrasing", "keywords", "general"]:
                        cat = "general"
                    imp = item.get("impact", "medium")
                    if imp not in ["high", "medium", "low"]:
                        imp = "medium"
                    results.append(
                        ResumeSuggestion(
                            category=cat,
                            suggestion=str(item["suggestion"]),
                            impact=imp,
                            example=item.get("example"),
                        )
                    )
            return results
    except Exception as err:
        logger.warning("Failed to parse Gemini JSON output: %s", err)
    return None


def _generate_heuristic_suggestions(req: SuggestionsRequest) -> list[ResumeSuggestion]:
    """Generate deterministic, high-quality heuristic rules."""
    suggestions: list[ResumeSuggestion] = []
    text_lower = req.resume_text.lower()

    # Rule 1: Missing Skills
    if req.missing_skills:
        top_missing = req.missing_skills[:5]
        skills_str = ", ".join(top_missing)
        suggestions.append(
            ResumeSuggestion(
                category="missing_skills",
                suggestion=f"Integrate key missing required skills: {skills_str}.",
                impact="high",
                example=f"Added skills section entry: Technical Proficiencies — {skills_str}.",
            )
        )

    # Rule 2: Missing Keywords
    if req.missing_keywords:
        kw_str = ", ".join(req.missing_keywords[:5])
        suggestions.append(
            ResumeSuggestion(
                category="keywords",
                suggestion=f"Incorporate domain keywords throughout project descriptions: {kw_str}.",
                impact="medium",
                example=f"Demonstrated experience with {req.missing_keywords[0]} in recent production deployment.",
            )
        )

    # Rule 3: Weak Action Verbs check
    weak_phrases = ["responsible for", "worked on", "helped with", "assisted in", "duties included"]
    found_weak = [p for p in weak_phrases if p in text_lower]
    if found_weak:
        suggestions.append(
            ResumeSuggestion(
                category="impact_phrasing",
                suggestion=f"Replace passive phrasing ('{found_weak[0]}') with strong action verbs.",
                impact="high",
                example="Instead of 'Responsible for managing database', use 'Architected and optimized PostgreSQL database schema'.",
            )
        )

    # Rule 4: Quantified metrics check
    numbers_found = len(re.findall(r"\b\d+%\b|\$\d+|\b\d+\s*(?:users|clients|projects|latency|ms|requests|x)\b", text_lower))
    if numbers_found < 2:
        suggestions.append(
            ResumeSuggestion(
                category="formatting",
                suggestion="Add quantified business impact metrics (% improvements, revenue, user scale, latency reductions).",
                impact="high",
                example="Optimized REST API endpoint response time by 42%, reducing p99 latency from 350ms to 200ms.",
            )
        )

    # Rule 5: Resume section completeness check
    missing_sections = []
    if "experience" not in text_lower and "employment" not in text_lower and "work history" not in text_lower:
        missing_sections.append("Professional Experience")
    if "education" not in text_lower and "academic" not in text_lower:
        missing_sections.append("Education")
    if "skills" not in text_lower and "technologies" not in text_lower:
        missing_sections.append("Technical Skills")

    if missing_sections:
        suggestions.append(
            ResumeSuggestion(
                category="formatting",
                suggestion=f"Ensure clear standard section headings: missing standard heading for {', '.join(missing_sections)}.",
                impact="medium",
                example="Use standard headers like 'Work Experience', 'Skills & Technologies', 'Education'.",
            )
        )

    # Rule 6: General length & structure fallback
    if len(suggestions) < 3:
        suggestions.append(
            ResumeSuggestion(
                category="general",
                suggestion="Tailor summary statement to align directly with target role requirements.",
                impact="low",
                example="Software Engineer specializing in scalable cloud microservices and real-time backend systems.",
            )
        )

    return suggestions
