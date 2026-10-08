"""Tests for skill extraction, aliases, boundary guards, and experience years parsing."""

from app.services.ai.experience_extraction import extract_experience
from app.services.ai.skill_extraction import extract_skills_from_text


def test_extract_skills_aliases():
    resume = """
    TECHNICAL SKILLS:
    Languages: JS, TS, Python, Go
    Databases: Postgres, Mongo, Redis
    DevOps: K8s, Docker, CI/CD
    """
    skills = extract_skills_from_text(resume)
    canonicals = {s.normalized_name for s in skills}

    assert "JavaScript" in canonicals
    assert "TypeScript" in canonicals
    assert "Python" in canonicals
    assert "Go" in canonicals
    assert "PostgreSQL" in canonicals
    assert "MongoDB" in canonicals
    assert "Redis" in canonicals
    assert "Kubernetes" in canonicals
    assert "Docker" in canonicals
    assert "CI/CD" in canonicals


def test_ambiguous_single_letter_boundary_guards():
    # 'Cat' or 'Run' should NOT trigger 'C' or 'R' skill extraction
    plain_text = "The Cat ran down the street in Rome."
    skills = extract_skills_from_text(plain_text)
    canonicals = {s.normalized_name for s in skills}

    assert "C" not in canonicals
    assert "R" not in canonicals

    # Explicit C programming language context should match
    c_context_text = "Experienced C developer writing system code."
    c_skills = extract_skills_from_text(c_context_text)
    c_canonicals = {s.normalized_name for s in c_skills}

    assert "C" in c_canonicals


def test_skills_section_confidence_boost():
    text_in_section = """
    SKILLS:
    Python, FastAPI, Docker
    """
    text_in_body = "I once wrote a script in Python."

    skills_sec = extract_skills_from_text(text_in_section)
    py_sec = next(s for s in skills_sec if s.normalized_name == "Python")

    skills_body = extract_skills_from_text(text_in_body)
    py_body = next(s for s in skills_body if s.normalized_name == "Python")

    assert py_sec.confidence > py_body.confidence


def test_years_pattern_parsing():
    text = "5+ years of Python and 3.5 years in Kubernetes"
    skills = extract_skills_from_text(text)

    py_skill = next((s for s in skills if s.normalized_name == "Python"), None)
    k8s_skill = next((s for s in skills if s.normalized_name == "Kubernetes"), None)

    assert py_skill is not None
    assert py_skill.years == 5.0

    assert k8s_skill is not None
    assert k8s_skill.years == 3.5


def test_extract_experience_and_total_years():
    exp_text = """
    Senior Developer at TechCorp
    Jan 2020 - Dec 2022

    Backend Engineer at StartupInc
    Jan 2022 - Present
    """
    items, total_years = extract_experience(exp_text)

    assert len(items) == 2
    assert total_years >= 4.0
