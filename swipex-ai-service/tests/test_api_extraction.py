from app.core.config import settings
from tests.fixtures.generate_sample_files import create_sample_docx, create_sample_pdf


def _auth():
    return {"X-Internal-API-Key": settings.INTERNAL_API_KEY or "test-secret"}


def test_api_extract_text(client):
    pdf_bytes = create_sample_pdf("John Doe\nPython Developer\nSkills: Python, FastAPI")
    files = {"file": ("resume.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/api/v1/extract/text", files=files, headers=_auth())

    assert res.status_code == 200
    data = res.json()
    assert "John Doe" in data["text"]
    assert "Python" in data["text"]
    assert data["method"] in ("pdfplumber", "pypdf")


def test_api_extract_skills(client):
    payload = {"text": "Senior Engineer with 5+ years of Python and React experience in AWS."}
    res = client.post("/api/v1/extract/skills", json=payload, headers=_auth())

    assert res.status_code == 200
    data = res.json()
    canonicals = [s["normalized_name"] for s in data["skills"]]
    assert "Python" in canonicals
    assert "React.js" in canonicals
    assert "Amazon Web Services" in canonicals


def test_api_extract_resume(client):
    docx_bytes = create_sample_docx(
        "Jane Doe\nSkills: Python, PostgreSQL, Docker\nExperience: Jan 2021 - Present as Developer"
    )
    files = {"file": ("resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    res = client.post("/api/v1/extract/resume", files=files, headers=_auth())

    assert res.status_code == 200
    data = res.json()
    assert "document" in data
    assert "sections" in data
    assert "skills" in data
    assert "experience" in data
    assert data["total_years_experience"] >= 1.0
