"""Tests for PDF/DOCX text extraction, magic bytes, and error conditions."""

import pytest

from app.core.exceptions import (
    CorruptFileError,
    FileTooLargeError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.services.ai.text_extraction import detect_file_format, extract_text
from tests.fixtures.generate_sample_files import (
    create_encrypted_pdf,
    create_multipage_pdf,
    create_sample_docx,
    create_sample_pdf,
    create_scanned_image_pdf,
)


def test_detect_file_format():
    pdf_bytes = create_sample_pdf()
    assert detect_file_format(pdf_bytes) == "pdf"

    docx_bytes = create_sample_docx()
    assert detect_file_format(docx_bytes) == "docx"

    with pytest.raises(UnsupportedFileTypeError):
        detect_file_format(b"PLAIN TEXT FILE WITHOUT MAGIC BYTES")

    with pytest.raises(CorruptFileError):
        detect_file_format(b"")


def test_extract_pdf_success():
    sample_text = "Alice Johnson\nBackend Developer\nSkills: Python, Django, PostgreSQL"
    pdf_bytes = create_sample_pdf(sample_text)
    doc = extract_text(pdf_bytes, filename="resume.pdf")

    assert "Alice Johnson" in doc.text
    assert "Python" in doc.text
    assert doc.pages == 1
    assert doc.method in ("pdfplumber", "pypdf")


def test_extract_multipage_pdf():
    pages = [f"Page {i} content with Python and FastApi" for i in range(1, 4)]
    pdf_bytes = create_multipage_pdf(pages)
    doc = extract_text(pdf_bytes, filename="multipage.pdf")

    assert doc.pages == 3
    assert "Page 1 content" in doc.text
    assert "Page 3 content" in doc.text


def test_extract_docx_success():
    sample_text = "Bob Smith\nFrontend Developer\nSkills: React, TypeScript, Tailwind"
    docx_bytes = create_sample_docx(sample_text)
    doc = extract_text(docx_bytes, filename="resume.docx")

    assert "Bob Smith" in doc.text
    assert "React" in doc.text
    assert doc.method == "python-docx"


def test_encrypted_pdf_raises_error():
    enc_pdf = create_encrypted_pdf()
    with pytest.raises(CorruptFileError) as exc:
        extract_text(enc_pdf, filename="encrypted.pdf")
    assert "password-protected" in str(exc.value.message).lower()


def test_scanned_image_pdf_raises_error():
    scanned_pdf = create_scanned_image_pdf()
    with pytest.raises(NoExtractableTextError) as exc:
        extract_text(scanned_pdf, filename="scanned.pdf")
    assert "no extractable text" in str(exc.value.message).lower()


def test_file_too_large():
    # 6 MB dummy payload (> 5 MB limit)
    large_payload = b"%PDF-1.5\n" + b"0" * (6 * 1024 * 1024)
    with pytest.raises(FileTooLargeError):
        extract_text(large_payload, filename="huge.pdf")


def test_invalid_docx_magic_bytes_mismatch():
    invalid_bytes = b"NOT A REAL ZIP OR PDF DOCUMENT"
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(invalid_bytes, filename="resume.docx")
