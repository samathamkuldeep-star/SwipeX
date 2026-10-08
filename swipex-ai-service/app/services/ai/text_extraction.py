"""PDF and DOCX text extraction with magic byte detection and library fallbacks."""

import io
import logging
import zipfile

import pdfplumber
import pypdf
from docx import Document

from app.core.config import settings
from app.core.exceptions import (
    CorruptFileError,
    FileTooLargeError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.schemas.extraction import ExtractedDocument

logger = logging.getLogger("swipex.ai.text_extraction")


def detect_file_format(data: bytes) -> str:
    """Detect file type via magic bytes instead of extension or MIME type."""
    if not data:
        raise CorruptFileError("Uploaded file is empty.")

    if data.startswith(b"%PDF-"):
        return "pdf"

    # DOCX is a ZIP file containing word/document.xml
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            raise CorruptFileError("Uploaded DOCX file is not a valid zip archive.") from None

    raise UnsupportedFileTypeError("Unsupported file format. Only PDF and DOCX files are allowed.")


def extract_pdf(data: bytes) -> ExtractedDocument:
    """Extract text from PDF using pdfplumber with fallback to pypdf."""
    warnings: list[str] = []
    text_chunks: list[str] = []
    page_count = 0
    method = "pdfplumber"

    # Check password protection via pypdf first
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CorruptFileError("PDF file is password-protected and cannot be parsed.")
        page_count = len(reader.pages)
    except CorruptFileError:
        raise
    except Exception as e:
        logger.warning(f"pypdf reader check failed: {e}")

    # Primary parsing using pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages_to_process = min(len(pdf.pages), settings.MAX_PDF_PAGES)
            if len(pdf.pages) > settings.MAX_PDF_PAGES:
                warnings.append(f"PDF exceeded max limit of {settings.MAX_PDF_PAGES} pages. Truncated extra pages.")

            for idx in range(pages_to_process):
                try:
                    p = pdf.pages[idx]
                    page_text = p.extract_text(layout=False) or ""
                    
                    # Extract text from tables if present
                    tables = p.extract_tables() or []
                    table_text_list = []
                    for table in tables:
                        for row in table:
                            row_str = " ".join([cell.strip() for cell in row if cell and isinstance(cell, str)])
                            if row_str:
                                table_text_list.append(row_str)
                    
                    combined = page_text.strip()
                    if table_text_list:
                        combined += "\n" + "\n".join(table_text_list)
                    
                    if combined:
                        text_chunks.append(combined)
                except Exception as page_err:
                    warnings.append(f"Failed to extract page {idx + 1}: {page_err}")

            page_count = pages_to_process
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}. Falling back to pypdf.")
        method = "pypdf"
        text_chunks = []

    # Fallback to pypdf if pdfplumber extracted < 50 chars or raised
    combined_text = "\n\n".join(text_chunks).strip()
    if len(combined_text) < 50 and method == "pdfplumber":
        logger.info("pdfplumber returned < 50 characters; attempting pypdf fallback...")
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            text_chunks = []
            pages_to_process = min(len(reader.pages), settings.MAX_PDF_PAGES)
            for idx in range(pages_to_process):
                try:
                    txt = reader.pages[idx].extract_text() or ""
                    if txt.strip():
                        text_chunks.append(txt.strip())
                except Exception as p_err:
                    warnings.append(f"pypdf page {idx + 1} error: {p_err}")
            combined_text = "\n\n".join(text_chunks).strip()
            method = "pypdf"
        except Exception as e:
            logger.error(f"pypdf fallback failed: {e}")
            raise CorruptFileError(f"Corrupt PDF document: {e}") from None

    if not combined_text or len(combined_text) < 10:
        raise NoExtractableTextError(
            "No extractable text found in PDF (e.g. scanned image PDF). OCR is not supported."
        )

    return ExtractedDocument(
        text=combined_text,
        pages=max(1, page_count),
        method=method,
        warnings=warnings,
    )


def extract_docx(data: bytes) -> ExtractedDocument:
    """Extract text from DOCX file including paragraphs, tables, and headers/footers."""
    warnings: list[str] = []
    text_chunks: list[str] = []

    try:
        doc = Document(io.BytesIO(data))

        # Paragraphs
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                text_chunks.append(txt)

        # Tables
        for table in doc.tables:
            for row in table.rows:
                row_str = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_str:
                    text_chunks.append(row_str)

        # Headers & Footers
        for section in doc.sections:
            if section.header and section.header.paragraphs:
                for hp in section.header.paragraphs:
                    if hp.text.strip():
                        text_chunks.append(hp.text.strip())
            if section.footer and section.footer.paragraphs:
                for fp in section.footer.paragraphs:
                    if fp.text.strip():
                        text_chunks.append(fp.text.strip())

    except Exception as e:
        logger.error(f"docx parsing failed: {e}")
        raise CorruptFileError(f"Corrupt or invalid DOCX document: {e}") from None

    combined_text = "\n".join(text_chunks).strip()
    if not combined_text:
        raise NoExtractableTextError("No extractable text found in DOCX file.")

    return ExtractedDocument(
        text=combined_text,
        pages=1,
        method="python-docx",
        warnings=warnings,
    )


def extract_text(
    data: bytes,
    content_type: str | None = None,
    filename: str | None = None,
) -> ExtractedDocument:
    """Extract clean text from PDF or DOCX file payload."""
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"File size ({len(data) / (1024 * 1024):.2f} MB) exceeds limit of {settings.MAX_FILE_SIZE_MB} MB."
        )

    file_format = detect_file_format(data)
    if file_format == "pdf":
        return extract_pdf(data)
    elif file_format == "docx":
        return extract_docx(data)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file format: {file_format}")
