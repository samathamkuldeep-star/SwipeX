"""Text preprocessing, normalization, section splitting, and PII redaction."""

import logging
import re
import unicodedata

from app.core.config import settings

logger = logging.getLogger("swipex.ai.preprocessing")

# Regex patterns for PII redaction
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
ADDRESS_PATTERN = re.compile(r"\d+\s+[A-Za-z0-9\s,.]+?\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b", re.IGNORECASE)

# Standard resume headings mapping
SECTION_HEADERS = {
    "summary": re.compile(r"^(?:summary|profile|about\s+me|professional\s+summary|objective|career\s+objective)\b", re.IGNORECASE),
    "skills": re.compile(r"^(?:skills|technical\s+skills|core\s+competencies|technologies|skills\s+&\s+tools|expertise)\b", re.IGNORECASE),
    "experience": re.compile(r"^(?:experience|work\s+experience|employment\s+history|professional\s+experience|work\s+history)\b", re.IGNORECASE),
    "education": re.compile(r"^(?:education|academic\s+background|academic\s+qualifications|qualification)\b", re.IGNORECASE),
    "projects": re.compile(r"^(?:projects|personal\s+projects|key\s+projects|academic\s+projects)\b", re.IGNORECASE),
    "certifications": re.compile(r"^(?:certifications|certificates|licenses|certifications\s+&\s+licenses)\b", re.IGNORECASE),
}


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Normalize text: NFKC, strip control/zero-width chars, collapse whitespace, de-hyphenate linebreaks."""
    warnings: list[str] = []
    if not text:
        return "", warnings

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width characters and control chars (except linebreaks and tabs)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C")

    # De-hyphenate line breaks (e.g., "commu-\nnication" -> "communication")
    text = re.sub(r"(\w+)-\s*[\r\n]+\s*(\w+)", r"\1\2", text)

    # Normalize bullet points into clean dashes
    text = re.sub(r"[\u2022\u2023\u2043\u204c\u204d\u2219\u25cb\u25cf\u25e6\u25aa\u25ab\u25b6]", "-", text)

    # Collapse multi-spaces per line while keeping structural linebreaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Truncate text if exceeding MAX_TEXT_CHARS
    if len(text) > settings.MAX_TEXT_CHARS:
        text = text[: settings.MAX_TEXT_CHARS]
        warnings.append(f"Text truncated to {settings.MAX_TEXT_CHARS} characters.")

    return text, warnings


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into sections (summary, skills, experience, education, projects, certifications)."""
    sections: dict[str, str] = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": "",
        "other": "",
    }

    if not text:
        return sections

    lines = text.splitlines()
    current_section = "summary"
    section_lines: dict[str, list[str]] = {key: [] for key in sections}

    for line in lines:
        cleaned_line = line.strip().strip(":").strip("-").strip()
        matched_new_section = None

        if len(cleaned_line) < 45:  # Headings are short lines
            for sec_key, pattern in SECTION_HEADERS.items():
                if pattern.match(cleaned_line):
                    matched_new_section = sec_key
                    break

        if matched_new_section:
            current_section = matched_new_section
        else:
            section_lines[current_section].append(line)

    for key, lines_list in section_lines.items():
        sections[key] = "\n".join(lines_list).strip()

    return sections


def redact_pii(text: str) -> str:
    """Mask PII (emails, phone numbers, URLs, addresses) for privacy protection before LLM dispatch."""
    if not text:
        return ""

    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = URL_PATTERN.sub("[REDACTED_URL]", redacted)
    redacted = ADDRESS_PATTERN.sub("[REDACTED_ADDRESS]", redacted)

    return redacted
