"""Experience extraction, date range parsing, and total years calculation."""

import logging
import re
from datetime import datetime

from app.schemas.extraction import ExperienceItem

logger = logging.getLogger("swipex.ai.experience_extraction")

# Date range extraction patterns
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
YEAR = r"(?:19|20)\d{2}"

DATE_RANGE_PATTERN = re.compile(
    rf"({MONTHS}\s+{YEAR}|{YEAR}|[0-1]?\d/{YEAR})\s*(?:–|-|to|until)\s*({MONTHS}\s+{YEAR}|{YEAR}|[0-1]?\d/{YEAR}|Present|Current|Now)",
    re.IGNORECASE,
)

ROLE_PATTERN = re.compile(
    r"\b(?:Senior|Junior|Lead|Principal|Staff|Software|Data|DevOps|Backend|Frontend|Full\s*Stack|Cloud|ML|AI|Product|QA|Test)\s+(?:Engineer|Developer|Architect|Manager|Analyst|Consultant|Scientist|Specialist)\b",
    re.IGNORECASE,
)


def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object."""
    date_str = date_str.strip()
    if date_str.lower() in ("present", "current", "now"):
        return datetime.now()

    # Try Month Year format
    for fmt in ("%b %Y", "%B %Y", "%m/%Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try partial matching for year
    year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
    if year_match:
        return datetime(int(year_match.group(0)), 1, 1)

    return datetime.now()


def calculate_years(start: datetime, end: datetime) -> float:
    """Calculate duration in fractional years."""
    days = (end - start).days
    return max(0.1, round(days / 365.25, 1))


def merge_date_intervals(intervals: list[tuple[datetime, datetime]]) -> float:
    """Merge overlapping date intervals and return total unique years."""
    if not intervals:
        return 0.0

    # Sort by start date
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []

    for current in sorted_intervals:
        if not merged:
            merged.append(current)
        else:
            prev_start, prev_end = merged[-1]
            if current[0] <= prev_end:
                # Overlap -> extend end date
                merged[-1] = (prev_start, max(prev_end, current[1]))
            else:
                merged.append(current)

    total_days = sum((end - start).days for start, end in merged)
    return round(total_days / 365.25, 1)


def extract_experience(text: str) -> tuple[list[ExperienceItem], float]:
    """Extract experience items, role titles, durations, and calculate total years."""
    items: list[ExperienceItem] = []
    intervals: list[tuple[datetime, datetime]] = []

    if not text:
        return items, 0.0

    lines = text.splitlines()

    for line in lines:
        match = DATE_RANGE_PATTERN.search(line)
        if match:
            start_str = match.group(1)
            end_str = match.group(2)
            is_current = end_str.lower() in ("present", "current", "now")

            start_dt = parse_date(start_str)
            end_dt = parse_date(end_str)

            if start_dt <= end_dt:
                yrs = calculate_years(start_dt, end_dt)
                intervals.append((start_dt, end_dt))

                # Extract potential job title from line
                title_match = ROLE_PATTERN.search(line)
                title = title_match.group(0) if title_match else None

                items.append(
                    ExperienceItem(
                        title=title,
                        company=None,
                        start=start_str,
                        end=end_str,
                        is_current=is_current,
                        years=yrs,
                        snippet=line.strip()[:200],
                    )
                )

    total_years = merge_date_intervals(intervals)
    return items, total_years
