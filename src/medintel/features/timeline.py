"""Patient timeline generator.

Scans documents for dated lines and classifies each into a clinical event type
(diagnosis, medication, lab, symptom, procedure). Deterministic and fully
auditable — every event keeps its source filename and page so it can be traced
back to the report. Dates are normalized to ISO where parseable.
"""

from __future__ import annotations

import re
from datetime import date

from medintel.logging_config import get_logger
from medintel.models.schemas import EntityType, RawDocument, TimelineEvent
from medintel.processing.vocab import COMMON_LAB_TESTS, COMMON_MEDICATIONS, DISEASES, SYMPTOMS

logger = get_logger(__name__)

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

# Ordered date patterns, each returning (year, month, day) via named groups.
_DATE_PATTERNS = [
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b"),               # 2025-03-14
    re.compile(r"\b(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})\b"),               # 14/03/2025
    re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})\b"),  # March 14, 2025
    re.compile(r"\b(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<y>\d{4})\b"),    # 14 March 2025
]

_PROCEDURE_TERMS = {"mri", "ct scan", "ct", "ultrasound", "x-ray", "xray", "biopsy",
                    "endoscopy", "colonoscopy", "ecg", "ekg", "echocardiogram", "surgery"}


class TimelineService:
    def build(self, documents: list[RawDocument]) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []
        for doc in documents:
            for page in doc.pages:
                for raw_line in page.text.splitlines():
                    line = raw_line.strip()
                    if not line or len(line) > 160:
                        continue
                    parsed = self._parse_date(line)
                    if parsed is None:
                        continue
                    iso, matched = parsed
                    description = line.replace(matched, "").strip(" -:•\t")
                    if len(description) < 3:
                        continue
                    events.append(
                        TimelineEvent(
                            date=iso,
                            event_type=self._classify(line),
                            description=description[:160],
                            page_number=page.page_number,
                            filename=doc.filename,
                        )
                    )
        events.sort(key=lambda e: e.date or "9999")
        logger.info("Timeline built with %d dated events.", len(events))
        return events

    # -- date parsing -------------------------------------------------------
    def _parse_date(self, line: str) -> tuple[str | None, str] | None:
        for pattern in _DATE_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            groups = m.groupdict()
            try:
                if "mon" in groups and groups.get("mon"):
                    month = _MONTHS.get(groups["mon"][:3].lower())
                    if not month:
                        continue
                    day, year = int(groups["d"]), int(groups["y"])
                else:
                    year, month, day = int(groups["y"]), int(groups["m"]), int(groups["d"])
                iso = date(year, month, day).isoformat()
                return iso, m.group(0)
            except (ValueError, KeyError):
                continue
        return None

    # -- classification -----------------------------------------------------
    @staticmethod
    def _classify(line: str) -> EntityType:
        low = line.lower()
        if any(p in low for p in _PROCEDURE_TERMS):
            return EntityType.PROCEDURE
        if any(d in low for d in DISEASES):
            return EntityType.DIAGNOSIS
        if any(m in low for m in COMMON_MEDICATIONS):
            return EntityType.MEDICATION
        if any(t in low for t in COMMON_LAB_TESTS):
            return EntityType.LAB
        if any(s in low for s in SYMPTOMS):
            return EntityType.SYMPTOM
        return EntityType.PHYSICIAN_NOTE
