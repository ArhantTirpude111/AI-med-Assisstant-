"""Structured metadata extraction from clinical reports.

This module performs *deterministic*, explainable extraction of:

* **Lab results** — name / value / unit / reference range / normal-abnormal flag
* **Section content** — diagnoses, medications, symptoms, recommendations,
  physician notes (detected via section headers)

It is complementary to :mod:`medintel.processing.entity_recognition` (free-text
NER). Rule-based extraction is preferred for *structured* fields like labs
because it is auditable and never hallucinates a number that wasn't on the page.
"""

from __future__ import annotations

import re

from medintel.logging_config import get_logger
from medintel.models.schemas import (
    EntityType,
    LabFlag,
    LabValue,
    MedicalEntities,
    MedicalEntity,
    Page,
    RawDocument,
)
from medintel.processing.vocab import COMMON_LAB_TESTS, SECTION_HEADERS

logger = get_logger(__name__)

# Matches e.g. "ALT (SGPT): 78 U/L (7-56) H"  or  "Hemoglobin 11.2 g/dL 13.5-17.5"
_LAB_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9 \-\(\)/]{1,38}?)\s*[:=\-]?\s+"
    r"(?P<value>\d{1,3}(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|[a-zA-Zµμ][a-zA-Z0-9µμ/\^]{0,9})?\s*"
    r"(?:\(?\s*(?:ref(?:erence)?\.?\s*(?:range)?\s*[:\-]?\s*)?"
    r"(?P<low>\d{1,3}(?:[.,]\d+)?)\s*[-–]\s*(?P<high>\d{1,3}(?:[.,]\d+)?)\s*\)?)?"
    r"\s*(?P<flag>HIGH|LOW|HI|LO|H|L|\*)?\s*$",
    re.IGNORECASE,
)

_HEADER_LOOKUP: dict[str, str] = {
    alias.lower(): section
    for section, aliases in SECTION_HEADERS.items()
    for alias in aliases
}

_SECTION_TO_ENTITY = {
    "diagnosis": EntityType.DIAGNOSIS,
    "medication": EntityType.MEDICATION,
    "symptom": EntityType.SYMPTOM,
    "recommendation": EntityType.RECOMMENDATION,
    "physician_note": EntityType.PHYSICIAN_NOTE,
}


class MedicalMetadataExtractor:
    """Deterministic extractor for structured clinical metadata."""

    def extract(self, document: RawDocument) -> MedicalEntities:
        entities = MedicalEntities()
        for page in document.pages:
            entities.lab_results.extend(self._extract_labs(page))
            self._extract_sections(page, entities)
        logger.info(
            "Extracted from %s: %d labs, %d dx, %d meds, %d symptoms",
            document.filename,
            len(entities.lab_results),
            len(entities.diagnoses),
            len(entities.medications),
            len(entities.symptoms),
        )
        return entities

    def extract_many(self, documents: list[RawDocument]) -> MedicalEntities:
        merged = MedicalEntities()
        for doc in documents:
            merged = merged.merge(self.extract(doc))
        return merged

    # -- labs ---------------------------------------------------------------
    def _extract_labs(self, page: Page) -> list[LabValue]:
        labs: list[LabValue] = []
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line or len(line) > 80:
                continue
            match = _LAB_PATTERN.match(line)
            if not match:
                continue
            name = match.group("name").strip().rstrip(":").strip()
            low, high = match.group("low"), match.group("high")
            flag_token = match.group("flag")

            # High-precision gate: accept only if it looks like a known test,
            # OR a reference range / explicit flag is present.
            known = name.lower() in COMMON_LAB_TESTS or any(
                t in name.lower() for t in COMMON_LAB_TESTS
            )
            if not (known or (low and high) or flag_token):
                continue

            value = match.group("value")
            lab = LabValue(
                name=name,
                value=value,
                unit=(match.group("unit") or None),
                reference_range=(f"{low}-{high}" if low and high else None),
                flag=self._determine_flag(value, low, high, flag_token),
                page_number=page.page_number,
            )
            labs.append(lab)
        return labs

    @staticmethod
    def _determine_flag(value: str, low: str | None, high: str | None, token: str | None) -> LabFlag:
        if token:
            t = token.upper()
            if t in {"H", "HI", "HIGH"}:
                return LabFlag.HIGH
            if t in {"L", "LO", "LOW"}:
                return LabFlag.LOW
            if t == "*":
                return LabFlag.ABNORMAL
        if low and high:
            try:
                v = float(value.replace(",", "."))
                lo = float(low.replace(",", "."))
                hi = float(high.replace(",", "."))
                if v < lo:
                    return LabFlag.LOW
                if v > hi:
                    return LabFlag.HIGH
                return LabFlag.NORMAL
            except ValueError:
                return LabFlag.UNKNOWN
        return LabFlag.UNKNOWN

    # -- header-driven sections --------------------------------------------
    def _extract_sections(self, page: Page, entities: MedicalEntities) -> None:
        lines = page.text.splitlines()
        current: str | None = None
        for raw in lines:
            line = raw.strip()
            if not line:
                current = None
                continue
            section = self._match_header(line)
            if section:
                current = section
                # Some reports put content on the same line as the header.
                inline = self._strip_header(line)
                if inline:
                    self._append(entities, current, inline, page.page_number)
                continue
            if current:
                self._append(entities, current, line, page.page_number)

    @staticmethod
    def _match_header(line: str) -> str | None:
        candidate = line.lower().split(":")[0].strip()
        return _HEADER_LOOKUP.get(candidate)

    @staticmethod
    def _strip_header(line: str) -> str:
        return line.split(":", 1)[1].strip() if ":" in line else ""

    @staticmethod
    def _append(entities: MedicalEntities, section: str, text: str, page: int) -> None:
        text = text.lstrip("-•* ").strip()
        if not text or len(text) < 3:
            return
        entity_type = _SECTION_TO_ENTITY.get(section)
        if entity_type is None:
            return
        entity = MedicalEntity(entity_type=entity_type, text=text, page_number=page)
        target = {
            EntityType.DIAGNOSIS: entities.diagnoses,
            EntityType.MEDICATION: entities.medications,
            EntityType.SYMPTOM: entities.symptoms,
            EntityType.RECOMMENDATION: entities.recommendations,
            EntityType.PHYSICIAN_NOTE: entities.physician_notes,
        }[entity_type]
        target.append(entity)
