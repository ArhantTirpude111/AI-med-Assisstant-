"""Medical Named-Entity Recognition.

Two-tier strategy:

1. **SciSpaCy / MedSpaCy** (if installed) — provides high-recall biomedical span
   detection. We use it to surface candidate entities, then classify each span
   against our curated lexicons to assign a clinical type.
2. **Rule/gazetteer fallback** — a dictionary matcher over curated lexicons. It
   guarantees the feature works out-of-the-box (CI, demos, offline) without the
   ~1GB SciSpaCy model download.

Both paths return a uniform ``MedicalEntities`` object, so downstream features
(knowledge graph, timeline, differential) are agnostic to which ran.
"""

from __future__ import annotations

import re
from typing import Any

from medintel.config import NERConfig, get_settings
from medintel.logging_config import get_logger
from medintel.models.schemas import EntityType, MedicalEntities, MedicalEntity, RawDocument
from medintel.processing.vocab import (
    ANATOMY,
    COMMON_MEDICATIONS,
    DISEASES,
    MEDICATION_SUFFIXES,
    SYMPTOMS,
)

logger = get_logger(__name__)


class MedicalNER:
    """Extract diseases, symptoms, medications, and anatomy from free text."""

    def __init__(self, config: NERConfig | None = None) -> None:
        self.config = config or get_settings().ner
        self._nlp: Any | None = None
        self._backend = "rules"
        if self.config.enabled and self.config.backend in ("auto", "scispacy"):
            self._try_load_scispacy()

    @property
    def backend(self) -> str:
        return self._backend

    def _try_load_scispacy(self) -> None:
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(self.config.scispacy_model)
            self._backend = "scispacy"
            logger.info("Loaded SciSpaCy model '%s'", self.config.scispacy_model)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "SciSpaCy model unavailable (%s); using rule-based NER fallback.", exc
            )
            self._nlp = None
            self._backend = "rules"

    # -- public API ---------------------------------------------------------
    def extract(self, document: RawDocument) -> MedicalEntities:
        entities = MedicalEntities()
        seen: set[tuple[str, str]] = set()
        for page in document.pages:
            text = page.text
            candidates = self._candidate_spans(text)
            for span in candidates:
                etype = self._classify(span)
                if etype is None:
                    continue
                key = (etype.value, span.lower())
                if key in seen:
                    continue
                seen.add(key)
                self._add(entities, etype, span, page.page_number)
        return entities

    def extract_many(self, documents: list[RawDocument]) -> MedicalEntities:
        merged = MedicalEntities()
        for doc in documents:
            merged = merged.merge(self.extract(doc))
        return merged

    # -- candidate generation ----------------------------------------------
    def _candidate_spans(self, text: str) -> set[str]:
        spans: set[str] = set()
        # 1. Gazetteer matches (always run — high precision).
        lowered = text.lower()
        for term in DISEASES | SYMPTOMS | COMMON_MEDICATIONS | ANATOMY:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                spans.add(term)
        # 2. Drug-suffix heuristic to catch meds not in the gazetteer.
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
            if token.lower().endswith(MEDICATION_SUFFIXES):
                spans.add(token.lower())
        # 3. SciSpaCy spans (recall booster), classified later.
        if self._nlp is not None:
            try:
                doc = self._nlp(text[:100_000])  # guard against huge pages
                for ent in doc.ents:
                    spans.add(ent.text.lower().strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("SciSpaCy inference failed on a page: %s", exc)
        return spans

    # -- classification -----------------------------------------------------
    @staticmethod
    def _classify(span: str) -> EntityType | None:
        s = span.lower().strip()
        if s in DISEASES or any(d in s for d in DISEASES):
            return EntityType.DIAGNOSIS
        if s in SYMPTOMS or any(sym in s for sym in SYMPTOMS):
            return EntityType.SYMPTOM
        if s in COMMON_MEDICATIONS or s.endswith(MEDICATION_SUFFIXES):
            return EntityType.MEDICATION
        if s in ANATOMY:
            return EntityType.ANATOMY
        return None

    @staticmethod
    def _add(entities: MedicalEntities, etype: EntityType, text: str, page: int) -> None:
        entity = MedicalEntity(entity_type=etype, text=text, normalized=text.lower(), page_number=page)
        bucket = {
            EntityType.DIAGNOSIS: entities.diagnoses,
            EntityType.SYMPTOM: entities.symptoms,
            EntityType.MEDICATION: entities.medications,
            EntityType.ANATOMY: entities.anatomy,
        }[etype]
        bucket.append(entity)
