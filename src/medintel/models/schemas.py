"""Domain models for MedIntel AI.

This module is the *single source of truth* for the data structures that flow
through the system: documents and pages produced by ingestion, chunks produced
by processing, vectors stored in FAISS, and the citation-rich answers returned
to the user.

All models are Pydantic v2 ``BaseModel`` subclasses so we get validation,
serialization, and self-documenting schemas for free — important in a regulated
domain where data provenance matters.
"""

from __future__ import annotations

import enum
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field


def _new_id(prefix: str) -> str:
    """Return a short, collision-resistant, human-debuggable identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class SourceType(str, enum.Enum):
    """How a document arrived / what kind of file it is."""

    PDF = "pdf"
    SCANNED_PDF = "scanned_pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ExtractionMethod(str, enum.Enum):
    """How the text for a page was recovered."""

    NATIVE = "native"   # embedded text layer
    OCR = "ocr"         # rasterized + Tesseract


class EntityType(str, enum.Enum):
    """Categories of clinically-relevant entities we extract."""

    DIAGNOSIS = "diagnosis"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    LAB = "lab"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    RECOMMENDATION = "recommendation"
    PHYSICIAN_NOTE = "physician_note"


class LabFlag(str, enum.Enum):
    """Interpretation of a lab value against its reference range."""

    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class ConfidenceLabel(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class ExplanationLevel(str, enum.Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    PATIENT_FRIENDLY = "patient_friendly"
    ELI5 = "eli5"


# ---------------------------------------------------------------------------
# Ingestion-layer models
# ---------------------------------------------------------------------------
class Page(BaseModel):
    """A single page of a document with its recovered text."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE
    char_count: int = 0

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        self.char_count = len(self.text)


class RawDocument(BaseModel):
    """An ingested document, page-segmented to preserve citation provenance."""

    doc_id: str = Field(default_factory=lambda: _new_id("doc"))
    filename: str
    source_type: SourceType = SourceType.UNKNOWN
    pages: list[Page] = Field(default_factory=list)
    sha256: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_count(self) -> int:
        return len(self.pages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def used_ocr(self) -> bool:
        return any(p.extraction_method is ExtractionMethod.OCR for p in self.pages)

    def compute_sha256(self, raw_bytes: bytes) -> str:
        """Compute and store a content hash for dedup / provenance."""
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        return self.sha256


# ---------------------------------------------------------------------------
# Processing-layer models
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """A retrievable unit of text plus the metadata needed to cite it."""

    chunk_id: str = Field(default_factory=lambda: _new_id("chk"))
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_estimate(self) -> int:
        # Cheap heuristic (~4 chars/token) used for context-budget accounting.
        return max(1, len(self.text) // 4)


class LabValue(BaseModel):
    """A structured laboratory result extracted from a report."""

    name: str
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: LabFlag = LabFlag.UNKNOWN
    page_number: int | None = None


class MedicalEntity(BaseModel):
    """A clinically-relevant span detected by NER / rule extraction."""

    entity_type: EntityType
    text: str
    normalized: str | None = None
    page_number: int | None = None
    score: float = 1.0


class MedicalEntities(BaseModel):
    """Aggregated structured metadata extracted from a document set."""

    diagnoses: list[MedicalEntity] = Field(default_factory=list)
    symptoms: list[MedicalEntity] = Field(default_factory=list)
    medications: list[MedicalEntity] = Field(default_factory=list)
    procedures: list[MedicalEntity] = Field(default_factory=list)
    anatomy: list[MedicalEntity] = Field(default_factory=list)
    recommendations: list[MedicalEntity] = Field(default_factory=list)
    physician_notes: list[MedicalEntity] = Field(default_factory=list)
    lab_results: list[LabValue] = Field(default_factory=list)

    def merge(self, other: MedicalEntities) -> MedicalEntities:
        """Combine two extraction results (used across multi-doc uploads)."""
        return MedicalEntities(
            diagnoses=self.diagnoses + other.diagnoses,
            symptoms=self.symptoms + other.symptoms,
            medications=self.medications + other.medications,
            procedures=self.procedures + other.procedures,
            anatomy=self.anatomy + other.anatomy,
            recommendations=self.recommendations + other.recommendations,
            physician_notes=self.physician_notes + other.physician_notes,
            lab_results=self.lab_results + other.lab_results,
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.diagnoses, self.symptoms, self.medications, self.procedures,
                self.anatomy, self.recommendations, self.physician_notes, self.lab_results,
            ]
        )


# ---------------------------------------------------------------------------
# Retrieval / answer models
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A chunk returned by semantic search, with its similarity score."""

    chunk: Chunk
    score: float = Field(..., description="Cosine similarity in [0, 1]")


class Citation(BaseModel):
    """A user-facing evidence reference attached to an answer."""

    doc_id: str
    filename: str
    page_number: int
    snippet: str
    score: float

    def render(self) -> str:
        return (
            f"Source: {self.filename} — Page {self.page_number} "
            f"(relevance {self.score:.0%})\n\"{self.snippet}\""
        )


class Confidence(BaseModel):
    """Quantified trust signal for an answer."""

    retrieval_confidence: float = Field(0.0, ge=0.0, le=1.0)
    top_similarity: float = 0.0
    mean_similarity: float = 0.0
    source_count: int = 0
    label: ConfidenceLabel = ConfidenceLabel.INSUFFICIENT


class GuardrailFlag(BaseModel):
    """A single guardrail event raised while answering."""

    code: str
    message: str
    severity: str = "info"  # info | warning | critical


class Answer(BaseModel):
    """The complete, trust-bearing response returned to the UI."""

    question: str
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    refused: bool = False
    disclaimer: str = ""
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def has_emergency_flag(self) -> bool:
        return any(f.code == "emergency" for f in self.guardrail_flags)


# ---------------------------------------------------------------------------
# Feature-layer models
# ---------------------------------------------------------------------------
class TerminologySimplification(BaseModel):
    term: str
    beginner: str = ""
    intermediate: str = ""
    patient_friendly: str = ""
    eli5: str = ""


class ReportSummary(BaseModel):
    executive_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    abnormal_values: list[str] = Field(default_factory=list)
    risk_indicators: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class DifferentialDiagnosis(BaseModel):
    condition: str
    rationale: str
    supporting_findings: list[str] = Field(default_factory=list)
    disclaimer: str = "For educational purposes only — not a diagnosis."


class TimelineEvent(BaseModel):
    date: str | None = None        # ISO date or free-text if unparseable
    event_type: EntityType
    description: str
    page_number: int | None = None
    filename: str | None = None
