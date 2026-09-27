"""MedIntelAssistant — the application service façade.

A single, UI-agnostic entry point that composes every layer: ingestion,
processing (chunking + extraction + NER), the RAG pipeline, the feature
services, and persistence. The Streamlit app and the CLI both drive this class,
keeping presentation thin and the core fully testable.

It also holds the in-session knowledge state (loaded documents + accumulated
structured entities) that powers the summary, timeline, differential, and
knowledge-graph features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from medintel.config import Settings, get_settings
from medintel.features.differential import DifferentialService
from medintel.features.followup import FollowupService
from medintel.features.knowledge_graph import KnowledgeGraphService
from medintel.features.summarizer import SummaryService
from medintel.features.terminology import TerminologyService
from medintel.features.timeline import TimelineService
from medintel.ingestion.document_service import DocumentService
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    Answer,
    DifferentialDiagnosis,
    MedicalEntities,
    RawDocument,
    ReportSummary,
    TerminologySimplification,
    TimelineEvent,
)
from medintel.processing.entity_recognition import MedicalNER
from medintel.processing.metadata_extractor import MedicalMetadataExtractor
from medintel.rag.pipeline import RAGPipeline
from medintel.storage.database import Repository

logger = get_logger(__name__)


@dataclass
class IngestReport:
    """Result of ingesting a batch of uploads."""

    documents: list[RawDocument] = field(default_factory=list)
    chunks_indexed: int = 0
    used_ocr: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        return len(self.documents)


class MedIntelAssistant:
    """High-level orchestrator for the entire MedIntel workflow."""

    def __init__(self, settings: Settings | None = None, persist: bool = True) -> None:
        self.settings = settings or get_settings()
        self.persist = persist

        # Core services
        self.documents = DocumentService(self.settings)
        self.pipeline = RAGPipeline(self.settings)
        self.extractor = MedicalMetadataExtractor()
        self.ner = MedicalNER(self.settings.ner)
        self.repository = Repository(self.settings.paths.db_url) if persist else None

        # Feature services (share the pipeline)
        self.summary_service = SummaryService(self.pipeline)
        self.terminology_service = TerminologyService(self.pipeline)
        self.differential_service = DifferentialService(self.pipeline)
        self.followup_service = FollowupService(self.pipeline)
        self.timeline_service = TimelineService()
        self.kg_service = KnowledgeGraphService()

        # In-session knowledge state
        self.loaded_documents: list[RawDocument] = []
        self.entities = MedicalEntities()

        # Recover previously-extracted structured metadata so a fresh process
        # (e.g. a CLI run) sees the same entities as the indexing session did.
        if self.repository and self.pipeline.is_ready:
            try:
                self.entities = self.repository.load_entities()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not reload entities from DB: %s", exc)

    # -- status -------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.pipeline.is_ready

    @property
    def chunk_count(self) -> int:
        return self.pipeline.indexed_chunk_count

    @property
    def document_count(self) -> int:
        return len(self.loaded_documents)

    @property
    def ocr_available(self) -> bool:
        return self.documents.ocr_available

    @property
    def ner_backend(self) -> str:
        return self.ner.backend

    # -- ingestion ----------------------------------------------------------
    def ingest(self, files: list[tuple[str, bytes]]) -> IngestReport:
        """Full ingestion: load → extract → index → persist."""
        report = IngestReport()
        for filename, data in files:
            try:
                doc = self.documents.ingest_bytes(filename, data)
            except Exception as exc:  # noqa: BLE001
                logger.error("Ingest failed for %s: %s", filename, exc)
                report.errors.append(f"{filename}: {exc}")
                continue

            # Structured + free-text extraction.
            extracted = self.extractor.extract(doc)
            ner_entities = self.ner.extract(doc)
            doc_entities = extracted.merge(ner_entities)
            self.entities = self.entities.merge(doc_entities)

            # Index for retrieval.
            chunks = self.pipeline.ingest_documents([doc], persist=self.persist)

            # Persist provenance.
            if self.repository:
                self.repository.save_document(doc, chunk_count=chunks, entities=doc_entities)

            self.loaded_documents.append(doc)
            report.documents.append(doc)
            report.chunks_indexed += chunks
            report.used_ocr = report.used_ocr or doc.used_ocr

        logger.info(
            "Ingestion complete: %d docs, %d chunks (%d errors)",
            report.document_count, report.chunks_indexed, len(report.errors),
        )
        return report

    # -- question answering -------------------------------------------------
    def ask(self, question: str) -> Answer:
        answer = self.pipeline.answer(question)
        if self.repository:
            try:
                self.repository.record_qa(answer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist Q&A: %s", exc)
        return answer

    # -- features -----------------------------------------------------------
    def summarize(self) -> ReportSummary:
        return self.summary_service.generate(entities=self.entities)

    def simplify_term(self, term: str) -> TerminologySimplification:
        return self.terminology_service.simplify(term)

    def differential(self) -> list[DifferentialDiagnosis]:
        return self.differential_service.suggest(entities=self.entities)

    def followups(self) -> list[str]:
        return self.followup_service.generate()

    def timeline(self) -> list[TimelineEvent]:
        return self.timeline_service.build(self.loaded_documents)

    def knowledge_graph(self) -> nx.Graph:
        return self.kg_service.build(self.entities)

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Clear the index, in-session state, and persisted records."""
        self.pipeline.reset_index()
        self.loaded_documents = []
        self.entities = MedicalEntities()
        if self.repository:
            self.repository.clear()
        logger.info("Assistant reset.")
