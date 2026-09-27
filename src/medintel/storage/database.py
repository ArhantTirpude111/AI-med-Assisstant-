"""Relational persistence (SQLAlchemy 2.0).

Tracks provenance and history alongside the vector index:

* ``documents``  — one row per ingested report (hash, page/chunk counts, status)
* ``chunks``     — chunk metadata (the text lives in the vector store)
* ``entities``   — extracted structured metadata (labs, dx, meds, …)
* ``qa_history`` — every question/answer with confidence and refusal flag
* ``citations``  — evidence references linked to a Q&A turn

SQLite is the default (zero-config, file-based); the same models run on Postgres
by changing ``paths.db_url`` — no code changes required.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from medintel.config import get_settings
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    Answer,
    EntityType,
    LabFlag,
    LabValue,
    MedicalEntities,
    MedicalEntity,
    RawDocument,
)

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    used_ocr: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(String(32), default="indexed")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    chunks: Mapped[list[ChunkRecord]] = relationship(back_populates="document", cascade="all, delete-orphan")
    entities: Mapped[list[EntityRecord]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"))
    page_number: Mapped[int] = mapped_column(default=1)
    chunk_index: Mapped[int] = mapped_column(default=0)
    char_start: Mapped[int] = mapped_column(default=0)
    char_end: Mapped[int] = mapped_column(default=0)
    content: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


class EntityRecord(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id"))
    entity_type: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(String(512))
    normalized: Mapped[str | None] = mapped_column(String(512), nullable=True)
    value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reference_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    flag: Mapped[str | None] = mapped_column(String(16), nullable=True)
    page_number: Mapped[int | None] = mapped_column(nullable=True)

    document: Mapped[Document] = relationship(back_populates="entities")


class QAHistory(Base):
    __tablename__ = "qa_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(default=0.0)
    confidence_label: Mapped[str] = mapped_column(String(16), default="insufficient")
    refused: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    citations: Mapped[list[CitationRecord]] = relationship(back_populates="qa", cascade="all, delete-orphan")


class CitationRecord(Base):
    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    qa_id: Mapped[int] = mapped_column(ForeignKey("qa_history.id"))
    filename: Mapped[str] = mapped_column(String(512))
    page_number: Mapped[int] = mapped_column(default=1)
    snippet: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(default=0.0)

    qa: Mapped[QAHistory] = relationship(back_populates="citations")


# ---------------------------------------------------------------------------
class Repository:
    """Thin data-access layer over the SQLAlchemy models."""

    def __init__(self, db_url: str | None = None) -> None:
        self.db_url = db_url or get_settings().paths.db_url
        connect_args = {"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
        self.engine = create_engine(self.db_url, connect_args=connect_args, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        logger.info("Repository ready (%s)", self.db_url)

    def session(self) -> Session:
        return self._session_factory()

    # -- writes -------------------------------------------------------------
    def save_document(
        self,
        document: RawDocument,
        chunk_count: int,
        entities: MedicalEntities | None = None,
    ) -> None:
        with self.session() as s, s.begin():
            s.merge(
                Document(
                    doc_id=document.doc_id,
                    filename=document.filename,
                    source_type=document.source_type.value,
                    sha256=document.sha256,
                    page_count=document.page_count,
                    chunk_count=chunk_count,
                    used_ocr=document.used_ocr,
                    status="indexed",
                )
            )
            if entities:
                self._persist_entities(s, document.doc_id, entities)

    @staticmethod
    def _persist_entities(s: Session, doc_id: str, entities: MedicalEntities) -> None:
        groups = [
            entities.diagnoses, entities.symptoms, entities.medications,
            entities.procedures, entities.anatomy, entities.recommendations,
            entities.physician_notes,
        ]
        for group in groups:
            for e in group:
                s.add(EntityRecord(
                    doc_id=doc_id, entity_type=e.entity_type.value, text=e.text,
                    normalized=e.normalized, page_number=e.page_number,
                ))
        for lab in entities.lab_results:
            s.add(EntityRecord(
                doc_id=doc_id, entity_type="lab", text=lab.name, value=lab.value,
                unit=lab.unit, reference_range=lab.reference_range,
                flag=lab.flag.value, page_number=lab.page_number,
            ))

    def record_qa(self, answer: Answer) -> int:
        with self.session() as s, s.begin():
            qa = QAHistory(
                question=answer.question,
                answer=answer.answer_text,
                confidence=answer.confidence.retrieval_confidence,
                confidence_label=answer.confidence.label.value,
                refused=answer.refused,
            )
            for c in answer.citations:
                qa.citations.append(CitationRecord(
                    filename=c.filename, page_number=c.page_number,
                    snippet=c.snippet, score=c.score,
                ))
            s.add(qa)
            s.flush()
            return qa.id

    # -- reads --------------------------------------------------------------
    def list_documents(self) -> list[Document]:
        with self.session() as s:
            return list(s.query(Document).order_by(Document.created_at.desc()).all())

    def recent_qa(self, limit: int = 20) -> list[QAHistory]:
        with self.session() as s:
            return list(s.query(QAHistory).order_by(QAHistory.created_at.desc()).limit(limit).all())

    def load_entities(self) -> MedicalEntities:
        """Reconstruct the accumulated ``MedicalEntities`` from persisted rows.

        Lets a fresh process (e.g. a CLI invocation) recover the structured
        metadata extracted during a previous ingestion run.
        """
        entities = MedicalEntities()
        buckets = {
            EntityType.DIAGNOSIS: entities.diagnoses,
            EntityType.SYMPTOM: entities.symptoms,
            EntityType.MEDICATION: entities.medications,
            EntityType.PROCEDURE: entities.procedures,
            EntityType.ANATOMY: entities.anatomy,
            EntityType.RECOMMENDATION: entities.recommendations,
            EntityType.PHYSICIAN_NOTE: entities.physician_notes,
        }
        with self.session() as s:
            rows = list(s.query(EntityRecord).all())
        for r in rows:
            if r.entity_type == EntityType.LAB.value:
                entities.lab_results.append(
                    LabValue(
                        name=r.text, value=r.value, unit=r.unit,
                        reference_range=r.reference_range,
                        flag=LabFlag(r.flag) if r.flag else LabFlag.UNKNOWN,
                        page_number=r.page_number,
                    )
                )
                continue
            try:
                etype = EntityType(r.entity_type)
            except ValueError:
                continue
            bucket = buckets.get(etype)
            if bucket is not None:
                bucket.append(
                    MedicalEntity(
                        entity_type=etype, text=r.text,
                        normalized=r.normalized, page_number=r.page_number,
                    )
                )
        return entities

    def clear(self) -> None:
        with self.session() as s, s.begin():
            for model in (CitationRecord, QAHistory, EntityRecord, ChunkRecord, Document):
                s.query(model).delete()
        logger.info("Repository cleared.")
