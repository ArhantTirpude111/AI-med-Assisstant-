"""The RAG pipeline — orchestrates retrieval-augmented, guarded answer generation.

This is the single composition point wiring together the embedder, vector store,
retriever, confidence scorer, guardrails, prompt templates, and LLM client. The
UI and CLI talk *only* to this class for Q&A, keeping presentation concerns out
of the core.

Answer lifecycle::

    question
      → emergency screening (guardrail)
      → embed + retrieve top-k chunks
      → confidence scoring
      → refusal gate (insufficient evidence?)        ─┐ refuse → canonical message
      → grounded prompt → LLM                          │
      → post-generation checks + citations + disclaimer┘
      → Answer
"""

from __future__ import annotations

import re

from medintel.config import Settings, get_settings
from medintel.features.confidence import ConfidenceScorer
from medintel.features.guardrails import Guardrails
from medintel.llm.client import BaseLLMClient, get_llm_client
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    Answer,
    Chunk,
    Citation,
    Confidence,
    GuardrailFlag,
    RawDocument,
    RetrievedChunk,
)
from medintel.processing.chunking import Chunker
from medintel.rag.embeddings import BaseEmbedder, get_embedder
from medintel.rag.prompts import build_qa_messages, format_context
from medintel.rag.retriever import Retriever
from medintel.rag.vector_store import FaissVectorStore

logger = get_logger(__name__)

_CITATION_TAG = re.compile(r"\[S(\d+)\]")


class RAGPipeline:
    """End-to-end retrieval-augmented medical Q&A."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: BaseEmbedder | None = None,
        store: FaissVectorStore | None = None,
        llm: BaseLLMClient | None = None,
        auto_load: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.llm = llm or get_llm_client(self.settings)
        self.chunker = Chunker(settings=self.settings)
        self.guardrails = Guardrails(self.settings.guardrails)
        self.scorer = ConfidenceScorer(self.settings.retrieval)

        self.store = store or self._init_store(auto_load)
        self.retriever = Retriever(self.embedder, self.store, self.settings.retrieval)

    # -- store lifecycle ----------------------------------------------------
    @property
    def _signature(self) -> str:
        return f"{self.settings.embeddings.provider}:{self.settings.embeddings.model}:{self.embedder.dimension}"

    def _init_store(self, auto_load: bool) -> FaissVectorStore:
        index_dir = self.settings.paths.index_dir
        if auto_load and FaissVectorStore.exists(index_dir):
            try:
                store = FaissVectorStore.load(index_dir)
                if store.dimension == self.embedder.dimension:
                    return store
                logger.warning("Persisted index dim mismatch; starting fresh.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load persisted index (%s); starting fresh.", exc)
        return FaissVectorStore(dimension=self.embedder.dimension, signature=self._signature)

    @property
    def is_ready(self) -> bool:
        return not self.store.is_empty

    @property
    def indexed_chunk_count(self) -> int:
        return self.store.count

    # -- ingestion ----------------------------------------------------------
    def ingest_documents(self, documents: list[RawDocument], persist: bool = True) -> int:
        """Chunk, embed, and index a list of documents. Returns chunks added."""
        chunks = self.chunker.chunk_documents(documents)
        return self._index_chunks(chunks, persist=persist)

    def _index_chunks(self, chunks: list[Chunk], persist: bool) -> int:
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        if persist:
            self.store.save(self.settings.paths.index_dir)
        logger.info("Indexed %d chunks (store total=%d)", len(chunks), self.store.count)
        return len(chunks)

    def reset_index(self) -> None:
        self.store.reset()
        self.store.save(self.settings.paths.index_dir)

    # -- question answering -------------------------------------------------
    def answer(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        flags: list[GuardrailFlag] = []

        # 1) Emergency screening on the raw question (independent of reports).
        emergency = self.guardrails.detect_emergency(question)
        if emergency:
            flags.append(emergency)

        # 2) Retrieve evidence.
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        confidence = self.scorer.score(retrieved)

        # 3) Refusal gate.
        if self.guardrails.should_refuse(retrieved, confidence):
            logger.info("Refusing: insufficient evidence (label=%s).", confidence.label.value)
            return Answer(
                question=question,
                answer_text=self.guardrails.refusal_message,
                citations=[],
                confidence=confidence,
                guardrail_flags=flags,
                refused=True,
                disclaimer=self.guardrails.disclaimer(),
            )

        # 4) Grounded generation.
        context = format_context(retrieved)
        messages = build_qa_messages(question, context)
        answer_text = self.llm.chat(messages)

        # 5) Post-generation safety + citations.
        flags.extend(self.guardrails.post_check(answer_text))
        refused = self.guardrails.refusal_message.lower() in answer_text.lower()
        citations = [] if refused else self._build_citations(answer_text, retrieved)

        return Answer(
            question=question,
            answer_text=answer_text,
            citations=citations,
            confidence=confidence,
            guardrail_flags=flags,
            refused=refused,
            disclaimer=self.guardrails.disclaimer(),
        )

    # -- citations ----------------------------------------------------------
    @staticmethod
    def _build_citations(answer_text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
        """Map [S#] tags in the answer back to their source chunks.

        Falls back to citing all retrieved chunks if the model omitted tags but
        still produced a substantive answer.
        """
        referenced = {int(n) for n in _CITATION_TAG.findall(answer_text)}
        chosen = (
            [retrieved[i - 1] for i in sorted(referenced) if 1 <= i <= len(retrieved)]
            if referenced
            else retrieved
        )
        citations: list[Citation] = []
        for rc in chosen:
            snippet = " ".join(rc.chunk.text.split())
            citations.append(
                Citation(
                    doc_id=rc.chunk.doc_id,
                    filename=rc.chunk.filename,
                    page_number=rc.chunk.page_number,
                    snippet=(snippet[:220] + "…") if len(snippet) > 220 else snippet,
                    score=rc.score,
                )
            )
        return citations

    # -- shared context builder (used by feature services) ------------------
    def retrieve_context(self, query: str, top_k: int | None = None) -> tuple[str, list[RetrievedChunk], Confidence]:
        """Retrieve and format context for downstream feature prompts."""
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        confidence = self.scorer.score(retrieved)
        return format_context(retrieved), retrieved, confidence
