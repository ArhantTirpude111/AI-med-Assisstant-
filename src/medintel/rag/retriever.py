"""Semantic retriever.

Bridges the embedder and the vector store: embeds the query, fetches a wider
candidate pool (``fetch_k``), removes near-duplicate chunks, and returns the
best ``top_k``. Keeping retrieval separate from generation means we can unit-test
recall independently of the LLM and later insert a re-ranker (cross-encoder)
without touching the pipeline.
"""

from __future__ import annotations

from medintel.config import RetrievalConfig, get_settings
from medintel.logging_config import get_logger
from medintel.models.schemas import RetrievedChunk
from medintel.rag.embeddings import BaseEmbedder
from medintel.rag.vector_store import FaissVectorStore

logger = get_logger(__name__)


class Retriever:
    """Embed a query and fetch the most relevant chunks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: FaissVectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or get_settings().retrieval

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to ``top_k`` deduplicated, score-sorted chunks."""
        if self.store.is_empty:
            logger.warning("Retrieval requested but the vector store is empty.")
            return []
        top_k = top_k or self.config.top_k
        query_vec = self.embedder.embed_query(query)
        candidates = self.store.search(query_vec, k=self.config.fetch_k)
        deduped = self._dedupe(candidates) if self.config.dedupe else candidates
        results = deduped[:top_k]
        logger.info(
            "Retrieved %d/%d candidates for query (top score=%.3f)",
            len(results), len(candidates), results[0].score if results else 0.0,
        )
        return results

    @staticmethod
    def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop chunks whose normalized text we've already seen (keep best score)."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for rc in chunks:  # candidates arrive in descending score order
            key = " ".join(rc.chunk.text.lower().split())[:200]
            if key in seen:
                continue
            seen.add(key)
            unique.append(rc)
        return unique
