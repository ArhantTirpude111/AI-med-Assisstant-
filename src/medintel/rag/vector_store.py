"""FAISS-backed vector store with a metadata sidecar.

We use ``IndexFlatIP`` (inner product) over **L2-normalized** vectors, which makes
the inner product equal to cosine similarity. Flat (exact) search is the right
default for the document volumes a single patient produces; the interface is
deliberately small so the backend can later be swapped for an ANN index
(``IndexHNSWFlat``) or an external store (pgvector / Qdrant / Milvus) without
touching the retriever.

Persistence
-----------
* ``index.faiss`` — the serialized FAISS index
* ``chunks.json`` — parallel ``Chunk`` metadata (row *i* in the index ↔ chunk *i*)
* ``meta.json``   — dimension + embedder signature for compatibility checks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medintel.exceptions import VectorStoreError
from medintel.logging_config import get_logger
from medintel.models.schemas import Chunk, RetrievedChunk

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"
_META_FILE = "meta.json"


class FaissVectorStore:
    """Cosine-similarity vector store backed by FAISS."""

    def __init__(self, dimension: int, signature: str = "") -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("faiss-cpu is not installed.") from exc
        self._faiss = faiss
        self.dimension = dimension
        self.signature = signature
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    # -- properties ---------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    # -- mutation -----------------------------------------------------------
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their vectors. ``embeddings`` is (n, dimension)."""
        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreError("chunks/embeddings length mismatch.")
        if embeddings.shape[0] == 0:
            return
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dim {embeddings.shape[1]} != index dim {self.dimension}."
            )
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info("Vector store: added %d chunks (total=%d)", len(chunks), self.count)

    def reset(self) -> None:
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        logger.info("Vector store reset.")

    # -- search -------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Return the top-``k`` most similar chunks with cosine scores in [0, 1]."""
        if self.is_empty:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        self._faiss.normalize_L2(query)
        k = min(k, self.count)
        scores, indices = self.index.search(query, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            cosine = float(max(0.0, min(1.0, score)))  # clamp to [0, 1]
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=cosine))
        return results

    # -- persistence --------------------------------------------------------
    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        try:
            self._faiss.write_index(self.index, str(path / _INDEX_FILE))
            (path / _CHUNKS_FILE).write_text(
                json.dumps([c.model_dump() for c in self.chunks]), encoding="utf-8"
            )
            (path / _META_FILE).write_text(
                json.dumps({"dimension": self.dimension, "signature": self.signature}),
                encoding="utf-8",
            )
            logger.info("Vector store saved to %s (%d chunks)", path, self.count)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to save vector store: {exc}") from exc

    @classmethod
    def load(cls, index_dir: str | Path) -> FaissVectorStore:
        path = Path(index_dir)
        meta_path = path / _META_FILE
        if not (path / _INDEX_FILE).exists() or not meta_path.exists():
            raise VectorStoreError(f"No persisted index found in {path}.")
        try:
            import faiss

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            store = cls(dimension=int(meta["dimension"]), signature=meta.get("signature", ""))
            store.index = faiss.read_index(str(path / _INDEX_FILE))
            raw_chunks = json.loads((path / _CHUNKS_FILE).read_text(encoding="utf-8"))
            store.chunks = [Chunk.model_validate(c) for c in raw_chunks]
            logger.info("Vector store loaded from %s (%d chunks)", path, store.count)
            return store
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to load vector store: {exc}") from exc

    @staticmethod
    def exists(index_dir: str | Path) -> bool:
        path = Path(index_dir)
        return (path / _INDEX_FILE).exists() and (path / _META_FILE).exists()
