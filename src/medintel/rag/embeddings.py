"""Embedding providers.

An embedder maps text → dense vector so that semantically similar clinical
phrasings ("elevated ALT" ≈ "high liver enzyme") land close together in vector
space. The interface is provider-agnostic:

* ``OpenAIEmbedder``      — production default (``text-embedding-3-small``)
* ``HuggingFaceEmbedder`` — local/offline (sentence-transformers)
* ``HashEmbedder``        — deterministic, dependency-free; used by tests/CI and
                            as a last-resort fallback so the app never hard-fails.

Vectors are returned **unnormalized**; the FAISS store owns L2 normalization so
there is exactly one place that defines the cosine convention.
"""

from __future__ import annotations

import abc
import hashlib

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from medintel.config import EmbeddingsConfig, Settings, get_settings
from medintel.exceptions import ConfigurationError, EmbeddingError
from medintel.logging_config import get_logger

logger = get_logger(__name__)


class BaseEmbedder(abc.ABC):
    """Abstract embedding provider."""

    dimension: int

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents → array of shape (len(texts), dimension)."""

    @abc.abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query → array of shape (dimension,)."""


# ---------------------------------------------------------------------------
class OpenAIEmbedder(BaseEmbedder):
    """Embeddings via the OpenAI API."""

    def __init__(self, config: EmbeddingsConfig, api_key: str | None) -> None:
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI embedder. "
                "Set it in .env or switch MEDINTEL_EMBEDDINGS__PROVIDER to 'huggingface'/'hash'."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("The 'openai' package is not installed.") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = config.model
        self._batch_size = config.batch_size
        self.dimension = config.dimension  # confirmed/overwritten on first call

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=batch)
        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + self._batch_size]]
            try:
                vectors.extend(self._embed_batch(batch))
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
        arr = np.asarray(vectors, dtype=np.float32)
        self.dimension = arr.shape[1]
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
class HuggingFaceEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers (no API key, fully offline)."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers not installed. `pip install -e \".[local-embeddings]\"`"
            ) from exc
        self._model = SentenceTransformer(config.hf_model)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._batch_size = config.batch_size
        logger.info("Loaded HF embedder '%s' (dim=%d)", config.hf_model, self.dimension)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
class HashEmbedder(BaseEmbedder):
    """Deterministic hashing embedder — no dependencies, no network.

    Uses the hashing trick: each token contributes to a fixed set of dimensions
    via a stable hash. Not semantically rich, but reproducible and good enough
    for tests, CI, and graceful degradation when no real embedder is configured.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


# ---------------------------------------------------------------------------
def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Factory selecting an embedder from configuration, with safe fallbacks."""
    settings = settings or get_settings()
    provider = settings.embeddings.provider.lower()

    if provider == "openai":
        try:
            return OpenAIEmbedder(settings.embeddings, settings.openai_api_key)
        except ConfigurationError as exc:
            logger.warning("%s Falling back to HashEmbedder.", exc)
            return HashEmbedder()
    if provider == "huggingface":
        return HuggingFaceEmbedder(settings.embeddings)
    if provider == "hash":
        return HashEmbedder()
    raise ConfigurationError(f"Unknown embeddings provider: {provider!r}")
