"""Domain-specific exception hierarchy.

A single base (`MedIntelError`) lets callers catch *all* application errors while
still allowing granular handling. Each layer raises a specific subclass so the
UI can present an actionable message instead of a stack trace.
"""

from __future__ import annotations


class MedIntelError(Exception):
    """Base class for all MedIntel application errors."""


# --- Ingestion -------------------------------------------------------------
class IngestionError(MedIntelError):
    """Raised when a document cannot be loaded or parsed."""


class UnsupportedFileTypeError(IngestionError):
    """Raised for file extensions the system does not know how to handle."""


class OCRError(IngestionError):
    """Raised when OCR is required but unavailable or fails."""


# --- Processing ------------------------------------------------------------
class ProcessingError(MedIntelError):
    """Raised during chunking / metadata extraction."""


# --- RAG / retrieval -------------------------------------------------------
class EmbeddingError(MedIntelError):
    """Raised when embedding generation fails."""


class VectorStoreError(MedIntelError):
    """Raised on vector index read/write/search failures."""


class RetrievalError(MedIntelError):
    """Raised when retrieval cannot be completed."""


# --- LLM -------------------------------------------------------------------
class LLMError(MedIntelError):
    """Raised when the language model call fails."""


class ConfigurationError(MedIntelError):
    """Raised when configuration is missing or invalid (e.g. no API key)."""
