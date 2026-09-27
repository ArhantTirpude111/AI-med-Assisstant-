"""MedIntel AI — Retrieval-Augmented Medical Report Analysis Assistant.

A production-grade RAG system for grounded, citation-backed medical report Q&A.

Public entry points::

    from medintel import get_settings, RAGPipeline, DocumentService
"""

from __future__ import annotations

from medintel.config import Settings, get_settings
from medintel.logging_config import configure_logging, get_logger

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "configure_logging",
    "get_logger",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export layer
    """Lazily expose heavy services to keep ``import medintel`` cheap."""
    if name == "RAGPipeline":
        from medintel.rag.pipeline import RAGPipeline

        return RAGPipeline
    if name == "DocumentService":
        from medintel.ingestion.document_service import DocumentService

        return DocumentService
    raise AttributeError(f"module 'medintel' has no attribute {name!r}")
