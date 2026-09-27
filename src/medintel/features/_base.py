"""Common base for feature services.

Feature services are thin, focused capabilities (summary, terminology,
differential, timeline, follow-ups) that compose the shared ``RAGPipeline`` —
reusing its embedder, vector store, retriever, and LLM client rather than
re-instantiating heavy resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from medintel.llm.client import BaseLLMClient
    from medintel.rag.pipeline import RAGPipeline


class FeatureService:
    """Base class holding a reference to the shared pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    @property
    def llm(self) -> BaseLLMClient:
        return self.pipeline.llm
