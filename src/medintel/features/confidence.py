"""Confidence scoring for retrieved evidence.

We surface *why* an answer should (or shouldn't) be trusted, derived purely from
retrieval signals — not the model's self-reported certainty (which is unreliable).
The score blends the best match, the average of the top matches, and how many
independent sources corroborate the answer.
"""

from __future__ import annotations

from medintel.config import RetrievalConfig, get_settings
from medintel.models.schemas import Confidence, ConfidenceLabel, RetrievedChunk

# Tunable weights for the blended retrieval-confidence score.
_W_TOP = 0.65      # the single best match dominates
_W_MEAN = 0.25     # corroboration across the top matches
_W_COUNT = 0.10    # diversity of supporting sources

_HIGH = 0.72
_MEDIUM = 0.52


class ConfidenceScorer:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or get_settings().retrieval

    def score(self, retrieved: list[RetrievedChunk]) -> Confidence:
        if not retrieved:
            return Confidence(label=ConfidenceLabel.INSUFFICIENT)

        scores = [rc.score for rc in retrieved]
        top = max(scores)
        mean = sum(scores) / len(scores)
        # Saturating bonus for having multiple distinct supporting sources.
        distinct_docs = len({rc.chunk.doc_id for rc in retrieved})
        count_signal = min(1.0, distinct_docs / 3.0)

        confidence = _W_TOP * top + _W_MEAN * mean + _W_COUNT * count_signal
        confidence = round(min(1.0, confidence), 4)

        return Confidence(
            retrieval_confidence=confidence,
            top_similarity=round(top, 4),
            mean_similarity=round(mean, 4),
            source_count=len(retrieved),
            label=self._label(confidence),
        )

    def _label(self, confidence: float) -> ConfidenceLabel:
        if confidence < self.config.min_confidence:
            return ConfidenceLabel.INSUFFICIENT
        if confidence >= _HIGH:
            return ConfidenceLabel.HIGH
        if confidence >= _MEDIUM:
            return ConfidenceLabel.MEDIUM
        return ConfidenceLabel.LOW
