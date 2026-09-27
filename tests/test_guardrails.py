"""Tests for safety guardrails and confidence scoring."""

from __future__ import annotations

from medintel.config import RetrievalConfig
from medintel.features.confidence import ConfidenceScorer
from medintel.features.guardrails import Guardrails
from medintel.models.schemas import (
    Chunk,
    Confidence,
    ConfidenceLabel,
    RetrievedChunk,
)


def _rc(score: float, doc_id: str = "doc_1") -> RetrievedChunk:
    chunk = Chunk(doc_id=doc_id, filename="f.txt", page_number=1, chunk_index=0, text="x")
    return RetrievedChunk(chunk=chunk, score=score)


def test_emergency_detection(settings):
    g = Guardrails(settings.guardrails)
    flag = g.detect_emergency("I have crushing chest pain and shortness of breath")
    assert flag is not None
    assert flag.code == "emergency"
    assert g.detect_emergency("what is my cholesterol level") is None


def test_refusal_when_no_evidence(settings):
    g = Guardrails(settings.guardrails)
    assert g.should_refuse([], Confidence(label=ConfidenceLabel.INSUFFICIENT)) is True


def test_post_check_flags_missing_citation(settings):
    g = Guardrails(settings.guardrails)
    flags = g.post_check("Your values look fine.")
    assert any(f.code == "missing_citation" for f in flags)
    assert g.post_check("Your ALT is elevated [S1].") == []


def test_confidence_scoring_labels():
    # Use the production threshold explicitly so this unit test is independent
    # of the relaxed offline-test configuration.
    scorer = ConfidenceScorer(RetrievalConfig(min_confidence=0.35))
    high = scorer.score([_rc(0.92), _rc(0.88, "doc_2"), _rc(0.81, "doc_3")])
    assert high.label is ConfidenceLabel.HIGH
    assert high.source_count == 3

    insufficient = scorer.score([_rc(0.10)])
    assert insufficient.label is ConfidenceLabel.INSUFFICIENT
    assert scorer.score([]).label is ConfidenceLabel.INSUFFICIENT
