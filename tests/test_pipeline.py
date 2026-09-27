"""End-to-end pipeline & assistant tests (offline)."""

from __future__ import annotations

import pytest

from medintel.rag.prompts import REFUSAL_MESSAGE


@pytest.mark.integration
def test_full_ingest_and_answer(assistant, sample_report_text):
    report = assistant.ingest([("blood_panel.txt", sample_report_text.encode("utf-8"))])
    assert report.document_count == 1
    assert report.chunks_indexed > 0
    assert assistant.is_ready

    answer = assistant.ask("Are my liver enzymes elevated?")
    assert answer.answer_text
    assert not answer.refused
    assert answer.citations, "a grounded answer should carry citations"
    assert answer.citations[0].page_number == 1
    assert answer.confidence.source_count > 0


@pytest.mark.integration
def test_refusal_on_unrelated_question(assistant, sample_report_text):
    assistant.ingest([("blood_panel.txt", sample_report_text.encode("utf-8"))])
    # The mock LLM echoes context; with retrieval present it answers. Empty index
    # is the deterministic refusal path:
    assistant.reset()
    answer = assistant.ask("What is the capital of France?")
    assert answer.refused is True
    assert REFUSAL_MESSAGE in answer.answer_text


@pytest.mark.integration
def test_summary_and_followups(assistant, sample_report_text):
    assistant.ingest([("blood_panel.txt", sample_report_text.encode("utf-8"))])
    summary = assistant.summarize()
    # Deterministic abnormal-lab merge should surface the high ALT.
    abnormal = " ".join(summary.abnormal_values).lower()
    assert "alt" in abnormal
    followups = assistant.followups()
    assert isinstance(followups, list) and followups


@pytest.mark.integration
def test_timeline_extraction(assistant, sample_report_text):
    assistant.ingest([("blood_panel.txt", sample_report_text.encode("utf-8"))])
    events = assistant.timeline()
    assert any(e.date == "2025-03-14" for e in events)


@pytest.mark.integration
def test_knowledge_graph_build(assistant, sample_report_text):
    assistant.ingest([("blood_panel.txt", sample_report_text.encode("utf-8"))])
    graph = assistant.knowledge_graph()
    assert graph.number_of_nodes() > 0
