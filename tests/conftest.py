"""Shared pytest fixtures.

All tests run **offline**: the LLM provider is forced to ``mock`` and embeddings
to ``hash``, so no API keys or model downloads are required. Each test gets an
isolated temp data directory and SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medintel.config import reload_settings
from medintel.models.schemas import ExtractionMethod, Page, RawDocument, SourceType


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDINTEL_LLM__PROVIDER", "mock")
    monkeypatch.setenv("MEDINTEL_EMBEDDINGS__PROVIDER", "hash")
    monkeypatch.setenv("MEDINTEL_NER__BACKEND", "rules")
    # The offline HashEmbedder is lexical, not semantic, so cosine scores are low.
    # Real (OpenAI/HF) embeddings make the production 0.35 gate meaningful; for the
    # offline suite we relax it so the grounded-answer path is exercised.
    monkeypatch.setenv("MEDINTEL_RETRIEVAL__MIN_CONFIDENCE", "0.05")
    monkeypatch.setenv("MEDINTEL_PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEDINTEL_PATHS__UPLOAD_DIR", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("MEDINTEL_PATHS__INDEX_DIR", str(tmp_path / "data" / "index"))
    monkeypatch.setenv("MEDINTEL_PATHS__DB_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return reload_settings()


@pytest.fixture()
def sample_report_text() -> str:
    return (
        "LABORATORY REPORT\n"
        "Patient seen on 2025-03-14.\n\n"
        "Diagnosis: Hepatic steatosis; Type 2 diabetes\n\n"
        "Lab Results:\n"
        "ALT 78 U/L (7-56) H\n"
        "AST 65 U/L (10-40) H\n"
        "Hemoglobin 11.2 g/dL (13.5-17.5) L\n"
        "Glucose 145 mg/dL (70-99) H\n"
        "Cholesterol 180 mg/dL (0-200)\n\n"
        "Medications:\n"
        "Metformin 500mg twice daily\n"
        "Atorvastatin 20mg nightly\n\n"
        "Symptoms: fatigue, abdominal pain\n\n"
        "Recommendations:\n"
        "Repeat liver function tests in 3 months.\n"
        "Maintain a low-fat diet.\n"
    )


@pytest.fixture()
def sample_document(sample_report_text: str) -> RawDocument:
    return RawDocument(
        filename="blood_panel.txt",
        source_type=SourceType.TXT,
        pages=[Page(page_number=1, text=sample_report_text, extraction_method=ExtractionMethod.NATIVE)],
    )


@pytest.fixture()
def assistant(settings):  # noqa: ANN001
    from medintel.assistant import MedIntelAssistant

    return MedIntelAssistant(settings)
