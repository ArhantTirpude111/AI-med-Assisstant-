"""Tests for chunking, metadata extraction, and NER."""

from __future__ import annotations

from medintel.models.schemas import EntityType, LabFlag
from medintel.processing.chunking import Chunker
from medintel.processing.entity_recognition import MedicalNER
from medintel.processing.metadata_extractor import MedicalMetadataExtractor


def test_chunking_preserves_page_and_offsets(settings, sample_document):
    chunks = Chunker(settings=settings).chunk_document(sample_document)
    assert chunks, "expected at least one chunk"
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.doc_id == sample_document.doc_id for c in chunks)
    assert all(c.char_end >= c.char_start for c in chunks)
    # chunk_index is sequential and unique
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_metadata_extractor_parses_labs(sample_document):
    entities = MedicalMetadataExtractor().extract(sample_document)
    names = {lab.name.lower() for lab in entities.lab_results}
    assert "alt" in names
    alt = next(lab for lab in entities.lab_results if lab.name.lower() == "alt")
    assert alt.value == "78"
    assert alt.flag is LabFlag.HIGH
    # Hemoglobin below range should be flagged LOW
    hgb = next(lab for lab in entities.lab_results if "hemoglobin" in lab.name.lower())
    assert hgb.flag is LabFlag.LOW


def test_metadata_extractor_sections(sample_document):
    entities = MedicalMetadataExtractor().extract(sample_document)
    dx_text = " ".join(e.text.lower() for e in entities.diagnoses)
    assert "steatosis" in dx_text or "diabetes" in dx_text
    meds = " ".join(e.text.lower() for e in entities.medications)
    assert "metformin" in meds


def test_rule_based_ner(settings, sample_document):
    ner = MedicalNER(settings.ner)
    assert ner.backend == "rules"
    entities = ner.extract(sample_document)
    diag = {e.text.lower() for e in entities.diagnoses}
    assert any("diabetes" in d or "steatosis" in d for d in diag)
    assert any(e.entity_type is EntityType.MEDICATION for e in entities.medications)
