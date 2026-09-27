"""Report summary generation.

Combines two signals:

* **Deterministic** structured extraction (abnormal labs) — auditable, never
  hallucinated.
* **LLM synthesis** over retrieved context — executive summary, key findings,
  risk indicators, follow-up questions — strictly grounded and JSON-structured.

The two are merged so the abnormal-values list is anchored in real parsed numbers.
"""

from __future__ import annotations

from medintel.features._base import FeatureService
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    Citation,
    LabFlag,
    MedicalEntities,
    ReportSummary,
)
from medintel.rag.prompts import build_summary_messages
from medintel.utils import extract_json

logger = get_logger(__name__)

_SUMMARY_QUERY = (
    "summary of diagnoses, key findings, abnormal laboratory values, "
    "risk factors, and recommended follow-up"
)


class SummaryService(FeatureService):
    def generate(self, entities: MedicalEntities | None = None, top_k: int = 12) -> ReportSummary:
        context, retrieved, _ = self.pipeline.retrieve_context(_SUMMARY_QUERY, top_k=top_k)
        if not retrieved:
            return ReportSummary(
                executive_summary="No documents have been indexed yet. Upload a report to generate a summary."
            )

        raw = self.pipeline.llm.chat(build_summary_messages(context))
        data = extract_json(raw, default={}) or {}

        summary = ReportSummary(
            executive_summary=str(data.get("executive_summary", "")).strip(),
            key_findings=self._as_list(data.get("key_findings")),
            abnormal_values=self._as_list(data.get("abnormal_values")),
            risk_indicators=self._as_list(data.get("risk_indicators")),
            followup_questions=self._as_list(data.get("followup_questions")),
            citations=self._citations(retrieved),
        )
        # Anchor abnormal values in deterministically-parsed labs.
        if entities:
            summary.abnormal_values = self._merge_abnormal_labs(summary.abnormal_values, entities)
        return summary

    @staticmethod
    def _as_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _merge_abnormal_labs(existing: list[str], entities: MedicalEntities) -> list[str]:
        deterministic: list[str] = []
        for lab in entities.lab_results:
            if lab.flag in (LabFlag.HIGH, LabFlag.LOW, LabFlag.ABNORMAL):
                unit = f" {lab.unit}" if lab.unit else ""
                ref = f" (ref {lab.reference_range})" if lab.reference_range else ""
                deterministic.append(
                    f"{lab.name}: {lab.value}{unit} — {lab.flag.value.upper()}{ref}"
                )
        # De-duplicate while preserving order; deterministic facts first.
        seen: set[str] = set()
        merged: list[str] = []
        for item in deterministic + existing:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    @staticmethod
    def _citations(retrieved) -> list[Citation]:  # noqa: ANN001
        return [
            Citation(
                doc_id=rc.chunk.doc_id,
                filename=rc.chunk.filename,
                page_number=rc.chunk.page_number,
                snippet=" ".join(rc.chunk.text.split())[:180],
                score=rc.score,
            )
            for rc in retrieved[:5]
        ]
