"""Educational differential-reasoning assistant.

Given findings extracted from the patient's reports, suggests conditions that
*could* be associated with such findings — strictly for learning, never as a
diagnosis. Every result carries an explicit educational disclaimer, and the
service refuses to speculate when evidence is thin.
"""

from __future__ import annotations

from medintel.features._base import FeatureService
from medintel.logging_config import get_logger
from medintel.models.schemas import DifferentialDiagnosis, MedicalEntities
from medintel.rag.prompts import build_differential_messages
from medintel.utils import extract_json

logger = get_logger(__name__)

_FINDINGS_QUERY = "abnormal findings, diagnoses, symptoms and out-of-range laboratory values"
_DISCLAIMER = "For educational purposes only — not a diagnosis. Consult a clinician."


class DifferentialService(FeatureService):
    def suggest(self, entities: MedicalEntities | None = None, top_k: int = 10) -> list[DifferentialDiagnosis]:
        context, retrieved, _ = self.pipeline.retrieve_context(_FINDINGS_QUERY, top_k=top_k)
        if not retrieved:
            return []

        # Enrich the context with deterministically-extracted findings.
        if entities and not entities.is_empty:
            context = self._entity_brief(entities) + "\n\n" + context

        raw = self.llm.chat(build_differential_messages(context))
        data = extract_json(raw, default=[]) or []
        if not isinstance(data, list):
            return []

        results: list[DifferentialDiagnosis] = []
        for item in data:
            if not isinstance(item, dict) or not item.get("condition"):
                continue
            results.append(
                DifferentialDiagnosis(
                    condition=str(item["condition"]).strip(),
                    rationale=str(item.get("rationale", "")).strip(),
                    supporting_findings=[
                        str(f).strip() for f in item.get("supporting_findings", []) if str(f).strip()
                    ],
                    disclaimer=_DISCLAIMER,
                )
            )
        return results

    @staticmethod
    def _entity_brief(entities: MedicalEntities) -> str:
        lines = ["EXTRACTED STRUCTURED FINDINGS:"]
        if entities.diagnoses:
            lines.append("Diagnoses: " + ", ".join(e.text for e in entities.diagnoses[:10]))
        if entities.symptoms:
            lines.append("Symptoms: " + ", ".join(e.text for e in entities.symptoms[:10]))
        abnormal = [
            f"{lab.name}={lab.value} ({lab.flag.value})"
            for lab in entities.lab_results
            if lab.flag.value in {"high", "low", "abnormal"}
        ]
        if abnormal:
            lines.append("Abnormal labs: " + ", ".join(abnormal[:12]))
        return "\n".join(lines)
