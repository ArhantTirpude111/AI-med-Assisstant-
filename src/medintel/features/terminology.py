"""Medical terminology simplifier.

Turns a clinical term (e.g. "Hepatic Steatosis") into four explanation levels:
beginner, intermediate, patient-friendly, and ELI5. Optionally grounds the
explanation in the user's report context so the wording matches their situation
without ever asserting a diagnosis.
"""

from __future__ import annotations

from medintel.features._base import FeatureService
from medintel.logging_config import get_logger
from medintel.models.schemas import ExplanationLevel, TerminologySimplification
from medintel.rag.prompts import build_terminology_messages
from medintel.utils import extract_json

logger = get_logger(__name__)


class TerminologyService(FeatureService):
    def simplify(self, term: str, use_context: bool = True) -> TerminologySimplification:
        term = term.strip()
        if not term:
            return TerminologySimplification(term=term)

        context = None
        if use_context and self.pipeline.is_ready:
            ctx, retrieved, _ = self.pipeline.retrieve_context(term, top_k=3)
            context = ctx if retrieved else None

        raw = self.llm.chat(build_terminology_messages(term, context))
        data = extract_json(raw, default={}) or {}
        return TerminologySimplification(
            term=term,
            beginner=str(data.get("beginner", "")).strip(),
            intermediate=str(data.get("intermediate", "")).strip(),
            patient_friendly=str(data.get("patient_friendly", "")).strip(),
            eli5=str(data.get("eli5", "")).strip(),
        )

    def explain_at_level(self, term: str, level: ExplanationLevel) -> str:
        result = self.simplify(term)
        return {
            ExplanationLevel.BEGINNER: result.beginner,
            ExplanationLevel.INTERMEDIATE: result.intermediate,
            ExplanationLevel.PATIENT_FRIENDLY: result.patient_friendly,
            ExplanationLevel.ELI5: result.eli5,
        }[level]
