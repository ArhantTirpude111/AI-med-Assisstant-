"""Follow-up question generator.

Produces specific, grounded questions the patient can bring to their doctor —
e.g. "Should my elevated ALT be rechecked?" — derived from the report context
rather than generic advice.
"""

from __future__ import annotations

from medintel.features._base import FeatureService
from medintel.logging_config import get_logger
from medintel.rag.prompts import build_followup_messages
from medintel.utils import extract_json

logger = get_logger(__name__)

_QUERY = "abnormal results, diagnoses, and recommendations to discuss with a doctor"

_FALLBACK = [
    "Are any of my results outside the normal range, and what does that mean?",
    "Do any of these findings need to be rechecked or monitored?",
    "Are further tests or investigations recommended?",
    "What lifestyle changes might improve these results?",
    "When should I schedule a follow-up appointment?",
]


class FollowupService(FeatureService):
    def generate(self, top_k: int = 8, limit: int = 7) -> list[str]:
        if not self.pipeline.is_ready:
            return _FALLBACK[:limit]
        context, retrieved, _ = self.pipeline.retrieve_context(_QUERY, top_k=top_k)
        if not retrieved:
            return _FALLBACK[:limit]

        raw = self.llm.chat(build_followup_messages(context))
        data = extract_json(raw, default=[]) or []
        questions = [str(q).strip() for q in data if str(q).strip()] if isinstance(data, list) else []
        return (questions or _FALLBACK)[:limit]
