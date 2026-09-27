"""Safety guardrails.

Layered safeguards applied around the RAG core:

* **Pre-retrieval** — scan the user's question for emergency red-flags and raise
  an urgent-care notice (independent of what the reports contain).
* **Gating** — block answer generation when retrieval confidence is below the
  configured threshold (returns the canonical refusal).
* **Post-generation** — detect when the model produced an answer without any
  citation (possible ungrounded output) and attach the educational disclaimer.

Guardrails are intentionally conservative: in a medical context, a false "please
seek care" is far cheaper than a missed emergency.
"""

from __future__ import annotations

import re

from medintel.config import GuardrailsConfig, get_settings
from medintel.logging_config import get_logger
from medintel.models.schemas import (
    Confidence,
    ConfidenceLabel,
    GuardrailFlag,
    RetrievedChunk,
)
from medintel.processing.vocab import EMERGENCY_TERMS
from medintel.rag.prompts import DISCLAIMER, REFUSAL_MESSAGE

logger = get_logger(__name__)

_EMERGENCY_BANNER = (
    "🚨 **Your message mentions symptoms that can signal a medical emergency.** "
    "If you are experiencing a medical emergency, call your local emergency number "
    "or go to the nearest emergency department immediately. This tool cannot help "
    "in an emergency."
)


class Guardrails:
    def __init__(self, config: GuardrailsConfig | None = None) -> None:
        self.config = config or get_settings().guardrails
        self._emergency_patterns = [
            re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in EMERGENCY_TERMS
        ]

    # -- pre-retrieval ------------------------------------------------------
    def detect_emergency(self, text: str) -> GuardrailFlag | None:
        """Return an emergency flag if the text contains red-flag terms."""
        if not self.config.detect_emergencies:
            return None
        for pattern in self._emergency_patterns:
            if pattern.search(text):
                logger.warning("Emergency red-flag detected in input.")
                return GuardrailFlag(
                    code="emergency",
                    message=_EMERGENCY_BANNER,
                    severity="critical",
                )
        return None

    # -- gating -------------------------------------------------------------
    def should_refuse(
        self, retrieved: list[RetrievedChunk], confidence: Confidence
    ) -> bool:
        """Decide whether to refuse instead of answering."""
        if not retrieved:
            return True
        return (
            self.config.refuse_below_confidence
            and confidence.label is ConfidenceLabel.INSUFFICIENT
        )

    @property
    def refusal_message(self) -> str:
        return REFUSAL_MESSAGE

    # -- post-generation ----------------------------------------------------
    def post_check(self, answer_text: str) -> list[GuardrailFlag]:
        """Inspect a generated answer for ungrounded-output signals."""
        flags: list[GuardrailFlag] = []
        is_refusal = REFUSAL_MESSAGE.lower() in answer_text.lower()
        has_citation = bool(re.search(r"\[S\d+\]", answer_text))
        if not is_refusal and not has_citation:
            flags.append(
                GuardrailFlag(
                    code="missing_citation",
                    message="Answer did not cite a source; treat with extra caution.",
                    severity="warning",
                )
            )
        return flags

    def disclaimer(self) -> str:
        return DISCLAIMER if self.config.attach_disclaimer else ""
