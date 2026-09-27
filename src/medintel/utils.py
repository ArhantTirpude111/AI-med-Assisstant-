"""Small shared utilities."""

from __future__ import annotations

import json
import re
from typing import Any

from medintel.logging_config import get_logger

logger = get_logger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str, default: Any = None) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles code fences and surrounding prose. Returns ``default`` if nothing
    parseable is found (rather than raising) so feature services degrade
    gracefully instead of crashing the UI.
    """
    if not text:
        return default
    candidate = text.strip()

    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    # Fast path: the whole thing is JSON.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.debug("Could not extract JSON from LLM output.")
    return default


def truncate(text: str, limit: int, ellipsis: str = "…") -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - len(ellipsis)] + ellipsis
