"""Language-model client abstraction.

Two implementations behind one interface:

* ``OpenAILLMClient`` — production chat completions with retry/backoff.
* ``MockLLMClient``   — deterministic, offline; inspects the prompt and returns
  *grounded-looking*, schema-valid responses. This keeps the entire app (and the
  test suite) runnable with no API key, which is essential for CI and demos.

Choosing the client is configuration-driven via :func:`get_llm_client`.
"""

from __future__ import annotations

import abc
import json
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from medintel.config import LLMConfig, Settings, get_settings
from medintel.exceptions import ConfigurationError, LLMError
from medintel.logging_config import get_logger

logger = get_logger(__name__)

Messages = list[dict[str, str]]


class BaseLLMClient(abc.ABC):
    provider: str

    @abc.abstractmethod
    def chat(self, messages: Messages) -> str:
        """Return the assistant message content for a chat completion."""

    def complete(self, system: str, user: str) -> str:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])


# ---------------------------------------------------------------------------
class OpenAILLMClient(BaseLLMClient):
    """OpenAI Chat Completions — also works with any OpenAI-compatible host
    (Groq, OpenRouter, Together, local servers) via ``llm.base_url``."""

    provider = "openai"

    def __init__(self, config: LLMConfig, api_key: str | None) -> None:
        if not api_key:
            raise ConfigurationError(
                "An API key is required for the OpenAI-compatible LLM client. "
                "Set OPENAI_API_KEY in .env, or use a free provider "
                "(MEDINTEL_LLM__PROVIDER=local for a zero-cost offline model, "
                "or point MEDINTEL_LLM__BASE_URL at a free host like Groq)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed.") from exc
        # base_url unlocks free OpenAI-compatible hosts without code changes.
        self._client = OpenAI(
            api_key=api_key, timeout=config.timeout_seconds, base_url=config.base_url
        )
        self._config = config

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _call(self, messages: Messages) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat(self, messages: Messages) -> str:
        try:
            return self._call(messages)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI chat completion failed: {exc}") from exc


# ---------------------------------------------------------------------------
class LocalHFLLMClient(BaseLLMClient):
    """Free, offline LLM via a local HuggingFace causal model (transformers).

    Runs entirely on the user's machine — no API key, no cost, no network at
    inference time (the model weights download once, then are cached). Defaults
    to a small instruction-tuned model (`Qwen2.5-0.5B-Instruct`) that is usable
    on CPU; configure a larger one via ``llm.local_model`` for better quality.
    """

    provider = "local"

    def __init__(self, config: LLMConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "transformers/torch are required for the local LLM. "
                'Install with: pip install -e ".[local-llm]"'
            ) from exc

        self._torch = torch
        self._config = config
        model_id = config.local_model
        logger.info("Loading local LLM '%s' (first run downloads weights)…", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        logger.info("Local LLM ready: %s", model_id)

    def chat(self, messages: Messages) -> str:
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
            sample = self._config.temperature and self._config.temperature > 0
            with self._torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_tokens,
                    do_sample=bool(sample),
                    temperature=self._config.temperature if sample else None,
                    top_p=0.9 if sample else None,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Local LLM generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
class MockLLMClient(BaseLLMClient):
    """Offline deterministic client. Produces grounded, schema-valid output."""

    provider = "mock"

    def chat(self, messages: Messages) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if "STRICT JSON" in system and "executive_summary" in system:
            return self._mock_summary(user)
        if "terminology simplifier" in system:
            return self._mock_terminology(user)
        if "differential" in system.lower():
            return self._mock_differential(user)
        if "questions the patient could ask" in system:
            return self._mock_followups(user)
        return self._mock_grounded_answer(user)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _context_blocks(user: str) -> list[tuple[str, str]]:
        """Return [(tag, text)] parsed from a [S#] formatted context."""
        blocks: list[tuple[str, str]] = []
        for match in re.finditer(r"\[S(\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\Z)", user, re.DOTALL):
            blocks.append((f"S{match.group(1)}", match.group(2).strip()))
        return blocks

    def _mock_grounded_answer(self, user: str) -> str:
        if "(no relevant context found)" in user:
            return "I cannot find evidence for that in the uploaded reports."
        blocks = self._context_blocks(user)
        if not blocks:
            return "I cannot find evidence for that in the uploaded reports."
        tag, text = blocks[0]
        snippet = " ".join(text.split())[:240]
        return (
            f"Based on your uploaded report, the relevant information states: "
            f"\"{snippet}\" [{tag}]. In plain language, this is what your report "
            f"records about your question. Please review these details with your "
            f"doctor for personalized interpretation."
        )

    def _mock_summary(self, user: str) -> str:
        blocks = self._context_blocks(user)
        findings = [f"{' '.join(t.split())[:120]} [{tag}]" for tag, t in blocks[:3]]
        return json.dumps(
            {
                "executive_summary": (
                    "This is an automated offline summary of the uploaded report(s). "
                    "Key extracted statements are listed below."
                    if findings else ""
                ),
                "key_findings": findings,
                "abnormal_values": [f for f in findings if re.search(r"\d", f)][:3],
                "risk_indicators": [],
                "followup_questions": [
                    "Should any abnormal results be rechecked?",
                    "Do these findings require further testing?",
                ],
            }
        )

    def _mock_terminology(self, user: str) -> str:
        term = "the term"
        m = re.search(r"TERM:\s*(.+)", user)
        if m:
            term = m.group(1).splitlines()[0].strip()
        return json.dumps(
            {
                "beginner": f"'{term}' is a medical term describing a clinical finding.",
                "intermediate": f"'{term}' refers to a specific condition or measurement clinicians track.",
                "patient_friendly": f"In everyday words, '{term}' is something your doctor noted; ask them what it means for you.",
                "eli5": f"'{term}' is like a label doctors use to describe how a part of your body is doing.",
            }
        )

    def _mock_differential(self, user: str) -> str:
        blocks = self._context_blocks(user)
        if not blocks:
            return "[]"
        return json.dumps(
            [
                {
                    "condition": "Finding requiring clinical correlation",
                    "rationale": f"Derived from extracted findings [{blocks[0][0]}] (offline mock).",
                    "supporting_findings": [" ".join(blocks[0][1].split())[:100]],
                }
            ]
        )

    def _mock_followups(self, user: str) -> str:
        return json.dumps(
            [
                "Are any of my results outside the normal range?",
                "Do I need to repeat any of these tests?",
                "Are further investigations recommended?",
                "What lifestyle changes might help my results?",
                "When should I schedule a follow-up?",
            ]
        )


# ---------------------------------------------------------------------------
def get_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Factory selecting an LLM client from configuration, with safe fallback."""
    settings = settings or get_settings()
    provider = settings.llm.provider.lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "local":
        try:
            return LocalHFLLMClient(settings.llm)
        except LLMError as exc:
            logger.warning("%s Falling back to MockLLMClient.", exc)
            return MockLLMClient()
    if provider == "openai":
        try:
            return OpenAILLMClient(settings.llm, settings.openai_api_key)
        except ConfigurationError as exc:
            logger.warning("%s Falling back to MockLLMClient.", exc)
            return MockLLMClient()
    raise ConfigurationError(f"Unknown LLM provider: {provider!r}")
