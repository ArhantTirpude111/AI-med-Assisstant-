"""Prompt templates and context formatting.

The system prompts are the primary *behavioral* guardrail: they constrain the
model to answer strictly from retrieved evidence, to cite sources, and to refuse
when the evidence is insufficient. Keeping them centralized makes the safety
posture auditable and testable.
"""

from __future__ import annotations

from medintel.models.schemas import RetrievedChunk

# The canonical refusal string (asserted in tests; surfaced to the user verbatim).
REFUSAL_MESSAGE = "I cannot find evidence for that in the uploaded reports."

DISCLAIMER = (
    "⚕️ This is an AI-generated explanation for educational purposes only and is "
    "not a medical diagnosis. Always consult a qualified healthcare professional."
)

# ---------------------------------------------------------------------------
GROUNDED_SYSTEM_PROMPT = f"""You are MedIntel, a careful medical-report explanation assistant.

You help patients understand their OWN uploaded medical reports. You must follow
these rules without exception:

1. GROUNDING: Answer ONLY using the information in the provided CONTEXT, which is
   extracted from the user's uploaded reports. Do NOT use outside medical
   knowledge to introduce facts that are not supported by the context.
2. CITATIONS: After each claim, reference the source using the bracket tag shown
   in the context, e.g. [S1], [S2]. Every factual statement must be traceable.
3. REFUSAL: If the context does not contain the information needed to answer,
   respond with EXACTLY this sentence and nothing else:
   "{REFUSAL_MESSAGE}"
4. NO DIAGNOSIS: Do not diagnose, prescribe, or recommend specific treatments.
   You may explain what the report says and what terms mean.
5. CLARITY: Write in clear, compassionate, plain language a non-medical person
   can understand. Define jargon briefly.
6. UNCERTAINTY: If values are borderline or context is partial, say so plainly.

Never fabricate lab values, page numbers, diagnoses, or medications."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered, citation-tagged context block."""
    if not chunks:
        return "(no relevant context found)"
    blocks: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        header = f"[S{i}] (file: {c.filename}, page: {c.page_number}, relevance: {rc.score:.2f})"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n".join(blocks)


def build_qa_messages(question: str, context: str) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded Q&A turn."""
    user = (
        f"CONTEXT (from the user's uploaded reports):\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above, citing sources as [S1], [S2], etc. "
        f"If the context is insufficient, reply exactly: \"{REFUSAL_MESSAGE}\""
    )
    return [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Feature prompts
# ---------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """You are MedIntel, summarizing a patient's medical reports.
Use ONLY the provided context. Produce a STRICT JSON object with keys:
  "executive_summary": string (2-4 sentences, plain language),
  "key_findings": string[] (most important findings, each citing [S#]),
  "abnormal_values": string[] (lab/imaging values outside reference range, each citing [S#]),
  "risk_indicators": string[] (potential concerns to monitor, each citing [S#]),
  "followup_questions": string[] (questions the patient could ask their doctor).
If the context lacks information for a field, return an empty array (or "" for
executive_summary). Do not invent values. Output JSON only — no prose."""


def build_summary_messages(context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nReturn the JSON summary."},
    ]


TERMINOLOGY_SYSTEM_PROMPT = """You are MedIntel, a medical terminology simplifier.
Given a medical term, explain it at four levels of complexity. Return STRICT JSON:
  "beginner": one clear sentence for a layperson,
  "intermediate": 2-3 sentences with slightly more clinical context,
  "patient_friendly": empathetic explanation of what it may mean for the patient
     (general, NOT a diagnosis),
  "eli5": an explanation a five-year-old could understand, using a simple analogy.
Output JSON only. Never imply the term is or isn't present in any specific report."""


def build_terminology_messages(term: str, context: str | None = None) -> list[dict[str, str]]:
    ctx = f"\n\nThe term appeared in this report context:\n{context}" if context else ""
    return [
        {"role": "system", "content": TERMINOLOGY_SYSTEM_PROMPT},
        {"role": "user", "content": f"TERM: {term}{ctx}\n\nReturn the JSON explanation."},
    ]


DIFFERENTIAL_SYSTEM_PROMPT = """You are MedIntel, an EDUCATIONAL differential-reasoning
assistant. Given findings extracted from a patient's reports, list possible
conditions that COULD be associated with such findings, for learning purposes only.
Return STRICT JSON: an array of objects with keys:
  "condition": string,
  "rationale": string (why these findings can relate to it, citing [S#] where possible),
  "supporting_findings": string[].
Rules: This is NOT a diagnosis. Only reason from the provided findings. If findings
are insufficient, return an empty array. Output JSON only."""


def build_differential_messages(context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIFFERENTIAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"EXTRACTED FINDINGS / CONTEXT:\n{context}\n\nReturn the JSON array."},
    ]


FOLLOWUP_SYSTEM_PROMPT = """You are MedIntel. Based ONLY on the provided report context,
generate a concise list of smart, specific questions the patient could ask their
doctor at their next visit. Return STRICT JSON: a string array of 4-7 questions.
Make them actionable and grounded in the context (e.g. about repeating a specific
abnormal test). Output JSON only."""


def build_followup_messages(context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nReturn the JSON array of questions."},
    ]
