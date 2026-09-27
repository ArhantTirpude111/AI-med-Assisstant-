"""MedIntel AI — Streamlit dashboard.

A healthcare-themed UI over the :class:`MedIntelAssistant` service. Presentation
only: all logic lives in the ``medintel`` package, so this file stays declarative.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Path bootstrap ---------------------------------------------------------
# Make both the package (`medintel`, under src/) and the local `app` namespace
# importable whether the app is launched via `streamlit run app/streamlit_app.py`
# (which only puts app/ on sys.path) or after `pip install -e .`.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402
from app.components.styles import (  # noqa: E402
    CUSTOM_CSS,
    citation_html,
    confidence_badge_html,
)

from medintel.assistant import MedIntelAssistant  # noqa: E402
from medintel.config import get_settings  # noqa: E402
from medintel.ingestion.loaders import supported_extensions  # noqa: E402
from medintel.logging_config import configure_logging  # noqa: E402
from medintel.models.schemas import Answer  # noqa: E402

# ---------------------------------------------------------------------------
st.set_page_config(page_title="MedIntel AI", page_icon="🧬", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "What does my report indicate?",
    "Are my lab values normal?",
    "Explain my diagnosis in simple terms.",
    "What should I discuss with my doctor?",
    "Summarize my report.",
]


@st.cache_resource(show_spinner="Initializing MedIntel engine…")
def get_assistant() -> MedIntelAssistant:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return MedIntelAssistant(settings)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])      # list[dict(role, content, answer?)]
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("ingest_report", None)


# ===========================================================================
# Sidebar
# ===========================================================================
def render_sidebar(assistant: MedIntelAssistant) -> None:
    with st.sidebar:
        st.markdown("### 🧬 MedIntel AI")
        st.caption("Grounded, citation-backed medical report analysis")

        st.markdown("#### 📤 Upload reports")
        uploads = st.file_uploader(
            "PDF · DOCX · TXT · scanned/image",
            type=supported_extensions(),
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("⚙️ Process & Index", use_container_width=True, type="primary"):
            if not uploads:
                st.warning("Please choose at least one file.")
            else:
                files = [(f.name, f.getvalue()) for f in uploads]
                with st.spinner("Extracting, embedding and indexing…"):
                    report = assistant.ingest(files)
                st.session_state.ingest_report = report
                st.session_state.summary = None
                if report.errors:
                    for err in report.errors:
                        st.error(err)
                if report.document_count:
                    st.success(
                        f"Indexed {report.document_count} document(s) · "
                        f"{report.chunks_indexed} chunks"
                        + (" · OCR used" if report.used_ocr else "")
                    )

        # Document status
        st.markdown("#### 📚 Documents")
        if assistant.loaded_documents:
            for doc in assistant.loaded_documents:
                ocr = " · 🔍 OCR" if doc.used_ocr else ""
                st.markdown(
                    f'<div class="mi-card" style="padding:0.55rem 0.8rem;">'
                    f"<b>{doc.filename}</b><br>"
                    f'<span class="mi-pill">{doc.page_count} pages</span>'
                    f'<span class="mi-pill">{doc.source_type.value}</span>{ocr}'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents indexed yet.")

        # Engine status
        st.markdown("#### 🩺 Engine status")
        ready = "✅ Ready" if assistant.is_ready else "⏳ Awaiting documents"
        ocr_status = (
            '<span class="mi-status-ok">available</span>'
            if assistant.ocr_available
            else '<span class="mi-status-off">not installed</span>'
        )
        st.markdown(
            f"""
            <div class="mi-card" style="font-size:0.85rem;">
            <b>Index:</b> {ready} ({assistant.chunk_count} chunks)<br>
            <b>LLM:</b> {assistant.settings.llm.provider} · {assistant.settings.llm.model}<br>
            <b>Embeddings:</b> {assistant.settings.embeddings.provider}<br>
            <b>OCR:</b> {ocr_status}<br>
            <b>NER backend:</b> {assistant.ner_backend}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Reset session", use_container_width=True):
            assistant.reset()
            st.session_state.messages = []
            st.session_state.summary = None
            st.session_state.ingest_report = None
            st.rerun()

        st.markdown(
            '<div class="mi-disclaimer">⚕️ Educational tool — not a medical device. '
            "Always consult a qualified clinician.</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# Rendering helpers
# ===========================================================================
def render_answer(answer: Answer) -> None:
    for flag in answer.guardrail_flags:
        if flag.code == "emergency":
            st.markdown(f'<div class="mi-emergency">{flag.message}</div>', unsafe_allow_html=True)

    st.markdown(answer.answer_text)

    conf = answer.confidence
    badge = confidence_badge_html(conf.label.value, conf.retrieval_confidence)
    st.markdown(
        f"{badge} &nbsp; <span style='color:#6b7280;font-size:0.82rem;'>"
        f"top similarity {conf.top_similarity:.0%} · {conf.source_count} sources</span>",
        unsafe_allow_html=True,
    )

    if answer.citations:
        with st.expander(f"📑 Sources ({len(answer.citations)})", expanded=True):
            for c in answer.citations:
                st.markdown(
                    citation_html(c.filename, c.page_number, c.score, c.snippet),
                    unsafe_allow_html=True,
                )
    if answer.disclaimer:
        st.markdown(f'<div class="mi-disclaimer">{answer.disclaimer}</div>', unsafe_allow_html=True)


# ===========================================================================
# Tabs
# ===========================================================================
def tab_chat(assistant: MedIntelAssistant) -> None:
    st.markdown("#### 💬 Ask about your reports")
    if not assistant.is_ready:
        st.info("Upload and index a report from the sidebar to begin.")

    cols = st.columns(len(SAMPLE_QUESTIONS))
    clicked: str | None = None
    for col, q in zip(cols, SAMPLE_QUESTIONS, strict=False):
        if col.button(q, use_container_width=True, disabled=not assistant.is_ready):
            clicked = q

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("answer"):
                render_answer(msg["answer"])
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your medical report…", disabled=not assistant.is_ready)
    question = prompt or clicked
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and composing a grounded answer…"):
                answer = assistant.ask(question)
            render_answer(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.answer_text, "answer": answer}
        )


def tab_summary(assistant: MedIntelAssistant) -> None:
    st.markdown("#### 📋 Report summary")
    if not assistant.is_ready:
        st.info("Index a report to generate a summary.")
        return
    if st.button("🧾 Generate / refresh summary", type="primary"):
        with st.spinner("Synthesizing summary from retrieved evidence…"):
            st.session_state.summary = assistant.summarize()

    summary = st.session_state.summary
    if not summary:
        return

    st.markdown(f'<div class="mi-card"><b>Executive summary</b><br>{summary.executive_summary or "—"}</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _bullet_card("🔑 Key findings", summary.key_findings)
        _bullet_card("⚠️ Abnormal values", summary.abnormal_values)
    with c2:
        _bullet_card("🚩 Risk indicators", summary.risk_indicators)
        _bullet_card("❓ Suggested questions", summary.followup_questions)

    if summary.citations:
        with st.expander("📑 Evidence"):
            for c in summary.citations:
                st.markdown(citation_html(c.filename, c.page_number, c.score, c.snippet),
                            unsafe_allow_html=True)


def tab_insights(assistant: MedIntelAssistant) -> None:
    st.markdown("#### 🧠 Insights & education")
    left, right = st.columns(2)

    with left:
        st.markdown("##### 🔤 Terminology simplifier")
        term = st.text_input("Medical term", placeholder="e.g. Hepatic Steatosis")
        if st.button("Explain term") and term.strip():
            with st.spinner("Generating multi-level explanation…"):
                result = assistant.simplify_term(term)
            st.markdown(f'<div class="mi-card"><b>Beginner</b><br>{result.beginner}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mi-card"><b>Intermediate</b><br>{result.intermediate}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mi-card"><b>Patient-friendly</b><br>{result.patient_friendly}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mi-card"><b>🧸 Explain like I\'m 5</b><br>{result.eli5}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("##### ❓ Follow-up questions for your doctor")
        if st.button("Generate questions", disabled=not assistant.is_ready):
            with st.spinner("Drafting questions…"):
                for q in assistant.followups():
                    st.markdown(f"- {q}")

    st.divider()
    st.markdown("##### 🩺 Differential reasoning (educational only)")
    st.caption("Possible conditions associated with the extracted findings — **not a diagnosis**.")
    if st.button("Suggest differentials", disabled=not assistant.is_ready):
        with st.spinner("Reasoning over extracted findings…"):
            diffs = assistant.differential()
        if not diffs:
            st.info("Not enough grounded evidence to suggest differentials.")
        for d in diffs:
            st.markdown(
                f'<div class="mi-card"><b>{d.condition}</b><br>{d.rationale}'
                + ("".join(f'<span class="mi-pill">{f}</span>' for f in d.supporting_findings))
                + f'<div class="mi-disclaimer">{d.disclaimer}</div></div>',
                unsafe_allow_html=True,
            )


def tab_graph(assistant: MedIntelAssistant) -> None:
    st.markdown("#### 🕸️ Clinical knowledge graph")
    st.caption("Relationships inferred from extracted entities (diagnoses ↔ symptoms ↔ medications ↔ labs).")
    if assistant.entities.is_empty:
        st.info("Index a report to build the knowledge graph.")
        return
    graph = assistant.knowledge_graph()
    if graph.number_of_nodes() == 0:
        st.info("No linkable clinical entities were detected.")
        return
    stats = assistant.kg_service.stats(graph)
    cols = st.columns(4)
    cols[0].metric("Nodes", stats.get("nodes", 0))
    cols[1].metric("Edges", stats.get("edges", 0))
    cols[2].metric("Diagnoses", stats.get("diagnosis", 0))
    cols[3].metric("Medications", stats.get("medication", 0))
    st.plotly_chart(assistant.kg_service.to_plotly_figure(graph), use_container_width=True)


def tab_timeline(assistant: MedIntelAssistant) -> None:
    st.markdown("#### 🗓️ Patient timeline")
    st.caption("Chronological clinical events extracted from dated lines in your reports.")
    if not assistant.loaded_documents:
        st.info("Index a report to build a timeline.")
        return
    events = assistant.timeline()
    if not events:
        st.info("No dated events were detected in the uploaded reports.")
        return
    icons = {
        "diagnosis": "🩺", "medication": "💊", "lab": "🧪",
        "symptom": "🤒", "procedure": "🔬", "physician_note": "📝",
    }
    for e in events:
        icon = icons.get(e.event_type.value, "•")
        st.markdown(
            f'<div class="mi-card" style="padding:0.55rem 0.9rem;">'
            f'<b>{icon} {e.date or "Undated"}</b> · '
            f'<span class="mi-pill">{e.event_type.value}</span><br>{e.description}'
            f'<div class="mi-disclaimer">{e.filename} · page {e.page_number}</div></div>',
            unsafe_allow_html=True,
        )


def _bullet_card(title: str, items: list[str]) -> None:
    body = "".join(f"<li>{i}</li>" for i in items) if items else "<li>—</li>"
    st.markdown(f'<div class="mi-card"><b>{title}</b><ul>{body}</ul></div>', unsafe_allow_html=True)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    _init_state()
    assistant = get_assistant()

    st.markdown(
        '<div class="mi-hero"><h1>🧬 MedIntel AI</h1>'
        "<p>Retrieval-Augmented Medical Report Analysis — grounded answers, "
        "real citations, built-in safety.</p></div>",
        unsafe_allow_html=True,
    )

    render_sidebar(assistant)

    chat, summary, insights, graph, timeline = st.tabs(
        ["💬 Chat", "📋 Summary", "🧠 Insights", "🕸️ Knowledge Graph", "🗓️ Timeline"]
    )
    with chat:
        tab_chat(assistant)
    with summary:
        tab_summary(assistant)
    with insights:
        tab_insights(assistant)
    with graph:
        tab_graph(assistant)
    with timeline:
        tab_timeline(assistant)


if __name__ == "__main__":
    main()
