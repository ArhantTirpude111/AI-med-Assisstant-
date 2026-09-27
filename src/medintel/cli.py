"""Command-line interface for headless ingestion and Q&A.

Examples
--------
    python -m medintel.cli status
    python -m medintel.cli ingest data/uploads/blood_panel.pdf report.docx
    python -m medintel.cli ask "Are my liver enzymes elevated?"
    python -m medintel.cli summary
    python -m medintel.cli reset
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from medintel.assistant import MedIntelAssistant
from medintel.config import get_settings
from medintel.logging_config import configure_logging


def _force_utf8() -> None:
    """Ensure stdout/stderr can emit unicode (emoji, symbols) on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        # Not all streams are reconfigurable (e.g. when piped/redirected).
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _build_assistant() -> MedIntelAssistant:
    _force_utf8()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return MedIntelAssistant(settings)


def _cmd_status(_: argparse.Namespace) -> int:
    a = _build_assistant()
    print(f"App            : {a.settings.app_name}")
    print(f"LLM provider   : {a.settings.llm.provider} ({a.settings.llm.model})")
    print(f"Embeddings     : {a.settings.embeddings.provider}")
    print(f"OCR available  : {a.ocr_available}")
    print(f"NER backend    : {a.ner_backend}")
    print(f"Indexed chunks : {a.chunk_count}")
    print(f"Index ready    : {a.is_ready}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    a = _build_assistant()
    files: list[tuple[str, bytes]] = []
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            print(f"  ! not found: {p}", file=sys.stderr)
            continue
        files.append((p.name, p.read_bytes()))
    if not files:
        print("No valid files to ingest.", file=sys.stderr)
        return 1
    report = a.ingest(files)
    print(f"Ingested {report.document_count} document(s), {report.chunks_indexed} chunks "
          f"(OCR used: {report.used_ocr}).")
    for err in report.errors:
        print(f"  ! {err}", file=sys.stderr)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    a = _build_assistant()
    if not a.is_ready:
        print("No documents indexed. Run `ingest` first.", file=sys.stderr)
        return 1
    answer = a.ask(args.question)
    print("\n=== ANSWER ===")
    print(answer.answer_text)
    print(f"\nConfidence: {answer.confidence.label.value} "
          f"({answer.confidence.retrieval_confidence:.0%})")
    if answer.citations:
        print("\n=== SOURCES ===")
        for i, c in enumerate(answer.citations, 1):
            print(f"[{i}] {c.filename} p.{c.page_number} ({c.score:.0%})")
            print(f"    \"{c.snippet}\"")
    if answer.disclaimer:
        print(f"\n{answer.disclaimer}")
    return 0


def _cmd_summary(_: argparse.Namespace) -> int:
    a = _build_assistant()
    if not a.is_ready:
        print("No documents indexed. Run `ingest` first.", file=sys.stderr)
        return 1
    s = a.summarize()
    print("\n=== EXECUTIVE SUMMARY ===")
    print(s.executive_summary or "(none)")
    for title, items in (
        ("KEY FINDINGS", s.key_findings),
        ("ABNORMAL VALUES", s.abnormal_values),
        ("RISK INDICATORS", s.risk_indicators),
        ("FOLLOW-UP QUESTIONS", s.followup_questions),
    ):
        print(f"\n=== {title} ===")
        for item in items or ["(none)"]:
            print(f" • {item}")
    return 0


def _cmd_reset(_: argparse.Namespace) -> int:
    _build_assistant().reset()
    print("Index, history, and in-session state cleared.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medintel", description="MedIntel AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show configuration and index status").set_defaults(func=_cmd_status)

    p_ingest = sub.add_parser("ingest", help="Ingest one or more report files")
    p_ingest.add_argument("paths", nargs="+", help="Paths to PDF/DOCX/TXT/image files")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question about indexed reports")
    p_ask.add_argument("question", help="Natural-language question")
    p_ask.set_defaults(func=_cmd_ask)

    sub.add_parser("summary", help="Generate a report summary").set_defaults(func=_cmd_summary)
    sub.add_parser("reset", help="Clear index, history, and state").set_defaults(func=_cmd_reset)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
