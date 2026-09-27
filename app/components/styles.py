"""Healthcare-themed CSS and small visual helpers for the Streamlit UI."""

from __future__ import annotations

# Calm clinical palette: teal/blue primary, soft surfaces, clear status colors.
CUSTOM_CSS = """
<style>
:root {
    --mi-primary: #0e7c86;
    --mi-primary-dark: #0a5d65;
    --mi-accent: #2563eb;
    --mi-bg-soft: #f0f7f8;
    --mi-text: #1f2937;
}
.stApp { background: linear-gradient(180deg, #f7fbfc 0%, #ffffff 100%); }

.mi-hero {
    background: linear-gradient(135deg, var(--mi-primary) 0%, var(--mi-accent) 100%);
    color: white; padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(14,124,134,0.18);
}
.mi-hero h1 { color: white; margin: 0; font-size: 1.7rem; }
.mi-hero p { color: #e6fbff; margin: 0.25rem 0 0; font-size: 0.95rem; }

.mi-card {
    background: white; border: 1px solid #e5eef0; border-radius: 14px;
    padding: 1rem 1.2rem; margin-bottom: 0.85rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.mi-citation {
    background: var(--mi-bg-soft); border-left: 4px solid var(--mi-primary);
    border-radius: 8px; padding: 0.6rem 0.85rem; margin: 0.4rem 0; font-size: 0.9rem;
}
.mi-citation .src { color: var(--mi-primary-dark); font-weight: 600; font-size: 0.82rem; }
.mi-citation .snippet { color: #374151; font-style: italic; margin-top: 0.25rem; }

.mi-badge { display:inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em; }
.mi-badge-high { background:#dcfce7; color:#166534; }
.mi-badge-medium { background:#fef9c3; color:#854d0e; }
.mi-badge-low { background:#ffedd5; color:#9a3412; }
.mi-badge-insufficient { background:#fee2e2; color:#991b1b; }

.mi-emergency {
    background:#fef2f2; border:2px solid #ef4444; color:#7f1d1d;
    border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.8rem; font-weight:600;
}
.mi-pill { display:inline-block; background:var(--mi-bg-soft); color:var(--mi-primary-dark);
    border-radius:999px; padding:0.15rem 0.6rem; font-size:0.78rem; margin:0.12rem; }
.mi-disclaimer { color:#6b7280; font-size:0.8rem; border-top:1px dashed #d1d5db;
    margin-top:0.6rem; padding-top:0.5rem; }
.mi-status-ok { color:#16a34a; font-weight:600; }
.mi-status-off { color:#9ca3af; font-weight:600; }
</style>
"""


def confidence_badge_html(label: str, pct: float) -> str:
    cls = {
        "high": "mi-badge-high",
        "medium": "mi-badge-medium",
        "low": "mi-badge-low",
        "insufficient": "mi-badge-insufficient",
    }.get(label, "mi-badge-insufficient")
    return f'<span class="mi-badge {cls}">{label.upper()} · {pct:.0%}</span>'


def citation_html(filename: str, page: int, score: float, snippet: str) -> str:
    return (
        '<div class="mi-citation">'
        f'<div class="src">📄 {filename} — Page {page} · relevance {score:.0%}</div>'
        f'<div class="snippet">“{snippet}”</div>'
        "</div>"
    )
