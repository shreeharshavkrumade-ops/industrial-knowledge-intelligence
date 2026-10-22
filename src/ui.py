"""Shared Streamlit UI helpers — pastel Liquid Glass theme.

All presentation helpers live here; no ML logic.
"""
from __future__ import annotations

import html
import json as _json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from .glass_component import render_glass_header
from .pipeline import KnowledgeBase
from .storage import df_to_csv_bytes

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "theme.css"

# Pastel colour palette for Plotly traces
PLOT_COLORS = [
    "#4A90D9",  # sky blue
    "#52B788",  # mint green
    "#D96A8A",  # blush rose
    "#C99A2E",  # butter gold
    "#7C5CBF",  # lavender purple
    "#3AAFA9",  # teal
    "#E8834A",  # peach
    "#8FA3BF",  # slate
]

STATUS_BADGE = {
    "processed":   ("ok",   "&#10004; Processed"),
    "warning":     ("warn", "&#9650; Warning"),
    "error":       ("err",  "&#10006; Error"),
    "empty":       ("warn", "&#9650; Empty"),
    "unsupported": ("err",  "&#10006; Unsupported"),
    "duplicate":   ("info", "&#8801; Duplicate"),
}
STATUS_TEXT = {
    "processed": "✔ Processed", "warning": "▲ Warning", "error": "✖ Error",
    "empty": "▲ Empty", "unsupported": "✖ Unsupported", "duplicate": "≡ Duplicate",
}
COMPLIANCE_TEXT = {
    "Compliant": "✔ Compliant", "Overdue": "✖ Overdue",
    "Missing evidence": "▲ Missing evidence", "Documentation gap": "▲ Documentation gap",
}

# Pastel chart colours for compliance status
COMPLIANCE_COLORS = {
    "Compliant":          "#52B788",
    "Overdue":            "#D96A8A",
    "Missing evidence":   "#C99A2E",
    "Documentation gap":  "#7C5CBF",
}


@st.cache_resource(show_spinner="Loading knowledge base…")
def get_kb() -> KnowledgeBase:
    """One KnowledgeBase per server process (single-user local app)."""
    kb = KnowledgeBase()
    kb.ensure_demo_data()
    return kb


def inject_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


def inject_auth_bg() -> None:
    """Inject a base64-encoded background image for Login and Sign Up pages.

    Reads the image file fresh on every call (no caching).
    Supports JPG, JPEG, PNG, and WebP.
    Falls back to a CSS gradient if no image file is found.
    Prints a diagnostic to the terminal on every call.
    """
    import base64
    import logging
    _log = logging.getLogger("iki.auth_bg")

    assets_dir = Path(__file__).resolve().parents[1] / "assets"

    # Search for background image in priority order
    candidates = [
        ("auth_background.jpg",  "image/jpeg"),
        ("auth_background.jpeg", "image/jpeg"),
        ("auth_background.png",  "image/png"),
        ("auth_background.webp", "image/webp"),
    ]

    bg_css_value = ""
    found_file: Path | None = None

    for fname, mime in candidates:
        p = assets_dir / fname
        if p.exists():
            try:
                raw = p.read_bytes()
                if len(raw) < 100:
                    _log.warning("inject_auth_bg: %s is too small (%d bytes), skipping", fname, len(raw))
                    continue
                b64 = base64.b64encode(raw).decode()
                sub_type = mime.split("/")[1]
                bg_css_value = f"url('data:{mime};base64,{b64}') center center / cover no-repeat fixed"
                found_file = p
                mtime = int(p.stat().st_mtime)
                size_kb = len(raw) // 1024
                _log.info("inject_auth_bg: loaded %s (%d KB, mtime=%d)", fname, size_kb, mtime)
                print(f"[auth_bg] Loaded: {p.name} | {size_kb} KB | {mime} | mtime={mtime}", flush=True)
                break
            except Exception as exc:
                _log.warning("inject_auth_bg: failed to read %s: %s", fname, exc)

    if not bg_css_value:
        # Graceful pastel-dark gradient fallback when no image is found
        bg_css_value = (
            "linear-gradient(135deg,"
            "#0d1b3e 0%, #1a2a5e 30%, #162040 60%, #0a1428 100%) fixed"
        )
        print(f"[auth_bg] No image found in {assets_dir} — using gradient fallback", flush=True)

    # Embed mtime as a CSS comment so the browser treats each new image as a
    # different stylesheet (effective cache-buster without query strings)
    mtime_comment = f"/* bg-version: {int(found_file.stat().st_mtime) if found_file else 0} */"

    css = f"""
    <style>
    {mtime_comment}
    /* ── Auth page full-screen background ─────────────────────────────── */
    [data-testid="stApp"] {{
        background: {bg_css_value} !important;
        min-height: 100vh;
    }}
    /* Semi-transparent overlay so the form card stays readable */
    [data-testid="stApp"]::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: linear-gradient(
            135deg,
            rgba(8, 18, 55, 0.60) 0%,
            rgba(18, 28, 75, 0.50) 50%,
            rgba(8, 14, 40, 0.65) 100%
        );
        z-index: 0;
        pointer-events: none;
    }}
    /* Lift Streamlit content above the overlay */
    [data-testid="stApp"] > * {{
        position: relative;
        z-index: 1;
    }}
    /* ── Auth form glass card ──────────────────────────────────────────── */
    .auth-page-wrapper {{
        padding-top: 44px;
        display: flex;
        justify-content: center;
    }}
    .auth-form-card {{
        background: rgba(255, 255, 255, 0.09) !important;
        backdrop-filter: blur(18px) saturate(140%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
        border: 1px solid rgba(255, 255, 255, 0.20) !important;
        border-radius: 26px !important;
        padding: 34px 38px 30px !important;
        box-shadow: 0 10px 48px rgba(0, 0, 0, 0.38),
                    inset 0 1px 0 rgba(255, 255, 255, 0.16) !important;
        max-width: 480px;
        width: 100%;
    }}
    .auth-title {{
        color: rgba(240, 248, 255, 0.97) !important;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.40) !important;
    }}
    .auth-subtitle {{
        color: rgba(200, 225, 255, 0.78) !important;
    }}
    .auth-logo {{
        filter: drop-shadow(0 2px 18px rgba(74, 144, 217, 0.65)) !important;
    }}
    /* ── Sidebar on auth pages ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: rgba(8, 18, 50, 0.88) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border-right: 1px solid rgba(74, 144, 217, 0.18) !important;
    }}
    [data-testid="stSidebarNavLink"] span,
    [data-testid="stSidebarNavLink"] p {{
        color: rgba(210, 230, 255, 0.88) !important;
    }}
    [data-testid="stSidebarNavLink"]:hover {{
        background: rgba(74, 144, 217, 0.22) !important;
    }}
    [data-testid="stSidebarNavLink"][aria-selected="true"] {{
        background: rgba(74, 144, 217, 0.30) !important;
    }}
    /* ── Form inputs stay readable on dark bg ──────────────────────────── */
    [data-testid="stTextInput"] input {{
        background: rgba(255, 255, 255, 0.10) !important;
        border: 1px solid rgba(180, 210, 255, 0.35) !important;
        color: rgba(240, 248, 255, 0.95) !important;
    }}
    [data-testid="stTextInput"] input::placeholder {{
        color: rgba(180, 210, 255, 0.55) !important;
    }}
    [data-testid="stTextInput"] label p {{
        color: rgba(210, 230, 255, 0.88) !important;
    }}
    /* Primary button */
    [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
    [data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(135deg, #3a7bd5 0%, #2563c7 100%) !important;
        border: 1px solid rgba(74, 144, 217, 0.55) !important;
        color: #ffffff !important;
        font-weight: 620 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 18px rgba(37, 99, 199, 0.40) !important;
    }}
    [data-testid="stFormSubmitButton"] > button:hover {{
        background: linear-gradient(135deg, #4a8be5 0%, #3573d7 100%) !important;
        box-shadow: 0 6px 24px rgba(37, 99, 199, 0.55) !important;
        transform: translateY(-2px) !important;
    }}
    /* Info / error / success alerts keep readable contrast */
    [data-testid="stAlert"] {{
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(180, 210, 255, 0.40) !important;
        color: rgba(240, 248, 255, 0.95) !important;
        backdrop-filter: blur(8px) !important;
    }}
    [data-testid="stAlert"] p {{ color: rgba(240, 248, 255, 0.95) !important; }}
    /* "Don't have an account?" footer text */
    .auth-footer-text {{ color: rgba(180, 210, 255, 0.75) !important; }}
    /* Mobile responsive */
    @media (max-width: 768px) {{
        .auth-form-card {{
            padding: 22px 18px !important;
            border-radius: 18px !important;
        }}
        .auth-page-wrapper {{ padding-top: 20px; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> KnowledgeBase:
    inject_css()
    render_glass_header(title, subtitle)
    return get_kb()


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
_ACCENT_CYCLE = ["accent-sky", "accent-mint", "accent-lav", "accent-butter", "accent-blush"]


def kpi_row(items: list[tuple[str, object, str, bool]]) -> None:
    """Render a row of pastel glass KPI cards.

    items: (label, value, hint, use_accent)  — accent = blush/rose tint
    """
    cards = []
    for i, (label, value, hint, accent) in enumerate(items):
        cls = _ACCENT_CYCLE[i % len(_ACCENT_CYCLE)] if not accent else "accent"
        cards.append(
            f'<div class="glass-card kpi {cls}">'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}</div>'
            f'<div class="kpi-hint">{html.escape(hint)}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Notices, badges, legends
# ---------------------------------------------------------------------------

def notice(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="notice {kind}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Light-themed pre / code / json helpers
# ---------------------------------------------------------------------------
_PRE_STYLE = (
    "background:rgba(240,246,255,.90);"
    "border:1px solid rgba(180,210,240,.55);"
    "border-radius:12px;"
    "padding:14px 16px;"
    "color:#1A2E44;"
    "font-family:'Cascadia Code','Fira Code',Consolas,monospace;"
    "font-size:13px;"
    "line-height:1.7;"
    "white-space:pre-wrap;"
    "word-break:break-word;"
    "overflow-x:auto;"
    "margin:0;"
    "box-shadow:inset 0 1px 4px rgba(60,100,160,.06);"
)


def light_pre(text: str, max_chars: int = 600) -> None:
    """Render extracted text in a light monospaced panel.

    Replaces st.code() everywhere we show extracted / cleaned document text
    so the panel background is always light regardless of OS color-scheme.
    """
    safe = html.escape((text or "(empty)")[:max_chars])
    st.markdown(f'<pre style="{_PRE_STYLE}">{safe}</pre>', unsafe_allow_html=True)


def light_json(data: object) -> None:
    """Render a dict / list as pretty-printed JSON in a light panel.

    Replaces st.json() to avoid the dark Streamlit JSON viewer.
    """
    safe = html.escape(_json.dumps(data, indent=2, default=str))
    st.markdown(f'<pre style="{_PRE_STYLE}">{safe}</pre>', unsafe_allow_html=True)


def badge(status: str) -> str:
    cls, label = STATUS_BADGE.get(status, ("info", html.escape(status)))
    return f'<span class="badge {cls}">{label}</span>'


def legend(colors: dict[str, str]) -> None:
    st.markdown(
        '<div class="legend">'
        + "".join(f'<span><i style="background:{c}"></i>{html.escape(k)}</span>'
                  for k, c in colors.items())
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart styling — light / pastel
# ---------------------------------------------------------------------------

def style_fig(fig, height: int = 360):
    """Light pastel chart style for all Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        height=height,
        font=dict(color="#263449", size=13),
        colorway=PLOT_COLORS,
        margin=dict(l=10, r=10, t=44, b=10),
        title_font=dict(size=14, color="#263449", family="Segoe UI, system-ui"),
        legend=dict(
            bgcolor="rgba(255,255,255,.75)",
            bordercolor="rgba(180,210,240,.50)",
            borderwidth=1,
            font=dict(color="#263449"),
        ),
        hoverlabel=dict(bgcolor="rgba(255,255,255,.95)", font_color="#263449"),
    )
    fig.update_xaxes(gridcolor="rgba(180,210,240,.30)", zerolinecolor="rgba(180,210,240,.50)",
                     tickfont=dict(color="#68758A"))
    fig.update_yaxes(gridcolor="rgba(180,210,240,.30)", zerolinecolor="rgba(180,210,240,.50)",
                     tickfont=dict(color="#68758A"))
    return fig


# ---------------------------------------------------------------------------
# Search result cards
# ---------------------------------------------------------------------------

def highlight(text: str, query: str) -> str:
    """HTML-escape text and wrap query words in <mark>."""
    out   = html.escape(text)
    words = {w for w in re.findall(r"[A-Za-z0-9\-]{3,}", query)} - {
        "the", "what", "which", "are", "was", "were", "and", "for", "with"
    }
    for w in sorted(words, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(html.escape(w))})\b",
                     r"<mark>\1</mark>", out)
    return out


def result_card(rank: int, document: str, location: str, score: float,
                excerpt: str, query: str, max_score: float = 1.0, extra: str = "") -> None:
    pct = max(2, min(100, int(100 * score / max(max_score, 1e-9))))
    st.markdown(
        f'<div class="glass-card result-card">'
        f'<div class="result-head">'
        f'<span class="result-title">#{rank}&nbsp;&nbsp;{html.escape(document)}</span>'
        f'<span class="result-meta">📍 {html.escape(str(location))}'
        f'&nbsp;·&nbsp;relevance {score:.3f}{extra}</span>'
        f'</div>'
        f'<div class="scorebar" role="img" aria-label="relevance {score:.3f}">'
        f'<div style="width:{pct}%"></div></div>'
        f'<div class="result-excerpt">{highlight(excerpt, query)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_csv(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        label, data=df_to_csv_bytes(df), file_name=filename, mime="text/csv",
        key=key, disabled=df is None or df.empty,
    )


def require_data(kb: KnowledgeBase) -> bool:
    if kb.is_empty:
        notice(
            "The knowledge base is empty. Open <b>Document Manager</b> and "
            "load the sample dataset or upload your own files.",
            "warn",
        )
        return False
    return True
