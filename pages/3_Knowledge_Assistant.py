"""AI Knowledge Assistant — Gemini AI + local extractive fallback.

Primary mode  : Google Gemini (gemini-2.0-flash, free tier) with evidence
                retrieved from documents AND maintenance logs.
Fallback mode : extractive QA from the same evidence pool.

Free-tier limits (Gemini API):
  • 15 requests / minute
  • 1 500 requests / day
  On quota exhaustion the app automatically falls back to local extractive QA.

No answers are invented. Every response is grounded in retrieved passages.
"""
from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

from src.auth import require_auth
user = require_auth()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.gemini_assistant import (
    GEMINI_MODEL, ask_gemini, is_available,
    reset_connection_cache, status_message, test_connection,
)
from src.knowledge_assistant import answer_question
from src.maintenance_log import init_db as ml_init_db, logs_as_text_chunks
from src.retrieval_engine import MODE_LABELS, RetrievalEngine
from src.ui import highlight, inject_css, notice, page_header, require_data

inject_css()
ml_init_db()

kb = page_header(
    "AI Knowledge Assistant",
    "Ask anything about your industrial documents and maintenance records",
)

if not require_data(kb):
    st.stop()

# ---------------------------------------------------------------------------
# Build a combined engine: documents + maintenance logs
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing maintenance logs…")
def _combined_engine(_kb, _ml_version: int):
    """Merge document chunks with maintenance log chunks into one engine."""
    import importlib.util
    doc_chunks = _kb.chunks.copy() if not _kb.chunks.empty else pd.DataFrame()
    ml_chunks  = pd.DataFrame(logs_as_text_chunks())
    if ml_chunks.empty and doc_chunks.empty:
        return _kb.engine, False
    all_chunks = pd.concat([doc_chunks, ml_chunks], ignore_index=True) if not ml_chunks.empty else doc_chunks
    engine = RetrievalEngine(all_chunks)
    # Auto-load neural model if available
    if importlib.util.find_spec("sentence_transformers"):
        try:
            engine.load_semantic()
        except Exception:
            pass
    return engine, not ml_chunks.empty

# Version key so cache refreshes when new logs are added
from src.maintenance_log import summary_stats as _ml_stats
_ml_ver = _ml_stats().get("total", 0)
eng, has_ml = _combined_engine(kb, _ml_ver)

usable = [m for m, (ok, _) in eng.availability().items() if ok]

# ---------------------------------------------------------------------------
# Compact API status (top of page — unobtrusive)
# ---------------------------------------------------------------------------
conn      = test_connection()
stat      = conn["status"]
is_gemini = stat == "connected"

if is_gemini:
    st.caption(f"🟢 Gemini AI connected · `{GEMINI_MODEL}` (free tier)")
elif stat == "missing":
    with st.expander("⚙️ Gemini AI not configured — using local search", expanded=False):
        st.markdown("""
**To enable AI-generated answers:**
1. Get a free API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Add it to your `.env` file in the project root:
   ```
   GEMINI_API_KEY=AIza...your-key-here
   ```
3. Restart the app.

**Free tier limits:** 15 requests/minute · 1 500 requests/day · No billing required.

Without the key the assistant uses local extractive search, which finds relevant passages
but does not generate detailed narrative answers.
""")
elif stat == "quota":
    st.caption("⏳ Gemini free-tier quota reached — using local search. Resets in a few minutes/hours.")
else:
    st.caption("⚠️ Gemini temporarily unavailable — using local search.")
    with st.expander("🔧 Details", expanded=False):
        st.caption(conn["message"])
        col_rt, _ = st.columns([1, 4])
        with col_rt:
            if st.button("🔄 Re-test connection", key="gemini_retest"):
                reset_connection_cache()
                st.rerun()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    mode = st.radio("Retrieval mode", usable, format_func=lambda m: MODE_LABELS[m],
                    horizontal=True, key="qa_mode")
with c2:
    use_gemini = st.toggle(
        "Use Gemini AI",
        value=True,
        disabled=False,
        key="qa_use_gemini",
        help="Requires GEMINI_API_KEY in .env — free tier, no billing needed",
    )
with c3:
    if st.button("🗑 Clear chat", key="qa_clear"):
        st.session_state.pop("gemini_chat",    None)
        st.session_state.pop("gemini_history", None)
        st.rerun()

if has_ml:
    st.caption("📋 Maintenance log records are included in the knowledge base.")

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
chat:    list[dict] = st.session_state.setdefault("gemini_chat",    [])
history: list[dict] = st.session_state.setdefault("gemini_history", [])

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------
SUGGESTED = [
    "What maintenance was performed on Pump P-101?",
    "Which critical faults are still pending?",
    "Who repaired Motor M-401 last time?",
    "What precautions apply before motor inspection?",
    "Summarize the maintenance history for this asset.",
]

if not chat:
    st.markdown("#### 💡 Suggested questions")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state["_prefill"] = q
            st.rerun()

prefill = st.session_state.pop("_prefill", "")


# ---------------------------------------------------------------------------
# Message renderers
# ---------------------------------------------------------------------------
def _citation_card(c: dict, q: str) -> str:
    tag    = "quoted in answer" if c.get("quoted") else "retrieved"
    is_log = str(c.get("document", "")).startswith("[Maintenance Log]")
    icon   = "📋" if is_log else "📄"
    return (
        f'<div class="citation-card">'
        f'<div class="cit-header">'
        f'<span class="cit-title">[{c["n"]}] {icon} {html.escape(c["document"])}</span>'
        f'<span class="cit-meta">'
        f'📍 {html.escape(str(c["location"]))} · score {c["score"]:.3f} · {tag}'
        f'</span></div>'
        f'<div class="cit-body">{highlight(c["excerpt"][:300], q)}</div>'
        f'</div>'
    )


def _mode_label(mode_key: str) -> str:
    return {
        "gemini":              "🟢 Gemini AI · gemini-2.0-flash",
        "extractive_fallback": "🔵 Local evidence search",
    }.get(mode_key, "")


def _show_message(msg: dict) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
        return

    r = msg["data"]
    with st.chat_message("assistant"):
        lbl = _mode_label(r.get("mode", ""))
        if lbl:
            st.caption(lbl)

        # Compact API error note inside message
        if r.get("error"):
            st.caption(f"⚠️ {r['error'][:120]}")

        if r["sufficient"]:
            st.markdown(r["answer"])
        else:
            notice(html.escape(r["answer"]), "warn")

        cits = r.get("citations", [])
        if cits:
            q_text = r.get("question", "")
            with st.expander(f"📄 Evidence & citations ({len(cits)} passages)",
                             expanded=r["sufficient"]):
                for c in cits:
                    st.markdown(_citation_card(c, q_text), unsafe_allow_html=True)

        for n in r.get("notes", []):
            st.caption(n)


# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in chat:
    _show_message(msg)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input(
    "Ask about equipment faults, maintenance history, documents, safety…",
    key="qa_input",
)
if prefill and not question:
    question = prefill

if question:
    question = question.strip()
    user_msg = {"role": "user", "text": question}
    chat.append(user_msg)
    _show_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and maintenance records…"):
            if use_gemini:
                result = ask_gemini(question, eng, mode, conversation_history=history)
            else:
                result = answer_question(eng, question, mode)
                result["mode"] = "extractive_fallback"

        if result.get("mode") == "gemini" and result.get("sufficient"):
            history.append({"role": "user",  "content": question})
            history.append({"role": "model", "content": result["answer"]})
            history = history[-12:]
            st.session_state["gemini_history"] = history

    asst_msg = {"role": "assistant", "data": result}
    chat.append(asst_msg)
    st.rerun()
