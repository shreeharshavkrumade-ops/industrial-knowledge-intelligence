"""Neural Search — semantic and keyword document retrieval.

Replaces the old Intelligent Search page.
Supports TF-IDF (always), LSA (local latent-semantic), and
Neural (sentence-transformers/all-MiniLM-L6-v2, auto-loaded).
"""
from __future__ import annotations

import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.auth import require_auth
require_auth()

from src import storage
from src.retrieval_engine import MODE_LABELS
from src.ui import download_csv, kpi_row, notice, page_header, require_data, result_card, style_fig

# Suppress Windows symlinks warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

kb = page_header(
    "Neural Search",
    "Semantic and keyword retrieval across all indexed documents",
)
if not require_data(kb):
    st.stop()

eng   = kb.engine
avail = eng.availability()
sem_ok, sem_note = avail["semantic"]

# ---------------------------------------------------------------------------
# Auto-load neural model (non-blocking — only if not already loaded)
# ---------------------------------------------------------------------------
if not sem_ok and not st.session_state.get("_sem_load_attempted"):
    st.session_state["_sem_load_attempted"] = True
    with st.spinner("Loading neural embedding model (first run — one-off download)…"):
        ok = eng.load_semantic()
    if ok:
        avail = eng.availability()
        sem_ok, sem_note = avail["semantic"]
        st.toast("✅ Neural model ready", icon="🧠")
    else:
        st.toast(f"Neural model unavailable: {eng.semantic_error}", icon="⚠️")

usable = [m for m, (ok, _) in avail.items() if ok]

# ---------------------------------------------------------------------------
# Neural model status (compact, non-intrusive)
# ---------------------------------------------------------------------------
if sem_ok:
    st.caption("🧠 Neural semantic search active (sentence-transformers/all-MiniLM-L6-v2)")
else:
    with st.expander("⚙️ Neural model status", expanded=False):
        st.caption(f"Neural search unavailable: {sem_note}")
        if st.button("Retry loading neural model", key="retry_sem"):
            st.session_state.pop("_sem_load_attempted", None)
            with st.spinner("Loading model…"):
                ok = eng.load_semantic()
            if ok:
                st.success("Neural model loaded.")
            else:
                st.error(f"Could not load: {eng.semantic_error}")
            st.rerun()

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------
EXAMPLES = [
    "What maintenance was performed on Pump P-101?",
    "Causes of overheating in Compressor C-202?",
    "Safety precautions before motor inspection?",
    "Which documents mention repeated bearing failures?",
    "Inspection findings for Boiler B-301?",
]

with st.container(border=True):
    st.subheader("🔍 Search")
    ex = st.pills("Example queries", EXAMPLES, selection_mode="single", key="ns_pill")
    q  = st.text_input(
        "Your query",
        value=ex or "",
        placeholder="e.g. What caused the bearing failure on Motor M-401?",
        key="ns_q",
    )
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    mode     = c1.radio("Mode", usable, format_func=lambda m: MODE_LABELS[m],
                        horizontal=True, key="ns_mode")
    k        = c2.slider("Results", 1, 15, 5, key="ns_k")
    min_sc   = c3.slider("Min score", 0.0, 1.0, 0.0, step=0.05, key="ns_minscore")
    hide_csv = c4.checkbox("Hide CSV rows", value=False, key="ns_hidecsv")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if q.strip():
    t0  = time.perf_counter()
    res = eng.search(q, mode, top_k=k * 3 if hide_csv else k, min_score=min_sc)
    if hide_csv and not res.empty:
        res = res[~res["document"].str.endswith(".csv")].head(k)
    ms  = (time.perf_counter() - t0) * 1000

    storage.append_search_log(
        q, mode, len(res),
        float(res["score"].iloc[0]) if not res.empty else 0.0,
        ms,
    )

    kpi_row([
        ("Results",   len(res),                                                    MODE_LABELS[mode], False),
        ("Top score", f"{res['score'].iloc[0]:.3f}" if not res.empty else "—",    "cosine similarity", False),
        ("Latency",   f"{ms:.1f} ms",                                              "this query",        False),
    ])

    if res.empty:
        notice("No passage matched this query. Try different words or switch mode.", "warn")
    else:
        for r in res.itertuples(index=False):
            result_card(r.rank, r.document, r.location, r.score, r.excerpt, q,
                        max_score=float(res["score"].max()))

        out = res.drop(columns=["doc_id"]).assign(query=q, mode=MODE_LABELS[mode])
        download_csv(out, "Download results (CSV)", "search_results.csv", "dl_ns")

        with st.expander("📊 Mode comparison"):
            rows = []
            for m_cmp in usable:
                r_cmp = eng.rank_documents(q, m_cmp)[:3]
                rows.append({"Mode": MODE_LABELS[m_cmp], "Top-3 documents": " | ".join(r_cmp)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Score scales differ across modes — compare rankings, not raw values.")

# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------
log = storage.load_table("search_log")
if not log.empty:
    with st.expander(f"📈 Search history ({len(log)} queries)", expanded=False):
        a, b = st.columns(2)
        a.plotly_chart(
            style_fig(px.histogram(log, x="mode", title="Searches by mode"), 220),
            use_container_width=True,
        )
        b.plotly_chart(
            style_fig(px.histogram(log, x="runtime_ms", nbins=15, title="Query latency (ms)"), 220),
            use_container_width=True,
        )
        download_csv(log, "Download history (CSV)", "search_log.csv", "dl_nslog")
