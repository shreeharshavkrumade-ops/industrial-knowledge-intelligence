"""Intelligent Search — TF-IDF keyword, LSA latent-semantic, and optional neural search."""
import time

from src.auth import require_auth
require_auth()

import pandas as pd
import plotly.express as px
import streamlit as st

from src import storage
from src.retrieval_engine import MODE_LABELS
from src.ui import download_csv, kpi_row, notice, page_header, require_data, result_card, style_fig

kb = page_header(
    "Intelligent Search",
    "Retrieve ranked evidence passages from your industrial documents",
)
if not require_data(kb):
    st.stop()

eng   = kb.engine
avail = eng.availability()

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
    st.subheader("🔍 Search the knowledge base")
    ex = st.pills("Example questions", EXAMPLES, selection_mode="single", key="search_pill")
    q  = st.text_input(
        "Your query",
        value=ex or "",
        placeholder="e.g. What maintenance was performed on Pump P-101?",
        key="search_q",
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    usable = [m for m, (ok, _) in avail.items() if ok]
    mode   = c1.radio("Retrieval mode", usable, format_func=lambda m: MODE_LABELS[m], horizontal=True)
    k      = c2.slider("Results to show", 1, 15, 5)
    hide_csv = c3.checkbox("Hide CSV records", value=False)

    # Neural search availability — collapsed unless relevant
    sem_ok, sem_note = avail["semantic"]
    if not sem_ok:
        with st.expander("⚙️ Neural semantic search", expanded=False):
            if "sentence_transformers" in sem_note.lower() or "not loaded" in sem_note.lower():
                st.caption(
                    "Neural (SBERT) search is not yet active. "
                    "If `sentence-transformers` is installed, it loads automatically at startup. "
                    "You can also trigger it manually below."
                )
            else:
                st.caption(f"Status: {sem_note}")
            if st.button("Load neural model now", key="load_sem"):
                with st.spinner("Loading sentence-transformers model…"):
                    ok = eng.load_semantic()
                if ok:
                    st.success("Neural model loaded — select 'Neural semantic (SBERT)' above.")
                else:
                    st.error(f"Could not load: {eng.semantic_error}")
                st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if q.strip():
    t0  = time.perf_counter()
    res = eng.search(q, mode, top_k=k * 3 if hide_csv else k)
    if hide_csv and not res.empty:
        res = res[~res["document"].str.endswith(".csv")].head(k)
    ms  = (time.perf_counter() - t0) * 1000
    storage.append_search_log(
        q, mode, len(res),
        float(res["score"].iloc[0]) if not res.empty else 0.0,
        ms,
    )

    kpi_row([
        ("Results",   len(res),                                                MODE_LABELS[mode], False),
        ("Top score", f"{res['score'].iloc[0]:.3f}" if not res.empty else "—", "cosine similarity", False),
        ("Latency",   f"{ms:.1f} ms",                                          "this query",       False),
    ])

    if res.empty:
        notice("No passage matched this query. Try different words or switch retrieval mode.", "warn")
    else:
        for r in res.itertuples(index=False):
            result_card(r.rank, r.document, r.location, r.score, r.excerpt, q,
                        max_score=float(res["score"].max()))

        out = res.drop(columns=["doc_id"]).assign(query=q, mode=MODE_LABELS[mode])
        download_csv(out, "Download search results (CSV)", "search_results.csv", "dl_search")

        with st.expander("📊 Compare retrieval modes on this query"):
            rows = []
            for m_cmp in usable:
                r_cmp = eng.rank_documents(q, m_cmp)[:3]
                rows.append({"mode": MODE_LABELS[m_cmp], "top-3 documents": " | ".join(r_cmp)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                "Scores from different models are on different scales — compare rankings, "
                "not raw score values. See Compliance & Evaluation → AIML Evaluation for benchmarked metrics."
            )

# ---------------------------------------------------------------------------
# Search activity
# ---------------------------------------------------------------------------
log = storage.load_table("search_log")
if not log.empty:
    with st.expander(f"📈 Search activity ({len(log)} searches)"):
        a, b = st.columns(2)
        a.plotly_chart(
            style_fig(px.histogram(log, x="mode", title="Searches by mode"), 240),
            use_container_width=True,
        )
        b.plotly_chart(
            style_fig(px.histogram(log, x="runtime_ms", nbins=15, title="Query latency (ms)"), 240),
            use_container_width=True,
        )
        download_csv(log, "Download search log (CSV)", "search_log.csv", "dl_log")
