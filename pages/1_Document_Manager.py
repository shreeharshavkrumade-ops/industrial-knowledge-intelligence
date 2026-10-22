"""Document Manager — ingest PDF / DOCX / TXT / CSV, OCR, chunking, entity extraction.

Fix (2026-10-22):
  - Process files individually so one failure does not freeze the batch.
  - Show per-file progress bar and status indicator.
  - Call st.rerun() after upload so the Documents table updates immediately.
  - Avoid blocking the Streamlit thread on SBERT model initialisation during rebuild.
  - Wrap rebuild() in try/except so index errors don't swallow uploaded docs.
  - Log exceptions to terminal with full tracebacks.
"""
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.auth import require_auth
from src.document_processor import SUPPORTED, easyocr_available, ocr_available, tesseract_available

LOG = logging.getLogger("iki.docmanager")

require_auth()
from src.ui import (
    STATUS_TEXT,
    download_csv,
    kpi_row,
    light_json,
    light_pre,
    notice,
    page_header,
    style_fig,
)
kb = page_header(
    "Document Manager",
    "Upload industrial documents — automatic text extraction, OCR, chunking, and entity extraction",
)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("📤 Upload documents")

    if tesseract_available():
        st.caption("✅ Tesseract OCR available — scanned PDF pages will be processed automatically.")
    elif easyocr_available():
        st.caption("✅ EasyOCR available — scanned PDF pages will be processed automatically (no Tesseract binary required).")
    else:
        st.caption(
            "⚙️ No OCR backend installed — text-layer PDFs work fine; scanned/image-only pages are skipped. "
            "Install EasyOCR (`pip install easyocr`) for scanned PDF support without a system binary."
        )

    with st.form("upload_form", clear_on_submit=True):
        files = st.file_uploader(
            "Drop files here or click to browse",
            type=[e.lstrip(".") for e in sorted(SUPPORTED)],
            accept_multiple_files=True,
            help="Files are stored in data/uploads. Duplicates are detected by content hash.",
        )
        st.caption(f"Supported formats: {', '.join(sorted(SUPPORTED)).upper().replace('.', '')}")
        go = st.form_submit_button("⚡ Process uploaded files", type="primary")

    if go:
        if not files:
            notice("No files selected.", "warn")
        else:
            results = []
            n = len(files)
            progress_bar = st.progress(0, text=f"Processing 0/{n} files…")

            for idx, f in enumerate(files):
                fname = f.name
                progress_bar.progress(idx / n, text=f"⏳ Processing **{fname}** ({idx + 1}/{n})…")
                try:
                    rec = kb.ingest(fname, f.getvalue())
                    results.append(rec)
                    LOG.info("Ingested %s: status=%s chunks=%d", fname, rec["status"], rec["n_chunks"])
                except Exception as exc:
                    LOG.exception("Upload ingestion failed for %s", fname)
                    results.append({
                        "name": fname,
                        "status": "error",
                        "n_chunks": 0,
                        "message": f"Unexpected error: {type(exc).__name__}: {exc}",
                    })

            progress_bar.progress(1.0, text="⏳ Rebuilding knowledge index…")
            try:
                kb.rebuild()
                LOG.info("rebuild() complete after upload of %d files", n)
            except Exception as exc:
                LOG.exception("rebuild() failed after upload")
                notice(f"⚠️ Index rebuild error (uploaded content was saved but not indexed): {exc}", "err")

            progress_bar.empty()

            # Show per-file outcomes
            for r in results:
                kind = {"processed": "info", "warning": "warn", "duplicate": "info"}.get(
                    r["status"], "err"
                )
                notice(
                    f"<b>{r['name']}</b> — {STATUS_TEXT.get(r['status'], r['status'])}: "
                    f"{r['n_chunks']} chunks. {r['message']}",
                    kind,
                )

            # CRITICAL: rerun so the Documents table re-reads from the updated kb state.
            # Without this, Streamlit renders the pre-upload snapshot of kb.documents.
            st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📚 Load / refresh sample dataset"):
            with st.spinner("Loading sample documents…"):
                res = kb.load_samples()
            new_count = sum(r["status"] != "duplicate" for r in res)
            dup_count = sum(r["status"] == "duplicate" for r in res)
            st.success(f"Sample dataset loaded: {new_count} new, {dup_count} already present.")
            st.rerun()
    with col_b:
        with st.popover("🗑 Reset processed data"):
            st.write(
                "Clears the processed database only. "
                "Your files in **data/uploads** and **data/sample_documents** are **not** deleted."
            )
            if st.button("Confirm reset", key="confirm_reset"):
                kb.reset_derived()
                kb.load_samples()
                st.rerun()

if kb.is_empty:
    notice("No documents loaded yet. Upload files or load the sample dataset above.", "warn")
    st.stop()

# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------
ps = kb.preprocessing_stats()
kpi_row([
    ("Documents",    ps["documents"],          ", ".join(f"{v} {k}" for k, v in ps["by_type"].items()), False),
    ("Chunks",       ps["chunks"],             f"avg {ps['avg_chunk_words']} words",                     False),
    ("Raw chars",    f"{ps['raw_chars']:,}",   "as extracted",                                           False),
    ("Clean chars",  f"{ps['clean_chars']:,}", "after normalisation",                                    False),
    ("Problems",     sum(v for k, v in ps["status_counts"].items() if k != "processed"),
                     "warnings / errors / empty",                                                         True),
])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_docs, tab_preview, tab_chunks, tab_feat, tab_ent = st.tabs([
    "📄 Documents",
    "🔍 Text preview",
    "📦 Chunks",
    "📊 TF-IDF features",
    "🏷 Entities",
])

with tab_docs, st.container(border=True):
    d = kb.documents.copy()
    d["status"] = d["status"].map(lambda x: STATUS_TEXT.get(x, x))
    st.dataframe(
        d[["name", "file_type", "source", "status", "n_pages", "text_length", "n_chunks", "message"]]
          .rename(columns={"n_pages": "pages/rows", "text_length": "text length", "n_chunks": "chunks"}),
        use_container_width=True, hide_index=True,
    )

    # ── Delete a document ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗑 Delete a document")

    doc_options: dict[str, str] = {
        f"{row['name']}  ({str(row['file_type']).upper()} · {row.get('source', 'upload')})": str(row["doc_id"])
        for _, row in kb.documents.iterrows()
    }

    if not doc_options:
        notice("No documents to delete.", "info")
    else:
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_label = st.selectbox(
                "Select document to delete",
                options=list(doc_options.keys()),
                key="del_doc_select",
                help="Choose the document you want to permanently remove.",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)   # vertical alignment
            open_confirm = st.button(
                "🗑 Delete", key="del_open_confirm",
                type="secondary", use_container_width=True,
            )

        if open_confirm:
            st.session_state["_del_confirm_open"] = True

        if st.session_state.get("_del_confirm_open"):
            selected_doc_id = doc_options.get(selected_label, "")
            selected_name   = selected_label.split("  (")[0]

            with st.container(border=True):
                st.warning(
                    f"⚠️ **Confirm deletion of:** `{selected_name}`\n\n"
                    "This will permanently remove the document file (if uploaded), all its extracted "
                    "text chunks, entities, and knowledge-graph relationships. "
                    "**This cannot be undone.**"
                )
                col_yes, col_no, _ = st.columns([1, 1, 3])
                confirmed = col_yes.button(
                    "✅ Yes, delete", key="del_confirm_yes",
                    type="primary", use_container_width=True,
                )
                cancelled = col_no.button(
                    "❌ Cancel", key="del_confirm_no",
                    use_container_width=True,
                )

            if cancelled:
                st.session_state.pop("_del_confirm_open", None)
                st.rerun()

            if confirmed:
                st.session_state.pop("_del_confirm_open", None)
                with st.spinner(f"Deleting {selected_name}…"):
                    result = kb.delete_document(selected_doc_id)

                if result["ok"]:
                    parts = [
                        f"**{result['name']}** deleted.",
                        f"Chunks removed: {result['chunks_removed']}",
                        f"Entities removed: {result['entities_removed']}",
                    ]
                    if result["file_deleted"]:
                        parts.append("File removed from disk.")
                    notice(" · ".join(parts), "info")
                    for w in result.get("warnings", []):
                        st.caption(f"⚠️ {w}")
                    # Invalidate the @st.cache_resource KnowledgeBase so the
                    # next page render loads the updated state from DB.
                    st.cache_resource.clear()
                else:
                    notice(f"Deletion failed: {result['error']}", "err")

                st.rerun()

with tab_preview, st.container(border=True):
    doc_names = kb.documents["name"].tolist()
    name = st.selectbox("Select document", doc_names, key="pre_doc")
    row  = kb.documents[kb.documents["name"] == name].iloc[0]
    ca, cb = st.columns(2)
    ca.markdown("**Raw extracted text (first 600 chars)**")
    with ca:
        light_pre(row.get("raw_preview", "") or "(not stored — reload sample data)")
    cb.markdown("**After cleaning (first 600 chars)**")
    with cb:
        light_pre(row.get("clean_preview", "") or "")
    with st.expander("Preprocessing details"):
        st.caption(
            "Cleaning: Unicode NFKC normalisation, control-character removal, "
            "hyphenation / line-wrap repair, whitespace collapse. "
            "CSV missing values: median imputation for downtime; rows without date/equipment dropped."
        )
        _, mstats = kb.maintenance()
        if mstats.get("rows_in"):
            st.markdown("**Maintenance table preprocessing report**")
            light_json(mstats)

with tab_chunks, st.container(border=True):
    name = st.selectbox("Select document", kb.documents["name"].tolist(), key="chunk_doc")
    ch   = kb.chunks[kb.chunks["document"] == name][["chunk_id", "location", "n_words", "text"]]
    st.caption(
        "Chunks are sentence-aware (~90 words) with ~20-word overlap so facts at page "
        "boundaries are not lost. CSV rows become individual single chunks."
    )
    st.dataframe(ch, use_container_width=True, hide_index=True)

with tab_feat, st.container(border=True):
    eng = kb.engine
    if eng is not None and eng.tfidf is not None:
        import numpy as np
        n, f = eng.tfidf.shape
        sparsity = 100 * (1 - eng.tfidf.nnz / (n * f))
        kpi_row([
            ("Chunks (rows)",  n,           "TF-IDF matrix",           False),
            ("Vocabulary",     f"{f:,}",    "uni+bi-grams + asset IDs", False),
            ("Sparsity",       f"{sparsity:.1f}%", "zero entries",      False),
            ("TF-IDF build",   f"{eng.build_seconds.get('tfidf', 0) * 1000:.0f} ms", "fit + transform", False),
        ])
        w     = np.asarray(eng.tfidf.sum(axis=0)).ravel()
        terms = np.array(eng.vectorizer.get_feature_names_out())
        top   = pd.DataFrame({
            "term":                terms[np.argsort(-w)[:20]],
            "summed tf-idf weight": np.sort(w)[::-1][:20],
        })
        fig = px.bar(
            top.sort_values("summed tf-idf weight"), x="summed tf-idf weight", y="term",
            orientation="h", title="Top-20 TF-IDF terms in the corpus",
            color="summed tf-idf weight",
            color_continuous_scale=["#DCEEFF", "#4A90D9"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
    else:
        notice("TF-IDF index not built yet.", "info")

with tab_ent, st.container(border=True):
    e = kb.entities
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect(
        "Entity type", sorted(e["entity_type"].unique()),
        default=["asset_id", "fault", "component", "maintenance_action"],
    )
    docs  = f2.multiselect("Document", sorted(e["document"].unique()))
    kinds = f3.multiselect("Basis", ["extracted", "inferred"], default=["extracted", "inferred"])
    v = e[e["entity_type"].isin(types) & e["kind"].isin(kinds)]
    if docs:
        v = v[v["document"].isin(docs)]
    st.caption(
        "**extracted** = found verbatim in text; "
        "**inferred** = derived by rule (e.g. equipment type from asset-ID prefix)."
    )
    st.dataframe(
        v[["entity_type", "value", "kind", "document", "location", "evidence"]],
        use_container_width=True, hide_index=True,
    )
    download_csv(v, "Download entities (CSV)", "extracted_entities.csv", "dl_ent")
