"""Compliance & Evaluation — combined page with two internal tabs.

Tab 1 — Compliance:      illustrative compliance checking, audit reports.
Tab 2 — AIML Evaluation: metrics computed from real predictions vs. labelled data.

Both sets of functionality are fully preserved from the original separate pages.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.auth import require_auth
require_auth()
from src.compliance_checker import DISCLAIMER, audit_pdf_bytes, summarize
from src.pipeline import DEFAULT_AS_OF
from src.ui import (
    COMPLIANCE_COLORS,
    COMPLIANCE_TEXT,
    download_csv,
    kpi_row,
    light_json,
    notice,
    page_header,
    require_data,
    style_fig,
)

kb = page_header(
    "Compliance & Evaluation",
    "Compliance monitoring and AIML model evaluation in one place",
)
if not require_data(kb):
    st.stop()

tab_comp, tab_eval = st.tabs(["📋 Compliance", "🔬 AIML Evaluation"])

# ===========================================================================
# TAB 1: COMPLIANCE
# ===========================================================================
with tab_comp:
    with st.expander("ℹ️ About compliance data", expanded=False):
        st.caption(f"**Illustrative only.** {DISCLAIMER}")

    with st.container(border=True):
        as_of = st.date_input(
            "Audit as-of date", value=DEFAULT_AS_OF,
            help="Fixed default makes sample results reproducible.",
        )

    res = kb.compliance(as_of)
    if res.empty:
        notice("No compliance requirements or records available.", "info")
        st.stop()

    s = summarize(res)
    kpi_row([
        ("Checks",           s["total"],                          "requirement × asset",    False),
        ("Compliant",        s["compliant"],                      "evidence found",         False),
        ("Gaps",             s["gaps"],                           "overdue / missing",      True),
        ("Overdue",          s["by_status"].get("Overdue", 0),    "past max interval",      False),
        ("Missing evidence", s["by_status"].get("Missing evidence", 0), "no matching record", False),
    ])

    with st.container(border=True):
        show = st.multiselect(
            "Status filter", list(COMPLIANCE_TEXT),
            default=["Overdue", "Missing evidence", "Documentation gap"],
        )
        v = res[res["status"].isin(show)] if show else res
        v = v.assign(status=v["status"].map(COMPLIANCE_TEXT))
        st.dataframe(
            v[["requirement_id", "title", "equipment_id", "status",
               "evidence", "source_document", "days_since_last", "limit_days", "note"]],
            use_container_width=True, hide_index=True,
        )
        fig = px.histogram(
            res, x="title", color="status",
            title="Check results by requirement",
            color_discrete_map=COMPLIANCE_COLORS,
        )
        st.plotly_chart(style_fig(fig.update_xaxes(tickangle=-25), 340),
                        use_container_width=True)

    with st.container(border=True):
        st.subheader("Audit report")
        a, b = st.columns(2)
        with a:
            download_csv(res, "Download audit (CSV)", "compliance_audit.csv", "dl_comp")
        with b:
            pdf = audit_pdf_bytes(res, as_of)
            if pdf:
                st.download_button(
                    "Download audit summary (PDF)", pdf,
                    "compliance_audit_summary.pdf", "application/pdf",
                )
            else:
                st.caption(
                    "PDF export needs `reportlab` (`pip install reportlab`); "
                    "CSV export is available above."
                )


# ===========================================================================
# TAB 2: AIML EVALUATION
# ===========================================================================
with tab_eval:
    if st.button("↺ Re-run all evaluations", key="eval_rerun"):
        ev = kb.evaluate(refresh=True)
    else:
        ev = kb.evaluate()

    ent, ret, clf, cl = ev["entity"], ev["retrieval"], ev["classification"], ev["clustering"]
    ps = kb.preprocessing_stats()

    subtabs = st.tabs([
        "Overview",
        "Dataset & preprocessing",
        "Entity extraction",
        "Retrieval",
        "Clustering",
        "Classification",
        "Interpretation & limits",
        "Export",
    ])

    # -- Overview -----------------------------------------------------------
    with subtabs[0], st.container(border=True):
        st.subheader("End-to-end pipeline status")
        rows = [
            ("1  Ingestion",     f"{ps['documents']} documents, {ps['chunks']} chunks",
             kb.timings.get("ingest_s")),
            ("2  Preprocessing", f"{len(kb.entities):,} entity mentions; "
             f"TF-IDF vocabulary {kb.engine.tfidf.shape[1]:,}",
             kb.timings.get("entity_extraction_s")),
            ("3  Models",        f"TF-IDF/LSA index, K-Means "
             f"(k={cl['metrics']['k'] if cl else '-'}), "
             f"classifier {'trained' if clf.get('available') else 'n/a'}",
             kb.timings.get("indexing_s")),
            ("4  Evaluation",    f"{len(ev['metrics_table'])} metrics computed",
             ev["evaluation_runtime_s"]),
            ("5  Visualisation", "charts on every page read these same tables", None),
            ("6  Application",   "search, assistant, graph, maintenance, compliance", None),
        ]
        cols = st.columns(3)
        for i, (a_t, b_t, t_t) in enumerate(rows):
            cols[i % 3].markdown(
                f'<div class="stage"><b>{a_t}</b><br>{b_t}<br>'
                f'<span>{"runtime %.3f s" % t_t if t_t is not None else "&nbsp;"}</span></div>',
                unsafe_allow_html=True,
            )

    # -- Dataset & preprocessing -------------------------------------------
    with subtabs[1], st.container(border=True):
        kpi_row([
            ("Documents",      ps["documents"],          ", ".join(f"{v} {k}" for k, v in ps["by_type"].items()), False),
            ("Chunks",         ps["chunks"],             f"avg {ps['avg_chunk_words']} words",                     False),
            ("Entity mentions",f"{len(kb.entities):,}", f"{kb.entities['entity_type'].nunique()} types",           False),
            ("Maintenance rows",len(kb.maintenance()[0]),"after cleaning",                                         False),
        ])
        with st.expander("Preprocessing details"):
            st.dataframe(
                kb.entities["entity_type"].value_counts().reset_index()
                  .rename(columns={"entity_type": "type", "count": "mentions"}),
                hide_index=True, use_container_width=True,
            )
            light_json({"maintenance_cleaning": kb.maintenance()[1],
                        "document_status":      ps["status_counts"]})
        fig = px.bar(
            kb.entities["entity_type"].value_counts().reset_index(),
            x="count", y="entity_type", orientation="h",
            title="Entity mentions by type",
            color="count",
            color_continuous_scale=["#DCEEFF", "#4A90D9"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    # -- Entity extraction -------------------------------------------------
    with subtabs[2], st.container(border=True):
        if not ent["available"]:
            notice(f"Entity evaluation unavailable: {ent['reason']}", "info")
        else:
            mi, ma = ent["micro"], ent["macro"]
            kpi_row([
                ("Micro precision", f"{mi['precision']:.3f}", f"{ent['n_predicted']} predicted", False),
                ("Micro recall",    f"{mi['recall']:.3f}",    f"{ent['n_gold']} gold",            False),
                ("Micro F1",        f"{mi['f1']:.3f}",        "all types pooled",                 True),
                ("Macro F1",        f"{ma['f1']:.3f}",        "mean over types",                  False),
            ])
            st.dataframe(ent["per_type"].round(3), use_container_width=True, hide_index=True)
            m_df = ent["per_type"].melt(
                id_vars="entity_type",
                value_vars=["precision", "recall", "f1"],
            )
            fig = px.bar(
                m_df, x="entity_type", y="value", color="variable", barmode="group",
                title="Entity extraction: precision / recall / F1 by type",
            )
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            with st.expander("Error analysis (false positives / false negatives)"):
                st.dataframe(ent["errors"], use_container_width=True, hide_index=True)
            notice(
                f"Evaluated on {len(ent['documents_evaluated'])} hand-labelled documents. "
                "<b>Optimism warning:</b> the same author wrote the lexicon, sample documents, "
                "and labels, so scores will be higher than on unseen real documents.",
                "warn",
            )

    # -- Retrieval ---------------------------------------------------------
    with subtabs[3], st.container(border=True):
        if not ret["available"]:
            notice(f"Retrieval evaluation unavailable: {ret['reason']}", "info")
        else:
            sm  = ret["summary"]
            ks  = ret["ks"]
            st.dataframe(sm.round(3), use_container_width=True, hide_index=True)
            long = sm.melt(
                id_vars="mode",
                value_vars=[f"{p}@{k}" for p in "PR" for k in ks] + ["MRR"],
                var_name="metric", value_name="score",
            )
            fig = px.bar(
                long, x="metric", y="score", color="mode", barmode="group",
                title="Precision@K, Recall@K and MRR by retrieval mode",
            )
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)
            for name, why in ret["unavailable"].items():
                st.caption(f"Not evaluated — **{name}**: {why}")
            with st.expander("Per-query results"):
                st.dataframe(ret["per_query"].round(3), use_container_width=True, hide_index=True)
            notice(
                "Document-level evaluation: each document is ranked by its best-matching chunk. "
                "Queries have 1–3 relevant documents, so P@5 cannot reach 1.0 by construction.",
                "info",
            )

    # -- Clustering --------------------------------------------------------
    with subtabs[4], st.container(border=True):
        if not cl:
            notice("Not enough maintenance records for clustering.", "info")
        else:
            mt = cl["metrics"]
            kpi_row(
                [("k",          mt["k"],                    "chosen by silhouette", False),
                 ("Silhouette", f"{mt['silhouette']:.3f}",  "internal validity",    False)]
                + ([("ARI",     f"{mt['adjusted_rand_index']:.3f}", "vs synthetic labels", False),
                    ("NMI",     f"{mt['normalized_mutual_info']:.3f}", "vs synthetic labels", False)]
                   if "adjusted_rand_index" in mt else [])
            )
            st.dataframe(cl["summary"], use_container_width=True, hide_index=True)
            fig = px.line(
                cl["silhouette_by_k"], x="k", y="silhouette", markers=True,
                title="Silhouette score vs k",
            )
            st.plotly_chart(style_fig(fig, 280), use_container_width=True)

    # -- Classification ----------------------------------------------------
    with subtabs[5], st.container(border=True):
        if not clf["available"]:
            notice(f"Classification not run: {clf['reason']}", "info")
        else:
            notice(
                "<b>Baseline only.</b> The target <code>failure_within_30d</code> is generated "
                "from a noisy synthetic rule — scores show the model can recover that rule, "
                "not real failure prediction.",
                "warn",
            )
            st.markdown(
                f"Target `{clf['target']}`; features {', '.join(clf['features'])}; "
                f"stratified split train {clf['n_train']} / test {clf['n_test']}; seed {clf['seed']}."
            )
            st.dataframe(clf["metrics"].round(3), use_container_width=True, hide_index=True)
            c_col, fi_col = st.columns(2)
            best = clf["metrics"].sort_values("f1", ascending=False).iloc[0]["model"]
            sel  = c_col.selectbox(
                "Confusion matrix for",
                list(clf["confusion_matrices"]),
                index=list(clf["confusion_matrices"]).index(best),
            )
            cm  = clf["confusion_matrices"][sel]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
                text=cm, texttemplate="%{text}",
                colorscale=[[0, "#DCEEFF"], [1, "#4A90D9"]],
                showscale=False,
            ))
            c_col.plotly_chart(
                style_fig(fig.update_layout(title=f"Confusion matrix — {sel}"), 310),
                use_container_width=True,
            )
            fi = clf["feature_importance"].head(8).sort_values("importance")
            fig2 = px.bar(
                fi, x="importance", y="feature", orientation="h",
                title="Random Forest feature importance",
                color="importance",
                color_continuous_scale=["#DDF5E5", "#52B788"],
            )
            fig2.update_layout(coloraxis_showscale=False)
            fi_col.plotly_chart(style_fig(fig2, 310), use_container_width=True)

    # -- Interpretation & limits -------------------------------------------
    with subtabs[6], st.container(border=True):
        st.subheader("Interpretation (generated from the numbers above)")
        pts: list[str] = []
        if ret.get("available") and len(ret["summary"]):
            s_r  = ret["summary"].sort_values("MRR", ascending=False)
            best = s_r.iloc[0]
            pts.append(f"On the {int(best['queries'])}-query benchmark, "
                        f"**{best['mode']}** has the highest MRR ({best['MRR']:.3f}).")
            if len(s_r) > 1:
                o   = s_r.iloc[1]
                gap = best["MRR"] - o["MRR"]
                pts.append(
                    f"The MRR gap to {o['mode']} ({o['MRR']:.3f}) is {gap:.3f}; "
                    + ("with so few queries a gap this small is **not statistically meaningful**."
                       if gap < 0.1 else "a larger difference, but still from a small benchmark.")
                )
        if ent.get("available"):
            w = ent["per_type"].sort_values("f1").iloc[0]
            pts.append(
                f"Entity extraction is weakest on **{w['entity_type']}** "
                f"(F1 {w['f1']:.3f}); see the error table for the exact misses."
            )
        if cl and "adjusted_rand_index" in cl["metrics"]:
            pts.append(
                f"K-Means found k={cl['metrics']['k']} groups "
                f"(silhouette {cl['metrics']['silhouette']:.3f}); "
                f"agreement with generator categories: ARI {cl['metrics']['adjusted_rand_index']:.3f}."
            )
        if clf.get("available"):
            b_clf  = clf["metrics"].sort_values("f1", ascending=False).iloc[0]
            b_base = clf["metrics"][clf["metrics"]["model"].str.contains("baseline")].iloc[0]
            pts.append(
                f"Best classifier by test F1: **{b_clf['model']}** ({b_clf['f1']:.3f}) "
                f"vs majority-class baseline ({b_base['f1']:.3f})."
            )
        st.markdown("\n".join(f"- {p}" for p in pts))

        st.subheader("Limitations")
        st.markdown(
            "- All data are fictional/synthetic and small; nothing here shows performance on real plant documents.\n"
            "- Entity extraction is lexicon/regex-based (not a trained NER model); "
            "lexicon, documents and labels share one author, so scores are optimistic.\n"
            "- Retrieval is evaluated at document level on a small hand-labelled benchmark; "
            "neural semantic search is evaluated only if sentence-transformers is installed.\n"
            "- LSA is a classical latent-semantic method, not a neural embedding model.\n"
            "- The failure-risk label is synthetic; clusters and recommendations are observed "
            "patterns, not root causes.\n"
            "- Compliance rules are fictional examples, not legal advice."
        )

    # -- Export ------------------------------------------------------------
    with subtabs[7], st.container(border=True):
        download_csv(ev["metrics_table"], "Download all evaluation metrics (CSV)",
                     "evaluation_metrics.csv", "dl_metrics")
        if ent.get("available"):
            download_csv(ent["errors"], "Download entity error analysis (CSV)",
                         "entity_errors.csv", "dl_errs")
        if ret.get("available"):
            download_csv(ret["per_query"], "Download per-query retrieval results (CSV)",
                         "retrieval_per_query.csv", "dl_rq")
        st.caption("Exports are generated from the same DataFrames that drive the charts above.")
