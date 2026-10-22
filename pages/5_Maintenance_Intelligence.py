"""Maintenance Intelligence — clustering, recurrence analysis, evidence-backed recommendations,
baseline risk classifier. Pastel Liquid Glass theme edition."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.auth import require_auth
require_auth()

from src.maintenance_analyzer import fault_frequency, monthly_trend
from src.ui import download_csv, kpi_row, notice, page_header, require_data, style_fig

kb = page_header(
    "Maintenance Intelligence",
    "TF-IDF + K-Means fault grouping, recurring-issue detection, and evidence-backed review suggestions",
)
if not require_data(kb):
    st.stop()

m, mstats = kb.maintenance()
if m.empty:
    notice(
        "No maintenance records found. Load the sample dataset or upload a CSV with columns "
        "<code>work_order_id, date, equipment_id, fault_description</code>.",
        "warn",
    )
    st.stop()

st.caption(
    "ℹ️ Results reflect **observed patterns** in the supplied records, not proven root causes. "
    "Sample data are synthetic."
)

with st.container(border=True):
    auto = st.toggle("Choose cluster count automatically (best silhouette)", value=True)
    k    = None if auto else st.slider("Number of clusters (k)", 2, 10, 8)

cl = kb.clustering(k)
if cl is None:
    notice("Need at least 4 maintenance records to cluster.", "warn")
    st.stop()

met  = cl["metrics"]
data = cl["data"]

kpi_row([
    ("Work orders",   len(data),                 f"{data['equipment_id'].nunique()} assets",       False),
    ("Clusters (k)",  met["k"],                  "auto by silhouette" if auto else "manual",        False),
    ("Silhouette",    f"{met['silhouette']:.3f}", "cluster cohesion (−1 to 1, higher = better)",    False),
    ("Total downtime",
     f"{data['downtime_hours'].sum():,.0f} h" if "downtime_hours" in data.columns else "n/a",
     "after median imputation",
     True),
])

tabs = st.tabs([
    "📊 Fault categories",
    "📈 Trends & frequency",
    "🔁 Recurring issues & recommendations",
    "🤖 Failure-risk classifier",
])

# ---- Tab 0: Fault categories -------------------------------------------
with tabs[0], st.container(border=True):
    st.dataframe(
        cl["summary"].drop(columns=["cluster"]),
        use_container_width=True, hide_index=True,
    )
    a, b = st.columns(2)
    fig_cat = px.bar(
        cl["summary"].sort_values("records"),
        x="records", y="fault_category", orientation="h",
        title="Records per fault category",
        color="records",
        color_continuous_scale=["#DDF5E5", "#52B788"],
    )
    fig_cat.update_layout(coloraxis_showscale=False)
    a.plotly_chart(style_fig(fig_cat, 380), use_container_width=True)

    fig_sil = px.line(
        cl["silhouette_by_k"], x="k", y="silhouette", markers=True,
        title="Silhouette score by k",
    )
    fig_sil.update_traces(line=dict(color="#7C5CBF", width=2.5),
                          marker=dict(color="#7C5CBF", size=7))
    b.plotly_chart(style_fig(fig_sil, 380), use_container_width=True)

    if "adjusted_rand_index" in met:
        st.caption(
            f"External check against the synthetic generator's template labels "
            f"(never used as input): ARI = {met['adjusted_rand_index']:.3f}, "
            f"NMI = {met['normalized_mutual_info']:.3f}. "
            "These labels exist only because the data is synthetic."
        )
    download_csv(
        data.drop(columns=["cluster"]),
        "Download maintenance analysis (CSV)", "maintenance_analysis.csv", "dl_maint",
    )

# ---- Tab 1: Trends & frequency -----------------------------------------
with tabs[1], st.container(border=True):
    ff  = fault_frequency(data)
    tr  = monthly_trend(data)

    fig_freq = px.bar(
        ff, x="equipment_id", y="count", color="fault_category",
        title="Equipment fault frequency by category",
    )
    st.plotly_chart(style_fig(fig_freq, 400), use_container_width=True)

    fig_trend = px.line(
        tr, x="month", y="work_orders", markers=True,
        title="Work orders per month",
        line_shape="spline",
    )
    fig_trend.update_traces(
        line=dict(color="#4A90D9", width=2.5),
        marker=dict(color="#4A90D9", size=6),
    )
    st.plotly_chart(style_fig(fig_trend, 320), use_container_width=True)

# ---- Tab 2: Recurring issues -------------------------------------------
with tabs[2]:
    rec, recs = kb.recommendations()
    with st.container(border=True):
        st.subheader("Recurring equipment issues (≥ 3 occurrences)")
        if rec.empty:
            notice("No equipment has 3 or more recorded occurrences of the same fault category.", "info")
        else:
            st.dataframe(rec, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Evidence-backed review suggestions")
        if recs.empty:
            notice("No review suggestions generated — load more maintenance records.", "info")
        else:
            for r in recs.itertuples(index=False):
                with st.expander(
                    f"**{r.equipment_id}** — {r.fault_category}  ({r.occurrences} records)"
                ):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Observed pattern:** {r.observed_pattern}")
                    c1.markdown(f"**Suggested review:** {r.suggested_review}")
                    c1.markdown(f"**Supporting work orders:** {r.supporting_work_orders}")
                    c2.markdown(
                        f"**Manual / document evidence:** `{r.manual_evidence_document}` "
                        f"(relevance {r.evidence_score})"
                    )
                    if r.manual_evidence_excerpt:
                        c2.caption(r.manual_evidence_excerpt)
            download_csv(
                recs, "Download recommendations (CSV)",
                "maintenance_recommendations.csv", "dl_recs",
            )

# ---- Tab 3: Failure-risk classifier ------------------------------------
with tabs[3], st.container(border=True):
    clf = kb.classification()
    if not clf["available"]:
        notice(f"Classifier not available: {clf['reason']}", "info")
    else:
        notice(
            "<b>Baseline only.</b> The target <code>failure_within_30d</code> is generated "
            "from a noisy synthetic rule — scores show the model can recover that rule, "
            "not real failure prediction.",
            "warn",
        )
        st.markdown(
            f"**Target:** `{clf['target']}`&nbsp; "
            f"**Features:** {', '.join(clf['features'])}&nbsp; "
            f"**Split:** stratified {int((1-clf['test_size'])*100)}/{int(clf['test_size']*100)} "
            f"(train {clf['n_train']}, test {clf['n_test']}), seed {clf['seed']}; "
            f"positive rate train {clf['train_positive_rate']:.2f} / "
            f"test {clf['test_positive_rate']:.2f}"
        )
        st.dataframe(clf["metrics"].round(3), use_container_width=True, hide_index=True)

        a, b = st.columns(2)
        best = clf["metrics"].sort_values("f1", ascending=False).iloc[0]["model"]
        sel  = a.selectbox(
            "Confusion matrix for",
            list(clf["confusion_matrices"]),
            index=list(clf["confusion_matrices"]).index(best),
        )
        cm  = clf["confusion_matrices"][sel]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
            text=cm, texttemplate="%{text}",
            colorscale=[[0, "#DCEEFF"], [1, "#4A90D9"]],
            showscale=False,
        ))
        a.plotly_chart(
            style_fig(fig_cm.update_layout(title=f"Confusion matrix (test set) — {sel}"), 330),
            use_container_width=True,
        )
        fi = clf["feature_importance"].head(8).sort_values("importance")
        fig_fi = px.bar(
            fi, x="importance", y="feature", orientation="h",
            title="Random Forest feature importance",
            color="importance",
            color_continuous_scale=["#FFF2C6", "#C99A2E"],
        )
        fig_fi.update_layout(coloraxis_showscale=False)
        b.plotly_chart(style_fig(fig_fi, 330), use_container_width=True)
