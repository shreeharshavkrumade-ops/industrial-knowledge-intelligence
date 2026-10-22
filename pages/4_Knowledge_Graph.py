"""Knowledge Graph — interactive entity-relationship graph with pastel theme."""
import streamlit as st

from src.auth import require_auth
require_auth()

from src.glass_component import render_html_frame
from src.knowledge_graph import COLORS, graph_stats, plotly_fallback, render_pyvis_html, subgraph_edges
from src.ui import download_csv, kpi_row, legend, notice, page_header, require_data

kb = page_header(
    "Knowledge Graph",
    "Equipment, components, faults, actions, and inspections — every edge is backed by document evidence",
)
if not require_data(kb):
    st.stop()

# Invalidate graph cache when maintenance log records change so new faults/technicians
# appear in the graph without requiring a manual app restart.
_ml_count = 0
try:
    from src.maintenance_log import count_logs
    _ml_count = count_logs()
except Exception:
    pass

_prev_ml_count = st.session_state.get("_kg_ml_count", None)
if _prev_ml_count != _ml_count:
    # Clear the edges cache so graph_edges() rebuilds from fresh DB data.
    if hasattr(kb, "_cache") and "edges" in kb._cache:
        del kb._cache["edges"]
    st.session_state["_kg_ml_count"] = _ml_count

edges = kb.graph_edges()
if edges.empty:
    notice("No relationships could be built from the current data. "
           "Load documents or the sample dataset to see the graph.", "warn")
    st.stop()

stats = graph_stats(edges)
kpi_row([
    ("Nodes",         stats["nodes"],      "entities",                  False),
    ("Edges",         stats["edges"],      "supported relations",        False),
    ("Connected subgraphs", stats["components"], "independent groups",  False),
    ("Co-occurrence edges",
     int((edges["basis"] == "sentence co-occurrence").sum()),
     "weaker proximity links (dashed)", True),
    ("Maintenance log edges",
     int((edges["basis"] == "maintenance log").sum()),
     "from structured maintenance records", False),
])

with st.container(border=True):
    st.subheader("🔍 Filter the graph")
    f1, f2, f3, f4 = st.columns([1.2, 2, 1, 1])

    assets = ["All"] + sorted(kb.assets()["equipment_id"].tolist())
    eq     = f1.selectbox(
        "Focus equipment", assets,
        index=assets.index("P-101") if "P-101" in assets else 0,
    )
    all_rels = sorted(edges["relation"].unique())
    rels     = f2.multiselect(
        "Relations", all_rels,
        default=[r for r in all_rels if r != "mentions"],
    )
    minsup = f3.number_input("Min. support", 1, 20, 1)
    maxe   = f4.number_input("Max edges", 20, 400, 120, step=20)

    sub = subgraph_edges(edges, eq, rels, minsup).head(int(maxe))
    legend(COLORS)

    if sub.empty:
        notice("No edges match these filters. Try relaxing the relation filter or min. support.", "info")
    else:
        htmls = render_pyvis_html(sub, 600)
        if htmls:
            render_html_frame(htmls, 624)
            st.caption(
                "Interactive: drag nodes, scroll to zoom, hover edges for evidence. "
                "Dashed edges = sentence co-occurrence (proximity in text, not a verified causal relation)."
            )
        else:
            notice(
                "PyVis is not installed — showing a static NetworkX / Plotly layout. "
                "Install with <code>pip install pyvis</code> for the interactive graph.",
                "warn",
            )
            st.plotly_chart(plotly_fallback(sub), use_container_width=True)

with st.container(border=True):
    st.subheader("📋 Relationship table")
    st.dataframe(
        sub[["source", "relation", "target", "basis",
             "support_count", "source_document", "evidence"]],
        use_container_width=True,
        hide_index=True,
    )
    download_csv(edges, "Download all relationships (CSV)",
                 "knowledge_graph_relationships.csv", "dl_kg")
