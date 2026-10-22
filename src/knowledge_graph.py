"""Knowledge graph built ONLY from supported evidence.

Edge bases (stored per edge, shown in the UI):
  structured record   - explicit fields of a work-order / inspection row
  sentence pattern    - a text pattern such as "<asset> is driven by <asset>"
  sentence co-occurrence - entities share one sentence that contains exactly one asset ID
                           (weaker: it shows proximity in text, not a verified causal relation)
  maintenance log     - explicit fields from the structured maintenance log database
Nothing is added merely to make the graph look connected.
"""
from __future__ import annotations

import html as _html
import re
import sqlite3
from pathlib import Path

import networkx as nx
import pandas as pd

from .entity_extractor import extract_from_sentence

_ML_DB = Path(__file__).resolve().parents[1] / "data" / "maintenance_log.db"

EDGE_COLS = ["source", "source_type", "relation", "target", "target_type", "basis", "evidence", "source_document", "support_count"]
COLORS = {"Document": "#8fa3bf", "Equipment": "#4f8ec9", "Component": "#6cc4b0", "Fault": "#e8824a",
          "Action": "#9bd27f", "Inspection": "#b49cf0", "Finding": "#e3c76b",
          "MaintenanceLog": "#f0a0c0", "Technician": "#c8a0e8"}
_DRIVE_1 = re.compile(r"\b([MP]-\d{3})\b[^.]*?\bdriven by\b[^.]*?\b([MP]-\d{3})\b", re.I)
_DRIVE_2 = re.compile(r"\b(M-\d{3})\b\s+drives\s+[A-Za-z ]*?\b([PC]-\d{3})\b", re.I)


def _edge(rows, s, st, rel, t, tt, basis, ev, doc):
    rows.append(dict(source=s, source_type=st, relation=rel, target=t, target_type=tt, basis=basis, evidence=str(ev)[:260], source_document=doc))


def _load_maintenance_log_rows() -> list[dict]:
    """Load maintenance log records from the SQLite DB (returns [] if DB absent)."""
    if not _ML_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(_ML_DB), timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT log_id, equipment_id, equipment_name, fault_title, description, "
            "       how_occurred, action_taken, assigned_to, reported_by, severity, status, "
            "       observed_at, resolved_at "
            "FROM maintenance_logs"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _build_maintenance_log_edges(rows: list[dict]) -> list[dict]:
    """Build graph edges from structured maintenance log records.

    Node types used:
      Equipment       — equipment_id
      Fault           — fault_title (labelled with severity)
      Action          — action_taken (from extracted maintenance actions)
      Technician      — assigned_to / reported_by  (people)
      MaintenanceLog  — log_id (the log record itself, as a hub node)

    All edges carry basis="maintenance log" so they are visually distinct
    from CSV-structured records and sentence co-occurrence edges.
    """
    edge_rows: list[dict] = []
    SOURCE = "maintenance_log.db"

    for r in rows:
        lid   = str(r.get("log_id", "")).strip()
        eid   = str(r.get("equipment_id", "")).strip().upper()
        ename = str(r.get("equipment_name", "")).strip()
        ftitle = str(r.get("fault_title", "")).strip()
        desc   = str(r.get("description", "")).strip()
        action = str(r.get("action_taken", "")).strip()
        tech   = str(r.get("assigned_to", "")).strip()
        rep    = str(r.get("reported_by", "")).strip()
        sev    = str(r.get("severity", "")).strip()
        status = str(r.get("status", "")).strip()
        date   = str(r.get("observed_at", ""))[:10]

        if not lid or not eid or not ftitle:
            continue

        fault_node = f"{ftitle} [{sev}]" if sev else ftitle
        ev_base = f"{lid} ({date}): {ftitle}"

        # Equipment → MaintenanceLog
        edge_rows.append(dict(source=eid, source_type="Equipment",
                              relation="has log entry", target=lid, target_type="MaintenanceLog",
                              basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
        # MaintenanceLog → Fault
        edge_rows.append(dict(source=lid, source_type="MaintenanceLog",
                              relation="records fault", target=fault_node, target_type="Fault",
                              basis="maintenance log", evidence=f"{ev_base} | status={status}"[:260], source_document=SOURCE))
        # Equipment → Fault (direct shortcut for subgraph_edges lookups)
        edge_rows.append(dict(source=eid, source_type="Equipment",
                              relation="experienced fault", target=fault_node, target_type="Fault",
                              basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
        # Maintenance actions extracted from action_taken text
        if action:
            acts = sorted({e["value"] for e in extract_from_sentence(action) if e["entity_type"] == "maintenance_action"})
            for act in acts:
                edge_rows.append(dict(source=fault_node, source_type="Fault",
                                      relation="addressed by", target=act, target_type="Action",
                                      basis="maintenance log", evidence=f"{ev_base} | {action[:100]}"[:260], source_document=SOURCE))
        # Equipment name label (if distinct from ID)
        if ename and ename.upper() != eid:
            edge_rows.append(dict(source=eid, source_type="Equipment",
                                  relation="named", target=ename, target_type="Component",
                                  basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
        # Technician edges
        if tech:
            edge_rows.append(dict(source=lid, source_type="MaintenanceLog",
                                  relation="assigned to", target=tech, target_type="Technician",
                                  basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
            edge_rows.append(dict(source=eid, source_type="Equipment",
                                  relation="maintained by", target=tech, target_type="Technician",
                                  basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
        if rep and rep != tech:
            edge_rows.append(dict(source=lid, source_type="MaintenanceLog",
                                  relation="reported by", target=rep, target_type="Technician",
                                  basis="maintenance log", evidence=ev_base[:260], source_document=SOURCE))
        # Sentence-level entity extraction from description
        if desc:
            for sentence in re.split(r"(?<=[.!?])\s+", desc):
                sentence = sentence.strip()
                if not sentence:
                    continue
                entities_found = extract_from_sentence(sentence)
                comps = [e["value"] for e in entities_found if e["entity_type"] == "component"]
                faults_found = [e["value"] for e in entities_found if e["entity_type"] == "fault"]
                for c in comps:
                    edge_rows.append(dict(source=eid, source_type="Equipment",
                                          relation="has component", target=c, target_type="Component",
                                          basis="maintenance log", evidence=f"{lid}: {sentence[:120]}"[:260], source_document=SOURCE))
                for f in faults_found:
                    if f.lower() not in fault_node.lower():  # avoid near-duplicates
                        edge_rows.append(dict(source=eid, source_type="Equipment",
                                              relation="experienced fault", target=f, target_type="Fault",
                                              basis="maintenance log", evidence=f"{lid}: {sentence[:120]}"[:260], source_document=SOURCE))
    return edge_rows


def build_edges(entities: pd.DataFrame, maintenance: pd.DataFrame | None, inspections: pd.DataFrame | None,
                maintenance_source: str = "maintenance_records.csv", inspection_source: str = "inspection_records.csv") -> pd.DataFrame:
    rows: list[dict] = []
    if entities is not None and not entities.empty:
        ent = entities
        for r in ent[ent["entity_type"] == "asset_id"].drop_duplicates(["document", "value"]).itertuples(index=False):
            _edge(rows, r.document, "Document", "mentions", r.value, "Equipment", "extracted entity", r.evidence, r.document)
        # sentence-level co-occurrence, single asset per sentence only
        # CSV rows are covered by the structured-record edges below, so only narrative documents
        # contribute (weaker) sentence co-occurrence edges.
        narrative = ent[~ent["document"].str.lower().str.endswith(".csv")]
        for (doc, ev), g in narrative.groupby(["document", "evidence"]):
            assets = g[g["entity_type"] == "asset_id"]["value"].unique()
            if len(assets) != 1:
                continue
            a = assets[0]
            for c in g[g["entity_type"] == "component"]["value"].unique():
                _edge(rows, a, "Equipment", "has component", c, "Component", "sentence co-occurrence", ev, doc)
            for f in g[g["entity_type"] == "fault"]["value"].unique():
                _edge(rows, a, "Equipment", "experienced fault", f, "Fault", "sentence co-occurrence", ev, doc)
                for act in g[g["entity_type"] == "maintenance_action"]["value"].unique():
                    _edge(rows, f, "Fault", "addressed by", act, "Action", "sentence co-occurrence", ev, doc)
        for (doc, ev) in narrative[["document", "evidence"]].drop_duplicates().itertuples(index=False):
            for m in _DRIVE_1.finditer(ev):
                if m.group(1) != m.group(2):
                    _edge(rows, m.group(1).upper(), "Equipment", "driven by", m.group(2).upper(), "Equipment", "sentence pattern", ev, doc)
            for m in _DRIVE_2.finditer(ev):
                _edge(rows, m.group(2).upper(), "Equipment", "driven by", m.group(1).upper(), "Equipment", "sentence pattern", ev, doc)
    if maintenance is not None and not maintenance.empty:
        for w in maintenance.itertuples(index=False):
            ev = f"{w.work_order_id} ({str(w.date)[:10]}): {w.fault_description} -> {getattr(w, 'action_taken', '')}"
            faults = sorted({e["value"] for e in extract_from_sentence(str(w.fault_description)) if e["entity_type"] == "fault"})
            acts = sorted({e["value"] for e in extract_from_sentence(str(getattr(w, "action_taken", ""))) if e["entity_type"] == "maintenance_action"})
            comp = str(getattr(w, "component", "")).strip().lower()
            if comp:
                _edge(rows, w.equipment_id, "Equipment", "has component", comp, "Component", "structured record", ev, maintenance_source)
            for f in faults:
                _edge(rows, w.equipment_id, "Equipment", "experienced fault", f, "Fault", "structured record", ev, maintenance_source)
                for a in acts:
                    _edge(rows, f, "Fault", "addressed by", a, "Action", "structured record", ev, maintenance_source)
    if inspections is not None and not inspections.empty:
        for i in inspections.itertuples(index=False):
            node = f"{i.inspection_id} {i.inspection_type}"
            ev = f"{i.inspection_id} on {str(i.date)[:10]} ({i.status}): {i.finding}"
            _edge(rows, i.equipment_id, "Equipment", "inspected in", node, "Inspection", "structured record", ev, inspection_source)
            _edge(rows, node, "Inspection", "reports", str(i.finding)[:70], "Finding", "structured record", ev, inspection_source)
    # --- Maintenance log DB (structured log records) ---
    ml_rows = _load_maintenance_log_rows()
    if ml_rows:
        rows.extend(_build_maintenance_log_edges(ml_rows))
    if not rows:
        return pd.DataFrame(columns=EDGE_COLS)
    df = pd.DataFrame(rows)
    keys = ["source", "source_type", "relation", "target", "target_type", "basis"]
    agg = df.groupby(keys, as_index=False).agg(evidence=("evidence", "first"), source_document=("source_document", "first"),
                                                support_count=("evidence", "nunique"))
    return agg[EDGE_COLS].sort_values("support_count", ascending=False).reset_index(drop=True)


def to_networkx(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for r in edges.itertuples(index=False):
        g.add_node(r.source, node_type=r.source_type)
        g.add_node(r.target, node_type=r.target_type)
        g.add_edge(r.source, r.target, relation=r.relation, basis=r.basis, support=int(r.support_count), source_document=r.source_document)
    return g


def graph_stats(edges: pd.DataFrame) -> dict:
    if edges.empty:
        return {"nodes": 0, "edges": 0}
    g = to_networkx(edges)
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "components": nx.number_weakly_connected_components(g),
            "most_connected": sorted(g.degree, key=lambda x: -x[1])[:5]}


def subgraph_edges(edges: pd.DataFrame, equipment: str | None = None, relations: list[str] | None = None, min_support: int = 1) -> pd.DataFrame:
    e = edges[edges["support_count"] >= min_support]
    if relations:
        e = e[e["relation"].isin(relations)]
    if equipment and equipment != "All":
        keep = set(e[(e["source"] == equipment) | (e["target"] == equipment)].index)
        nodes = {equipment} | set(e.loc[list(keep), "source"]) | set(e.loc[list(keep), "target"])
        # two-hop neighbourhood: fault -> action edges of the faults attached to this asset
        e = e[(e["source"].isin(nodes)) | (e["target"].isin(nodes))]
        e = e[(e["source"] == equipment) | (e["target"] == equipment) | (e["source_type"].isin(["Fault", "Inspection"]) & e["source"].isin(nodes))]
    return e


def render_pyvis_html(edges: pd.DataFrame, height_px: int = 620) -> str | None:
    """Interactive graph as standalone HTML (resources inlined so it works offline). None if PyVis is missing."""
    try:
        from pyvis.network import Network
    except ImportError:
        return None
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#0e1a30", font_color="#e8eef7", cdn_resources="in_line")
    nodes = {}
    for r in edges.itertuples(index=False):
        nodes[r.source], nodes[r.target] = r.source_type, r.target_type
    deg = pd.concat([edges["source"], edges["target"]]).value_counts()
    for n, t in nodes.items():
        # all text comes from user documents -> escape before it reaches the HTML page
        net.add_node(n, label=_html.escape(n if len(n) < 28 else n[:26] + "..."), title=_html.escape(f"{t}: {n}"), color=COLORS.get(t, "#999"),
                     size=12 + 3 * min(int(deg.get(n, 1)), 10), shape="dot" if t != "Document" else "square")
    for r in edges.itertuples(index=False):
        net.add_edge(r.source, r.target, label=_html.escape(r.relation), title=_html.escape(f"{r.relation} [{r.basis}] support={r.support_count} | {r.source_document} | {r.evidence}"),
                     color="#5f7da6", dashes=r.basis == "sentence co-occurrence", arrows="to")
    net.set_options('{"physics":{"barnesHut":{"gravity":-9000,"springLength":140}},"edges":{"font":{"size":9,"color":"#b8c7dd","strokeWidth":0},"smooth":{"type":"continuous"}},"interaction":{"hover":true,"tooltipDelay":120}}')
    return net.generate_html()


def plotly_fallback(edges: pd.DataFrame):
    """Static NetworkX layout drawn with Plotly (used when PyVis is unavailable)."""
    import plotly.graph_objects as go
    g = to_networkx(edges)
    pos = nx.spring_layout(g, seed=42, k=0.6)
    ex, ey = [], []
    for a, b in g.edges():
        ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.7, color="#5f7da6"), hoverinfo="none"))
    for t, col in COLORS.items():
        ns = [n for n, d in g.nodes(data=True) if d.get("node_type") == t]
        if ns:
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=t, text=[n[:18] for n in ns],
                                     textposition="top center", marker=dict(size=11, color=col), hovertext=ns, hoverinfo="text"))
    fig.update_layout(showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False), height=620)
    return fig
