"""Industrial Knowledge Intelligence — Unified Asset & Operations Brain.

Run:  streamlit run app.py

Authentication is enforced here and inside every page via src.auth.require_auth().
"""
from __future__ import annotations

import logging
import os

# Suppress Windows huggingface_hub symlinks warning (non-fatal)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon=":material/hub:",
    layout="wide",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from src import storage                   # noqa: E402
from src.auth import init_db, logout, validate_session  # noqa: E402
from src.compliance_checker import summarize  # noqa: E402
from src.ui import (                      # noqa: E402
    COMPLIANCE_COLORS,
    inject_css,
    kpi_row,
    notice,
    page_header,
    require_data,
    style_fig,
)

inject_css()
init_db()

# ---------------------------------------------------------------------------
# Auth gate — validate session token against DB on every run
# ---------------------------------------------------------------------------
_token = st.session_state.get("_auth_token")
_user  = validate_session(_token)

if _user:
    # Keep session_state in sync with validated DB record
    st.session_state["_auth_token"]    = _user["token"]
    st.session_state["_auth_user"]     = _user["user_id"]
    st.session_state["_auth_username"] = _user["username"]

# ---------------------------------------------------------------------------
# Helper: logout action (sidebar button)
# ---------------------------------------------------------------------------
def _do_logout() -> None:
    token = st.session_state.pop("_auth_token", None)
    st.session_state.pop("_auth_user",     None)
    st.session_state.pop("_auth_username", None)
    logout(token)


# ---------------------------------------------------------------------------
# Dashboard (protected)
# ---------------------------------------------------------------------------
def dashboard() -> None:
    # Auth guard — validated inside page_header flow; also guard here explicitly
    from src.auth import require_auth
    require_auth()

    kb = page_header(
        "Dashboard",
        "One workspace for industrial documents, maintenance insights, and operational knowledge.",
    )

    # Hero banner
    st.markdown(
        '<div class="dashboard-hero">'
        '<h1>Industrial Knowledge Intelligence</h1>'
        '<p>One workspace for industrial documents, maintenance insights, and operational knowledge.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not require_data(kb):
        col_a, col_b = st.columns(2)
        with col_a, st.container(border=True):
            st.subheader("📂 Load sample data")
            st.write("Load the built-in sample industrial documents to explore the application.")
            if st.button("Load sample dataset", type="primary", key="dash_load_samples"):
                with st.spinner("Loading…"):
                    kb.load_samples()
                st.rerun()
        with col_b, st.container(border=True):
            st.subheader("📤 Upload documents")
            st.write("Go to **Document Manager** to upload your own PDF, DOCX, TXT, or CSV files.")
        return

    s   = kb.stats()
    log = storage.load_table("search_log")

    kpi_row([
        ("Documents",           s["documents"],              "processed & indexed",     False),
        ("Equipment assets",    s["assets"],                 "unique IDs identified",   False),
        ("Maintenance records", s["incidents"],              "work orders analysed",    False),
        ("Knowledge entities",  f"{s['entities']:,}",        "rule-based + lexicon",    False),
        ("Compliance gaps",     s["compliance_gaps"],        "illustrative rules",      True),
    ])

    col_qa, col_qb = st.columns(2)
    with col_qa:
        if st.button("📤 Upload documents", use_container_width=True):
            st.switch_page("pages/1_Document_Manager.py")
    with col_qb:
        if st.button("🤖 Ask AI assistant", use_container_width=True):
            st.switch_page("pages/3_Knowledge_Assistant.py")

    cl   = kb.clustering()
    m, _ = kb.maintenance()

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        if not m.empty:
            top = m["equipment_id"].value_counts().head(8).reset_index()
            top.columns = ["equipment_id", "work_orders"]
            fig = px.bar(
                top, x="equipment_id", y="work_orders",
                title="Most frequently reported equipment faults",
                color="work_orders",
                color_continuous_scale=["#DCEEFF", "#4A90D9"],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        else:
            notice("No maintenance records available.", "info")

    with c2, st.container(border=True):
        if cl:
            d = cl["summary"].sort_values("records")
            fig = px.bar(
                d, x="records", y="fault_category", orientation="h",
                title="Fault categories (K-Means clusters)",
                color="records",
                color_continuous_scale=["#DDF5E5", "#52B788"],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        else:
            notice("No maintenance records available for clustering.", "info")

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        if cl and not m.empty:
            from src.maintenance_analyzer import monthly_trend
            tr  = monthly_trend(m)
            fig = px.line(
                tr, x="month", y="work_orders", markers=True,
                title="Maintenance trend (work orders per month)",
                line_shape="spline",
            )
            fig.update_traces(
                line=dict(color="#4A90D9", width=2.5),
                marker=dict(color="#4A90D9", size=6),
            )
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            notice("Not enough date data for trend chart.", "info")

    with c4, st.container(border=True):
        comp = kb.compliance()
        sm   = summarize(comp)
        if sm["total"]:
            cc = pd.Series(sm["by_status"]).reset_index()
            cc.columns = ["status", "checks"]
            fig = px.pie(
                cc, names="status", values="checks",
                hole=.58,
                title="Compliance status (illustrative rules)",
                color="status",
                color_discrete_map=COMPLIANCE_COLORS,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            notice("No compliance data available.", "info")

    with st.container(border=True):
        st.subheader("Recent documents")
        from src.ui import STATUS_TEXT
        d = kb.documents.copy().tail(8).iloc[::-1]
        d["status"] = d["status"].map(lambda x: STATUS_TEXT.get(x, x))
        st.dataframe(
            d[["name", "file_type", "status", "n_pages", "text_length", "n_chunks"]].rename(
                columns={"n_pages": "pages", "text_length": "chars", "n_chunks": "chunks"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "All figures are computed from loaded sample / synthetic data. "
        "This is an academic prototype, not a system for controlling industrial equipment."
    )


# ---------------------------------------------------------------------------
# Navigation — conditional on authentication
# ---------------------------------------------------------------------------
if _user:
    # ---- Sidebar: user info + logout ----------------------------------------
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-user-badge">'
            f'<span class="sidebar-user-icon">👤</span>'
            f'<span class="sidebar-user-name">{_user["username"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("🚪 Sign Out", key="logout_btn", use_container_width=True):
            _do_logout()
            st.rerun()
        st.divider()

    # ---- Protected pages ---------------------------------------------------
    nav = st.navigation([
        st.Page(dashboard,
                title="Dashboard",            icon=":material/dashboard:",   default=True, url_path="dashboard"),
        st.Page("pages/1_Document_Manager.py",
                title="Document Manager",     icon=":material/folder_open:"),
        st.Page("pages/8_Neural_Search.py",
                title="Neural Search",        icon=":material/search:"),
        st.Page("pages/3_Knowledge_Assistant.py",
                title="AI Knowledge Assistant", icon=":material/forum:"),
        st.Page("pages/4_Knowledge_Graph.py",
                title="Knowledge Graph",      icon=":material/hub:"),
        st.Page("pages/7_Maintenance_Log.py",
                title="Maintenance Log",      icon=":material/assignment:"),
        st.Page("pages/5_Maintenance_Intelligence.py",
                title="Maintenance Intelligence", icon=":material/build:"),
        st.Page("pages/6_Compliance_Evaluation.py",
                title="Compliance & Evaluation",  icon=":material/fact_check:"),
    ])
    nav.run()

else:
    # ---- Unauthenticated: show only Login and Register ---------------------
    nav = st.navigation([
        st.Page("pages/0_Login.py",
                title="Login",    icon=":material/login:",    default=True, url_path="login"),
        st.Page("pages/00_Register.py",
                title="Sign Up",  icon=":material/person_add:"),
    ])
    nav.run()
