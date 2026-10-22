"""Maintenance Log page — create, view, filter, and update equipment fault records."""
from __future__ import annotations

import streamlit as st

from src.auth import require_auth
user = require_auth()
username = user["username"]

import pandas as pd
from datetime import datetime, date

from src.maintenance_log import (
    SEVERITY_OPTS, STATUS_OPTS,
    create_log, get_log, get_audit_trail, get_attachments,
    get_attachment_bytes, list_logs, save_attachment, summary_stats,
    update_log, init_db,
)
from src.ui import inject_css, kpi_row, notice, page_header

inject_css()
init_db()

page_header(
    "Maintenance Log",
    "Record, track, and resolve equipment faults and maintenance actions",
)

# ===========================================================================
# Tabs
# ===========================================================================
tab_dash, tab_new, tab_detail = st.tabs([
    "📋 Log Dashboard", "➕ New Fault Report", "🔍 View / Update Record"
])

# ===========================================================================
# TAB 1: DASHBOARD
# ===========================================================================
with tab_dash:
    stats = summary_stats()
    kpi_row([
        ("Total logs",      stats["total"],         "all records",          False),
        ("Pending",         stats["pending"],        "awaiting action",      True),
        ("In Progress",     stats["in_progress"],   "being worked on",      False),
        ("Resolved/Closed", stats["resolved"],      "completed",            False),
        ("Critical open",   stats["critical_open"], "unresolved critical",  True),
    ])

    # ---- Filters -----------------------------------------------------------
    with st.container(border=True):
        st.subheader("🔎 Filter logs")
        fc1, fc2, fc3 = st.columns(3)
        f_equip    = fc1.text_input("Equipment ID", placeholder="e.g. P-101")
        f_status   = fc2.selectbox("Status",   ["All"] + STATUS_OPTS,   index=0)
        f_severity = fc3.selectbox("Severity", ["All"] + SEVERITY_OPTS, index=0)
        fd1, fd2, fd3, fd4 = st.columns(4)
        f_from     = fd1.date_input("From date", value=None, key="ml_from")
        f_to       = fd2.date_input("To date",   value=None, key="ml_to")
        f_reporter = fd3.text_input("Reporter",     placeholder="username")
        f_assigned = fd4.text_input("Assigned to",  placeholder="technician")

    df = list_logs(
        equipment_id=f_equip,
        status=""      if f_status   == "All" else f_status,
        severity=""    if f_severity == "All" else f_severity,
        date_from=str(f_from) if f_from else "",
        date_to=str(f_to)     if f_to   else "",
        reporter=f_reporter,
        assigned_to=f_assigned,
    )

    if df.empty:
        notice("No maintenance logs found matching the current filters.", "info")
    else:
        st.caption(f"{len(df)} record(s)")
        display = df[[
            "log_id", "equipment_id", "fault_title", "observed_at",
            "reported_by", "assigned_to", "severity", "status", "updated_at",
        ]].rename(columns={
            "log_id":       "Log ID",
            "equipment_id": "Equipment",
            "fault_title":  "Fault",
            "observed_at":  "Observed",
            "reported_by":  "Reporter",
            "assigned_to":  "Assigned to",
            "severity":     "Severity",
            "status":       "Status",
            "updated_at":   "Last updated",
        })
        # Severity colour helper — use pandas Styler.
        # IMPORTANT: always include an explicit color so AG-Grid's internal
        # theme cannot override text to near-white on pastel backgrounds.
        _SEV_BG = {
            "Critical": "#fde8e8",
            "High":     "#fef3cd",
            "Medium":   "#e8f4fd",
            "Low":      "#e8f8e8",
        }
        _TEXT_DARK = "#1e2d42"  # readable dark navy on all pastel bg colours

        def _row_style(row):
            bg = _SEV_BG.get(row.get("Severity", ""), "")
            if bg:
                cell_css = f"background-color:{bg};color:{_TEXT_DARK};font-weight:500"
            else:
                cell_css = f"color:{_TEXT_DARK}"
            return [cell_css for _ in row]

        st.dataframe(
            display.style.apply(_row_style, axis=1),
            hide_index=True, use_container_width=True,
        )

        # Quick open by log ID
        st.info("To open a record, paste its **Log ID** into the **View / Update Record** tab.")

# ===========================================================================
# TAB 2: NEW FAULT REPORT
# ===========================================================================
with tab_new:
    st.subheader("📝 New Fault Report")
    with st.form("new_log_form", clear_on_submit=True):
        n1, n2 = st.columns(2)
        equip_id   = n1.text_input("Equipment / Asset ID *", placeholder="e.g. P-101, M-401")
        equip_name = n2.text_input("Equipment name",         placeholder="e.g. Centrifugal Pump")

        fault_title = st.text_input("Fault / Issue title *",
                                    placeholder="Brief one-line description of the fault")

        description = st.text_area("Detailed description *",
                                   placeholder="What happened? What is broken or abnormal?",
                                   height=110)
        how_occurred = st.text_area("How the fault occurred",
                                    placeholder="Circumstances, symptoms observed, sequence of events…",
                                    height=90)

        d1, d2, d3 = st.columns(3)
        observed_date = d1.date_input("Date observed *", value=date.today())
        observed_time = d2.time_input("Time observed *", value=datetime.now().time())
        severity      = d3.selectbox("Severity *", SEVERITY_OPTS, index=1)

        a1, a2 = st.columns(2)
        action_taken = a1.text_area("Action taken so far",
                                    placeholder="Troubleshooting steps, temporary fixes…",
                                    height=90)
        assigned_to  = a2.text_input("Assign to technician",
                                     placeholder="Name or username")

        status = st.selectbox("Initial status *", STATUS_OPTS, index=0)

        # Resolution fields — only shown when relevant
        if status in ("Resolved", "Closed"):
            r1, r2 = st.columns(2)
            resolution  = r1.text_area("Resolution details *",
                                       placeholder="What was done to fix the fault?",
                                       height=90)
            resolved_at = r2.date_input("Date resolved", value=date.today())
        else:
            resolution  = ""
            resolved_at = None

        attachments = st.file_uploader(
            "Attach files (photos, reports, logs)",
            accept_multiple_files=True,
        )

        submitted = st.form_submit_button("💾 Save Fault Report", type="primary")

    if submitted:
        obs_dt = datetime.combine(observed_date, observed_time).isoformat()
        res_dt = resolved_at.isoformat() if resolved_at and status in ("Resolved", "Closed") else None

        result = create_log({
            "equipment_id":   equip_id,
            "equipment_name": equip_name,
            "fault_title":    fault_title,
            "description":    description,
            "how_occurred":   how_occurred,
            "observed_at":    obs_dt,
            "reported_by":    username,
            "severity":       severity,
            "action_taken":   action_taken,
            "assigned_to":    assigned_to,
            "resolution":     resolution,
            "status":         status,
            "resolved_at":    res_dt,
        }, username=username)

        if result["ok"]:
            log_id = result["log_id"]
            # Save attachments
            for f in (attachments or []):
                save_attachment(log_id, f.name, f.read(), username)
            st.success(f"✅ Fault report saved — **{log_id}**. Use the View/Update tab to track it.")
            st.rerun()
        else:
            st.error(result["error"])

# ===========================================================================
# TAB 3: VIEW / UPDATE RECORD
# ===========================================================================
with tab_detail:
    st.subheader("🔍 View & Update a Maintenance Record")
    log_id_input = st.text_input("Enter Log ID", placeholder="e.g. ML-A1B2C3D4",
                                  key="detail_log_id")

    if log_id_input.strip():
        rec = get_log(log_id_input.strip().upper())
        if not rec:
            st.error(f"Log **{log_id_input.strip().upper()}** not found.")
        else:
            # ---- Display record ------------------------------------------
            with st.container(border=True):
                h1, h2, h3 = st.columns([3, 1, 1])
                h1.markdown(f"### {rec['fault_title']}")
                sev_clr = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                h2.metric("Severity", f"{sev_clr.get(rec['severity'],'')} {rec['severity']}")
                h3.metric("Status", rec["status"])

                st.caption(
                    f"**Log ID:** {rec['log_id']} · "
                    f"**Equipment:** {rec['equipment_id']} {rec['equipment_name']} · "
                    f"**Reported by:** {rec['reported_by']} on {rec['observed_at'][:10]} · "
                    f"**Assigned to:** {rec['assigned_to'] or '—'}"
                )
                st.divider()

                i1, i2 = st.columns(2)
                with i1:
                    st.markdown("**Description**")
                    st.write(rec["description"] or "—")
                    st.markdown("**How it occurred**")
                    st.write(rec["how_occurred"] or "—")
                with i2:
                    st.markdown("**Action taken**")
                    st.write(rec["action_taken"] or "—")
                    st.markdown("**Resolution**")
                    st.write(rec["resolution"] or "—")
                if rec.get("resolved_at"):
                    st.caption(f"Resolved at: {rec['resolved_at']}")

            # ---- Update form ----------------------------------------------
            with st.expander("✏️ Update this record", expanded=False):
                with st.form(f"update_{rec['log_id']}"):
                    u1, u2 = st.columns(2)
                    new_status   = u1.selectbox("Status",   STATUS_OPTS,
                                                index=STATUS_OPTS.index(rec["status"]))
                    new_severity = u2.selectbox("Severity", SEVERITY_OPTS,
                                                index=SEVERITY_OPTS.index(rec["severity"]))
                    new_assigned = st.text_input("Assigned to", value=rec.get("assigned_to", ""))
                    new_action   = st.text_area("Update action taken",
                                                value=rec.get("action_taken", ""), height=90)
                    new_resolution = st.text_area(
                        "Resolution details" + (" *" if new_status in ("Resolved", "Closed") else ""),
                        value=rec.get("resolution", ""), height=90,
                    )
                    new_res_date = None
                    if new_status in ("Resolved", "Closed"):
                        new_res_date = st.date_input(
                            "Date resolved",
                            value=date.today() if not rec.get("resolved_at") else
                                  date.fromisoformat(rec["resolved_at"][:10]),
                        )

                    note_col, btn_col = st.columns([3, 1])
                    with btn_col:
                        do_update = st.form_submit_button("💾 Save changes", type="primary")

                if do_update:
                    upd = {
                        "status":      new_status,
                        "severity":    new_severity,
                        "assigned_to": new_assigned,
                        "action_taken":new_action,
                        "resolution":  new_resolution,
                        "resolved_at": new_res_date.isoformat() if new_res_date and new_status in ("Resolved","Closed") else rec.get("resolved_at"),
                    }
                    r = update_log(rec["log_id"], upd, username)
                    if r["ok"]:
                        st.success("Record updated.")
                        st.rerun()
                    else:
                        st.error(r["error"])

            # ---- Attachments ---------------------------------------------
            atts = get_attachments(rec["log_id"])
            if atts:
                with st.expander(f"📎 Attachments ({len(atts)})", expanded=False):
                    for a in atts:
                        col_n, col_d = st.columns([3, 1])
                        col_n.write(f"📄 {a['filename']} — uploaded by {a['uploaded_by']}")
                        result = get_attachment_bytes(a["attachment_id"])
                        if result:
                            col_d.download_button(
                                "Download", data=result[1], file_name=result[0],
                                key=f"dl_{a['attachment_id']}",
                            )

            # ---- Audit trail ---------------------------------------------
            audit = get_audit_trail(rec["log_id"])
            if not audit.empty:
                with st.expander(f"🕐 Audit trail ({len(audit)} changes)", expanded=False):
                    audit["changed_at"] = audit["changed_at"].str[:19].str.replace("T", " ")
                    st.dataframe(
                        audit[["changed_at","changed_by","field_name","old_value","new_value"]].rename(
                            columns={"changed_at":"When","changed_by":"Who",
                                     "field_name":"Field","old_value":"Old","new_value":"New"}
                        ),
                        hide_index=True, use_container_width=True,
                    )
