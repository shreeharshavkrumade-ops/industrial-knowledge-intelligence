"""Illustrative compliance checking against FICTIONAL requirements (not legal advice).

Each result row = (requirement x asset) with status, evidence found and its source document.
Statuses: Compliant | Overdue | Missing evidence | Documentation gap.
"""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd

DISCLAIMER = ("All requirements shown are fictional, illustrative examples for an academic prototype. "
              "They are not verified legal or regulatory advice. Keyword evidence in documents does not prove a procedure is adequate.")
GAP_STATUSES = {"Overdue", "Missing evidence", "Documentation gap"}
INSPECTION_SOURCE = "inspection_records.csv"


def _applies(applies_to: str, etype: str) -> bool:
    items = [s.strip().lower() for s in str(applies_to).split(";")]
    return "all" in items or str(etype).lower() in items


def run_compliance(requirements: pd.DataFrame, inspections: pd.DataFrame, maintenance: pd.DataFrame,
                   assets: pd.DataFrame, chunks: pd.DataFrame, as_of: date) -> pd.DataFrame:
    cols = ["requirement_id", "title", "check_type", "equipment_id", "status", "evidence", "source_document", "days_since_last", "limit_days", "note"]
    if requirements is None or requirements.empty:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(as_of)
    insp = inspections.copy() if inspections is not None else pd.DataFrame()
    if not insp.empty:
        insp["date"] = pd.to_datetime(insp["date"], errors="coerce")
        insp = insp.dropna(subset=["date"])
        insp = insp[insp["date"] <= asof]
    mnt = maintenance.copy() if maintenance is not None else pd.DataFrame()
    if not mnt.empty:
        mnt["date"] = pd.to_datetime(mnt["date"], errors="coerce")
    rows = []
    for req in requirements.itertuples(index=False):
        base = dict(requirement_id=req.requirement_id, title=req.title, check_type=req.check_type)
        if req.check_type == "periodic_inspection":
            limit = int(req.max_interval_days)
            for a in assets.itertuples(index=False):
                if not _applies(req.applies_to, a.equipment_type):
                    continue
                sub = insp[(insp["equipment_id"] == a.equipment_id) & (insp["inspection_type"] == req.inspection_type)] if not insp.empty else insp
                if sub.empty:
                    rows.append({**base, "equipment_id": a.equipment_id, "status": "Missing evidence", "evidence": "No matching inspection record found",
                                 "source_document": INSPECTION_SOURCE, "days_since_last": None, "limit_days": limit,
                                 "note": f"Requires '{req.inspection_type}' at least every {limit} days."})
                    continue
                last = sub.sort_values("date").iloc[-1]
                age = int((asof - last["date"]).days)
                status = "Compliant" if age <= limit else "Overdue"
                rows.append({**base, "equipment_id": a.equipment_id, "status": status,
                             "evidence": f"{last['inspection_id']} on {last['date'].date()}: {last['status']} - {last['finding']}",
                             "source_document": INSPECTION_SOURCE, "days_since_last": age, "limit_days": limit,
                             "note": f"Due by {(last['date'] + pd.Timedelta(days=limit)).date()}"})
        elif req.check_type == "follow_up":
            limit = int(req.max_interval_days)
            fails = insp[insp["status"].astype(str).str.lower() == "fail"] if not insp.empty else insp
            for f in fails.itertuples(index=False):
                win = mnt[(mnt["equipment_id"] == f.equipment_id) & (mnt["date"] >= f.date) & (mnt["date"] <= f.date + pd.Timedelta(days=limit))] if not mnt.empty else mnt
                if win.empty:
                    rows.append({**base, "equipment_id": f.equipment_id, "status": "Documentation gap",
                                 "evidence": f"{f.inspection_id} ({f.date.date()}) failed: {f.finding}; no work order within {limit} days",
                                 "source_document": INSPECTION_SOURCE, "days_since_last": int((asof - f.date).days), "limit_days": limit,
                                 "note": "A follow-up work order may exist but is not documented in the provided records."})
                else:
                    w = win.sort_values("date").iloc[0]
                    rows.append({**base, "equipment_id": f.equipment_id, "status": "Compliant",
                                 "evidence": f"{f.inspection_id} failed on {f.date.date()}; follow-up {w['work_order_id']} on {w['date'].date()}",
                                 "source_document": "maintenance_records.csv", "days_since_last": int((w["date"] - f.date).days), "limit_days": limit, "note": ""})
        elif req.check_type == "documented_procedure":
            kws = [k.strip() for k in str(req.evidence_keywords).split(";") if k.strip()]
            hit, doc, snippet = False, "-", ""
            if chunks is not None and not chunks.empty and kws:
                rx = re.compile("|".join(re.escape(k) for k in kws), re.I)
                narrative = chunks[~chunks["document"].str.endswith(".csv")]
                for c in narrative.itertuples(index=False):
                    m = rx.search(str(c.text))
                    if m:
                        hit, doc = True, c.document
                        s = max(0, m.start() - 80)
                        snippet = "..." + str(c.text)[s:m.end() + 120] + "..."
                        break
            rows.append({**base, "equipment_id": "(plant-wide)", "status": "Compliant" if hit else "Missing evidence",
                         "evidence": snippet if hit else f"No document mentions: {', '.join(kws)}", "source_document": doc,
                         "days_since_last": None, "limit_days": None,
                         "note": "Keyword match only; adequacy not assessed." if hit else "Upload a procedure document to close this gap."})
    return pd.DataFrame(rows, columns=cols)


def summarize(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"total": 0, "compliant": 0, "gaps": 0, "by_status": {}}
    vc = results["status"].value_counts().to_dict()
    return {"total": len(results), "compliant": int(vc.get("Compliant", 0)),
            "gaps": int(results["status"].isin(GAP_STATUSES).sum()), "by_status": vc}


def audit_pdf_bytes(results: pd.DataFrame, as_of: date) -> bytes | None:
    """Single downloadable audit summary (PDF). Returns None if reportlab is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    st = getSampleStyleSheet()
    s = summarize(results)
    story = [Paragraph("Illustrative Compliance Audit Summary", st["Title"]),
             Paragraph(f"As-of date: {as_of} &nbsp; | &nbsp; Checks: {s['total']} &nbsp; | &nbsp; Compliant: {s['compliant']} &nbsp; | &nbsp; Gaps: {s['gaps']}", st["Normal"]),
             Spacer(1, 6), Paragraph(DISCLAIMER, st["Italic"]), Spacer(1, 10)]
    small = st["BodyText"].clone("small", fontSize=7, leading=8.5)
    data = [["Req", "Asset", "Status", "Evidence", "Source"]]
    for r in results.itertuples(index=False):
        data.append([r.requirement_id, r.equipment_id, r.status, Paragraph(str(r.evidence)[:220], small), Paragraph(str(r.source_document), small)])
    t = Table(data, colWidths=[40, 70, 90, 470, 120], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14284b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()
