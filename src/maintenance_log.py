"""Persistent Maintenance Log — storage, validation, and audit trail.

All writes use parameterised SQL. Audit records capture who made each change.
No plaintext passwords or session tokens are stored in this module.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LOG = logging.getLogger("iki.maintenance_log")

ROOT   = Path(__file__).resolve().parents[1]
ML_DB  = ROOT / "data" / "maintenance_log.db"

SEVERITY_OPTS = ["Low", "Medium", "High", "Critical"]
STATUS_OPTS   = ["Pending", "In Progress", "Resolved", "Closed"]

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _connect():
    ML_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ML_DB), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create maintenance log tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                log_id            TEXT PRIMARY KEY,
                equipment_id      TEXT NOT NULL,
                equipment_name    TEXT NOT NULL DEFAULT '',
                fault_title       TEXT NOT NULL,
                description       TEXT NOT NULL DEFAULT '',
                how_occurred      TEXT NOT NULL DEFAULT '',
                observed_at       TEXT NOT NULL,
                reported_by       TEXT NOT NULL,
                severity          TEXT NOT NULL DEFAULT 'Medium',
                action_taken      TEXT NOT NULL DEFAULT '',
                assigned_to       TEXT NOT NULL DEFAULT '',
                resolution        TEXT NOT NULL DEFAULT '',
                status            TEXT NOT NULL DEFAULT 'Pending',
                resolved_at       TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                created_by        TEXT NOT NULL,
                updated_by        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS maintenance_log_attachments (
                attachment_id     TEXT PRIMARY KEY,
                log_id            TEXT NOT NULL REFERENCES maintenance_logs(log_id) ON DELETE CASCADE,
                filename          TEXT NOT NULL,
                file_bytes        BLOB NOT NULL,
                uploaded_by       TEXT NOT NULL,
                uploaded_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS maintenance_log_audit (
                audit_id          TEXT PRIMARY KEY,
                log_id            TEXT NOT NULL REFERENCES maintenance_logs(log_id) ON DELETE CASCADE,
                changed_by        TEXT NOT NULL,
                changed_at        TEXT NOT NULL,
                field_name        TEXT NOT NULL,
                old_value         TEXT,
                new_value         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ml_equipment ON maintenance_logs(equipment_id);
            CREATE INDEX IF NOT EXISTS idx_ml_status    ON maintenance_logs(status);
            CREATE INDEX IF NOT EXISTS idx_ml_severity  ON maintenance_logs(severity);
            CREATE INDEX IF NOT EXISTS idx_ml_created   ON maintenance_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_log    ON maintenance_log_audit(log_id);
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "ML") -> str:
    return f"{prefix}-{secrets.token_hex(4).upper()}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_log(data: dict) -> list[str]:
    """Return list of validation error messages. Empty = valid."""
    errs = []
    if not str(data.get("equipment_id", "")).strip():
        errs.append("Equipment / Asset ID is required.")
    if not str(data.get("fault_title", "")).strip():
        errs.append("Fault title is required.")
    if not str(data.get("observed_at", "")).strip():
        errs.append("Observation date/time is required.")
    if data.get("severity") not in SEVERITY_OPTS:
        errs.append(f"Severity must be one of: {', '.join(SEVERITY_OPTS)}.")
    if data.get("status") not in STATUS_OPTS:
        errs.append(f"Status must be one of: {', '.join(STATUS_OPTS)}.")
    if data.get("status") in ("Resolved", "Closed") and not str(data.get("resolution", "")).strip():
        errs.append("Resolution details are required when status is Resolved or Closed.")
    return errs


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_log(data: dict, username: str) -> dict[str, Any]:
    """Insert a new maintenance log. Returns {ok, log_id, error}."""
    errs = validate_log(data)
    if errs:
        return {"ok": False, "error": "; ".join(errs)}

    log_id = _new_id()
    now    = _now()
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO maintenance_logs
                   (log_id, equipment_id, equipment_name, fault_title, description,
                    how_occurred, observed_at, reported_by, severity, action_taken,
                    assigned_to, resolution, status, resolved_at,
                    created_at, updated_at, created_by, updated_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (log_id,
                 data["equipment_id"].strip(),
                 data.get("equipment_name", "").strip(),
                 data["fault_title"].strip(),
                 data.get("description", "").strip(),
                 data.get("how_occurred", "").strip(),
                 data["observed_at"],
                 data.get("reported_by", username).strip() or username,
                 data["severity"],
                 data.get("action_taken", "").strip(),
                 data.get("assigned_to", "").strip(),
                 data.get("resolution", "").strip(),
                 data["status"],
                 data.get("resolved_at"),
                 now, now, username, username)
            )
        LOG.info("Maintenance log created: %s by %s", log_id, username)
        return {"ok": True, "log_id": log_id}
    except Exception as exc:
        LOG.exception("Failed to create log")
        return {"ok": False, "error": str(exc)}


def update_log(log_id: str, updates: dict, username: str) -> dict[str, Any]:
    """Update fields of an existing log and record audit trail."""
    errs = validate_log({**get_log(log_id), **updates})
    if errs:
        return {"ok": False, "error": "; ".join(errs)}

    now = _now()
    try:
        with _connect() as conn:
            # Fetch current values for audit
            old = conn.execute(
                "SELECT * FROM maintenance_logs WHERE log_id=?", (log_id,)
            ).fetchone()
            if old is None:
                return {"ok": False, "error": f"Log {log_id} not found."}

            # Build SET clause from updates (whitelist)
            allowed = {
                "equipment_id", "equipment_name", "fault_title", "description",
                "how_occurred", "observed_at", "reported_by", "severity",
                "action_taken", "assigned_to", "resolution", "status", "resolved_at",
            }
            filtered = {k: v for k, v in updates.items() if k in allowed}
            if not filtered:
                return {"ok": True, "log_id": log_id}  # nothing to do

            set_clause = ", ".join(f"{k}=?" for k in filtered)
            values     = list(filtered.values()) + [now, username, log_id]
            conn.execute(
                f"UPDATE maintenance_logs SET {set_clause}, updated_at=?, updated_by=? WHERE log_id=?",
                values
            )

            # Audit trail
            for field, new_val in filtered.items():
                old_val = old[field] if old[field] is not None else ""
                if str(old_val) != str(new_val if new_val is not None else ""):
                    conn.execute(
                        "INSERT INTO maintenance_log_audit"
                        "(audit_id,log_id,changed_by,changed_at,field_name,old_value,new_value)"
                        "VALUES(?,?,?,?,?,?,?)",
                        (_new_id("AU"), log_id, username, now, field, str(old_val), str(new_val))
                    )

        LOG.info("Maintenance log updated: %s by %s", log_id, username)
        return {"ok": True, "log_id": log_id}
    except Exception as exc:
        LOG.exception("Failed to update log %s", log_id)
        return {"ok": False, "error": str(exc)}


def get_log(log_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM maintenance_logs WHERE log_id=?", (log_id,)
        ).fetchone()
    return dict(row) if row else {}


def get_audit_trail(log_id: str) -> pd.DataFrame:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT audit_id, changed_by, changed_at, field_name, old_value, new_value "
            "FROM maintenance_log_audit WHERE log_id=? ORDER BY changed_at",
            (log_id,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def list_logs(
    equipment_id: str = "",
    status: str = "",
    severity: str = "",
    date_from: str = "",
    date_to: str = "",
    reporter: str = "",
    assigned_to: str = "",
) -> pd.DataFrame:
    """Return filtered maintenance logs as a DataFrame."""
    clauses, params = [], []
    if equipment_id:
        clauses.append("LOWER(equipment_id) LIKE ?")
        params.append(f"%{equipment_id.lower()}%")
    if status:
        clauses.append("status=?")
        params.append(status)
    if severity:
        clauses.append("severity=?")
        params.append(severity)
    if date_from:
        clauses.append("observed_at >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("observed_at <= ?")
        params.append(date_to + "T23:59:59")
    if reporter:
        clauses.append("LOWER(reported_by) LIKE ?")
        params.append(f"%{reporter.lower()}%")
    if assigned_to:
        clauses.append("LOWER(assigned_to) LIKE ?")
        params.append(f"%{assigned_to.lower()}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM maintenance_logs {where} ORDER BY created_at DESC",
            params
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def save_attachment(log_id: str, filename: str, data: bytes, username: str) -> str:
    att_id = _new_id("AT")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO maintenance_log_attachments"
            "(attachment_id,log_id,filename,file_bytes,uploaded_by,uploaded_at)"
            "VALUES(?,?,?,?,?,?)",
            (att_id, log_id, filename, data, username, _now())
        )
    return att_id


def get_attachments(log_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT attachment_id, filename, uploaded_by, uploaded_at "
            "FROM maintenance_log_attachments WHERE log_id=?",
            (log_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_attachment_bytes(attachment_id: str) -> tuple[str, bytes] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename, file_bytes FROM maintenance_log_attachments WHERE attachment_id=?",
            (attachment_id,)
        ).fetchone()
    return (row["filename"], bytes(row["file_bytes"])) if row else None


def summary_stats() -> dict:
    with _connect() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM maintenance_logs WHERE status='Pending'").fetchone()[0]
        inprog   = conn.execute("SELECT COUNT(*) FROM maintenance_logs WHERE status='In Progress'").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM maintenance_logs WHERE status IN ('Resolved','Closed')").fetchone()[0]
        critical = conn.execute("SELECT COUNT(*) FROM maintenance_logs WHERE severity='Critical' AND status NOT IN ('Resolved','Closed')").fetchone()[0]
    return {"total": total, "pending": pending, "in_progress": inprog,
            "resolved": resolved, "critical_open": critical}


def count_logs() -> int:
    """Return the total number of maintenance log records (0 if DB doesn't exist)."""
    if not ML_DB.exists():
        return 0
    try:
        with _connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0]
    except Exception:
        return 0


def logs_as_text_chunks(limit: int = 500) -> list[dict]:
    """Return maintenance logs as text chunks suitable for the retrieval engine."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT log_id, equipment_id, equipment_name, fault_title, description, "
            "how_occurred, observed_at, reported_by, severity, action_taken, "
            "assigned_to, resolution, status, resolved_at "
            "FROM maintenance_logs ORDER BY observed_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    chunks = []
    for r in rows:
        d = dict(r)
        text = (
            f"Maintenance Log {d['log_id']} | Equipment: {d['equipment_id']} {d['equipment_name']} | "
            f"Fault: {d['fault_title']} | Severity: {d['severity']} | Status: {d['status']} | "
            f"Observed: {d['observed_at']} | Reported by: {d['reported_by']} | "
            f"Assigned to: {d['assigned_to']} | "
            f"Description: {d['description']} | "
            f"How occurred: {d['how_occurred']} | "
            f"Action taken: {d['action_taken']} | "
            f"Resolution: {d['resolution']} | "
            f"Resolved at: {d['resolved_at'] or 'Not yet resolved'}"
        )
        chunks.append({
            "doc_id":   f"ml_{d['log_id']}",
            "document": f"[Maintenance Log] {d['equipment_id']}",
            "file_type":"log",
            "chunk_id": f"ml_{d['log_id']}-000",
            "page":     None,
            "location": d["log_id"],
            "text":     text,
            "n_words":  len(text.split()),
        })
    return chunks
