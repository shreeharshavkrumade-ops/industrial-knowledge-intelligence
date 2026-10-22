"""Local persistence: SQLite for processed tables, plain files for uploads/exports.

All paths are resolved relative to the project root and checked so that file
operations never leave the project directory.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("iki.storage")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_documents"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = DATA_DIR / "knowledge.db"

TABLES = ("documents", "chunks", "entities", "search_log")


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise ValueError(f"Path outside project directory: {path}")
    return resolved


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLE_DIR, UPLOAD_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


@contextmanager
def connect(db_path: Path | None = None):
    ensure_dirs()
    conn = sqlite3.connect(str(_inside_project(db_path or DB_PATH)))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_table(name: str, df: pd.DataFrame, db_path: Path | None = None) -> None:
    """Replace a table with the DataFrame contents (processed tables are derived data)."""
    if name not in TABLES:
        raise ValueError(f"Unknown table {name}")
    with connect(db_path) as conn:
        if len(df.columns) == 0:  # nothing to store: drop the table instead of creating a column-less one
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        else:
            df.to_sql(name, conn, if_exists="replace", index=False)


def load_table(name: str, db_path: Path | None = None) -> pd.DataFrame:
    try:
        with connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {name}", conn)
    except Exception:  # missing table / empty db is a normal first-run state
        return pd.DataFrame()


def append_search_log(query: str, mode: str, n_results: int, top_score: float, runtime_ms: float,
                      db_path: Path | None = None) -> None:
    row = pd.DataFrame([{"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "query": query, "mode": mode,
                         "n_results": int(n_results), "top_score": float(top_score), "runtime_ms": float(runtime_ms)}])
    try:
        with connect(db_path) as conn:
            row.to_sql("search_log", conn, if_exists="append", index=False)
    except Exception as exc:
        LOG.warning("Could not log search: %s", exc)


def safe_filename(name: str) -> str:
    name = Path(name).name  # drop any directory components (path traversal)
    name = re.sub(r"[^A-Za-z0-9._\- ]", "_", name).strip() or "unnamed"
    return name[:120]


def save_upload(name: str, data: bytes) -> Path:
    """Store an uploaded file without ever overwriting an existing one."""
    ensure_dirs()
    target = UPLOAD_DIR / safe_filename(name)
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = UPLOAD_DIR / f"{stem}_{i}{suffix}"
        i += 1
    _inside_project(target).write_bytes(data)
    return target


def export_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    path = _inside_project(OUTPUT_DIR / safe_filename(filename))
    df.to_csv(path, index=False)
    return path


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def delete_upload(filename: str) -> tuple[bool, str]:
    """Delete an uploaded file from data/uploads/.

    Returns (success, message). Never deletes sample files or files
    outside the uploads directory.
    """
    safe = safe_filename(filename)
    try:
        path = _inside_project(UPLOAD_DIR / safe)
    except ValueError as exc:
        return False, f"Security: {exc}"
    if not path.exists():
        return False, f"File not found in uploads: {safe}"
    # Double-check: the resolved path must be directly inside UPLOAD_DIR
    if path.parent.resolve() != UPLOAD_DIR.resolve():
        return False, "Security: path outside uploads directory"
    try:
        path.unlink()
        LOG.info("Deleted upload: %s", safe)
        return True, f"Deleted: {safe}"
    except Exception as exc:
        return False, f"Failed to delete {safe}: {exc}"
