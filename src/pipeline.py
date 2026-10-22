"""KnowledgeBase: the single object that connects all pipeline stages.

ingest (Stage 1) -> preprocess + chunk + extract entities (Stage 2) -> index / cluster / classify (Stage 3)
-> evaluate (Stage 4) -> visualise & apply (Stages 5-6). UI pages only call this class.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from . import storage
from .compliance_checker import run_compliance, summarize
from .document_processor import process_bytes
from .entity_extractor import extract_entities
from .evaluation import evaluate_entities, evaluate_retrieval, flatten_metrics
from .knowledge_graph import build_edges
from .maintenance_analyzer import (cluster_faults, is_maintenance_frame, prepare_maintenance,
                                   recommendations, recurring_issues, train_risk_model)
from .retrieval_engine import RetrievalEngine

LOG = logging.getLogger("iki.pipeline")
DEFAULT_AS_OF = date(2026, 9, 30)  # fixed so sample-data compliance results are reproducible
SAMPLE_CSVS = ("maintenance_records.csv", "inspection_records.csv")


class KnowledgeBase:
    def __init__(self):
        storage.ensure_dirs()
        self.documents = storage.load_table("documents")
        self.chunks = storage.load_table("chunks")
        self.entities = storage.load_table("entities")
        self.timings: dict[str, float] = {}
        self.engine: RetrievalEngine | None = None
        self.preproc_samples: list[dict] = []
        self._cache: dict = {}
        if not self.chunks.empty:
            self._index()
        self.requirements = self._read_data("compliance_requirements.csv")
        self.ground_truth = self._read_data("entity_ground_truth.csv")
        self.benchmark = self._read_data("retrieval_benchmark.csv")

    # ---------- helpers -------------------------------------------------------------------
    @staticmethod
    def _read_data(name: str) -> pd.DataFrame:
        p = storage.DATA_DIR / name
        try:
            return pd.read_csv(p) if p.exists() else pd.DataFrame()
        except Exception as exc:
            LOG.warning("Could not read %s: %s", name, exc)
            return pd.DataFrame()

    @property
    def is_empty(self) -> bool:
        return self.documents.empty

    def _index(self) -> None:
        """Rebuild TF-IDF + LSA index from current chunks.

        SBERT is NOT loaded here — it is heavy (90 MB, ~60s on CPU) and would
        block the Streamlit thread on every upload.  Load it explicitly via
        engine.load_semantic() on the Neural Search page when the user requests it.
        """
        t0 = time.perf_counter()
        self.engine = RetrievalEngine(self.chunks)
        self.timings["indexing_s"] = time.perf_counter() - t0
        self._cache.clear()
        LOG.info(
            "Index built: %d chunks, tfidf=%s, lsa=%s (SBERT deferred)",
            len(self.chunks),
            self.engine.tfidf is not None,
            self.engine.lsa is not None,
        )

    # ---------- Stage 1: ingestion --------------------------------------------------------
    def ingest(self, name: str, data: bytes, source: str = "upload", save_file: bool = True) -> dict:
        """Process a single file. Returns the document record (with status). Duplicates are rejected by content hash."""
        t0 = time.perf_counter()
        doc, chunks = process_bytes(name, data, source)
        rec = doc.record()
        if not self.documents.empty and doc.doc_id in set(self.documents["doc_id"]):
            rec.update(status="duplicate", message="Identical content already in the knowledge base (skipped).", n_chunks=0)
            return rec
        if source == "upload" and save_file and doc.status in ("processed", "warning"):
            storage.save_upload(name, data)
        self.documents = pd.concat([self.documents, pd.DataFrame([rec])], ignore_index=True)
        if chunks:
            self.chunks = pd.concat([self.chunks, pd.DataFrame(chunks)], ignore_index=True)
        if doc.pages:
            raw_first = doc.raw_chars
            self.preproc_samples.append({"document": doc.name, "raw_chars": raw_first, "clean_chars": doc.text_length,
                                         "preview_after": doc.pages[0][2][:260]})
        self.timings["ingest_s"] = self.timings.get("ingest_s", 0.0) + time.perf_counter() - t0
        return rec

    def load_samples(self) -> list[dict]:
        out = []
        files = sorted(storage.SAMPLE_DIR.glob("*")) + [storage.DATA_DIR / n for n in SAMPLE_CSVS]
        for p in files:
            if p.is_file() and p.name != ".gitkeep":
                out.append(self.ingest(p.name, p.read_bytes(), source="sample", save_file=False))
        self.rebuild()
        return out

    def ensure_demo_data(self) -> None:
        if self.is_empty:
            self.load_samples()

    # ---------- Stage 2/3: preprocessing, extraction, indexing --------------------------------
    def rebuild(self) -> None:
        """Recompute derived data (entities, index) from stored chunks. Source files are never deleted."""
        t0 = time.perf_counter()
        self.entities = extract_entities(self.chunks)
        self.timings["entity_extraction_s"] = time.perf_counter() - t0
        for name, df in (("documents", self.documents), ("chunks", self.chunks), ("entities", self.entities)):
            if not df.empty or name == "entities":
                storage.save_table(name, df)
        if not self.chunks.empty:
            self._index()

    def reset_derived(self) -> None:
        """Clear the processed database (sample/uploaded files on disk are untouched)."""
        for n in ("documents", "chunks", "entities", "search_log"):
            storage.save_table(n, pd.DataFrame())
        self.documents = self.chunks = self.entities = pd.DataFrame()
        self.engine = None
        self._cache.clear()

    # ---------- structured records ---------------------------------------------------------
    def _csv_documents(self) -> list[tuple[str, pd.DataFrame]]:
        """All CSVs in the knowledge base as DataFrames (sample files + uploads on disk)."""
        out = []
        if self.documents.empty:
            return out
        for r in self.documents[self.documents["file_type"] == "csv"].itertuples(index=False):
            p = storage.DATA_DIR / r.name if r.source == "sample" else storage.UPLOAD_DIR / r.name
            if p.exists():
                try:
                    out.append((r.name, pd.read_csv(p)))
                except Exception as exc:
                    LOG.warning("Cannot read %s: %s", p, exc)
        return out

    def maintenance(self) -> tuple[pd.DataFrame, dict]:
        if "maintenance" not in self._cache:
            frames = [df for _, df in self._csv_documents() if is_maintenance_frame(df)]
            if not frames:
                self._cache["maintenance"] = (pd.DataFrame(), {"rows_in": 0, "rows_out": 0})
            else:
                self._cache["maintenance"] = prepare_maintenance(pd.concat(frames, ignore_index=True))
        return self._cache["maintenance"]

    def inspections(self) -> pd.DataFrame:
        frames = [df for _, df in self._csv_documents() if {"inspection_id", "equipment_id", "inspection_type", "status"} <= set(df.columns)]
        return pd.concat(frames, ignore_index=True).drop_duplicates("inspection_id") if frames else pd.DataFrame()

    def clustering(self, k: int | None = None) -> dict | None:
        key = ("cluster", k)
        if key not in self._cache:
            m, _ = self.maintenance()
            self._cache[key] = cluster_faults(m, k=k) if len(m) >= 4 else None
        return self._cache[key]

    def classification(self) -> dict:
        if "clf" not in self._cache:
            m, _ = self.maintenance()
            self._cache["clf"] = train_risk_model(m) if not m.empty else {"available": False, "reason": "No maintenance records loaded."}
        return self._cache["clf"]

    def assets(self) -> pd.DataFrame:
        """Unique assets from entities (extracted IDs) and structured records."""
        ids: dict[str, str] = {}
        if not self.entities.empty:
            for v in self.entities.loc[self.entities["entity_type"] == "asset_id", "value"].unique():
                ids[v] = {"P": "Pump", "M": "Motor", "C": "Compressor", "B": "Boiler", "V": "Valve", "T": "Tank"}.get(v[0], "Unknown")
        m, _ = self.maintenance()
        if not m.empty:
            for a, t in m[["equipment_id", "equipment_type"]].drop_duplicates().itertuples(index=False):
                ids[a] = t
        return pd.DataFrame(sorted(ids.items()), columns=["equipment_id", "equipment_type"])

    def graph_edges(self) -> pd.DataFrame:
        if "edges" not in self._cache:
            self._cache["edges"] = build_edges(self.entities, self.maintenance()[0], self.inspections())
        return self._cache["edges"]

    def compliance(self, as_of: date = DEFAULT_AS_OF) -> pd.DataFrame:
        key = ("compliance", as_of)
        if key not in self._cache:
            self._cache[key] = run_compliance(self.requirements, self.inspections(), self.maintenance()[0], self.assets(), self.chunks, as_of)
        return self._cache[key]

    def recommendations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cl = self.clustering()
        if cl is None:
            return pd.DataFrame(), pd.DataFrame()
        rec = recurring_issues(cl["data"])
        return rec, recommendations(rec, cl["data"], self.engine)

    # ---------- Document deletion -----------------------------------------------------------
    def delete_document(self, doc_id: str) -> dict:
        """Delete a document and all its derived data from the knowledge base.

        Steps:
          1. Validate doc_id exists in self.documents
          2. Determine filename and source (upload vs sample)
          3. Delete physical file for uploads (never deletes sample files)
          4. Remove document row, chunks, entities from DataFrames
          5. Clear _cache (invalidates graph_edges, maintenance, compliance, etc.)
          6. Persist updated tables to DB
          7. Rebuild retrieval index from remaining chunks

        Returns a dict with keys: ok, doc_id, name, file_deleted,
        chunks_removed, entities_removed, error, warnings.
        """
        result: dict = {
            "ok": False,
            "doc_id": doc_id,
            "name": "",
            "file_deleted": False,
            "chunks_removed": 0,
            "entities_removed": 0,
            "error": None,
            "warnings": [],
        }

        # 1. Validate
        if self.documents.empty or doc_id not in self.documents["doc_id"].values:
            result["error"] = f"Document ID not found: {doc_id}"
            return result

        row = self.documents[self.documents["doc_id"] == doc_id].iloc[0]
        doc_name = str(row["name"])
        doc_source = str(row["source"]) if "source" in row.index else "upload"
        result["name"] = doc_name

        # 2. Delete physical file (uploads only — sample files stay on disk)
        if doc_source == "upload":
            ok, msg = storage.delete_upload(doc_name)
            result["file_deleted"] = ok
            if not ok:
                result["warnings"].append(f"File deletion: {msg}")
        else:
            result["warnings"].append(f"Sample file kept on disk (not deleted): {doc_name}")

        # 3. Remove document row
        self.documents = self.documents[self.documents["doc_id"] != doc_id].reset_index(drop=True)

        # 4. Remove chunks — match by doc_id first, fall back to document name
        if not self.chunks.empty:
            before = len(self.chunks)
            if "doc_id" in self.chunks.columns:
                self.chunks = self.chunks[self.chunks["doc_id"] != doc_id].reset_index(drop=True)
            else:
                self.chunks = self.chunks[self.chunks["document"] != doc_name].reset_index(drop=True)
            result["chunks_removed"] = before - len(self.chunks)

        # 5. Remove entities (keyed by document name)
        if not self.entities.empty and "document" in self.entities.columns:
            before = len(self.entities)
            self.entities = self.entities[self.entities["document"] != doc_name].reset_index(drop=True)
            result["entities_removed"] = before - len(self.entities)

        # 6. Clear all caches (graph_edges, maintenance, compliance, clustering, etc.)
        self._cache.clear()

        # 7. Persist to DB
        for tname, df in (
            ("documents", self.documents),
            ("chunks", self.chunks),
            ("entities", self.entities),
        ):
            storage.save_table(tname, df if not df.empty else pd.DataFrame())

        # 8. Rebuild index from remaining chunks (sets self.engine = None if empty)
        if not self.chunks.empty:
            self._index()
        else:
            self.engine = None

        result["ok"] = True
        LOG.info(
            "Deleted document %r (doc_id=%s): %d chunks, %d entities removed",
            doc_name, doc_id, result["chunks_removed"], result["entities_removed"],
        )
        return result

    # ---------- Stage 4: evaluation ---------------------------------------------------------
    def evaluate(self, refresh: bool = False) -> dict:
        """Compute every metric from live predictions. Cached until the data changes (index rebuild clears the cache)."""
        if "eval" in self._cache and not refresh:
            return self._cache["eval"]
        t0 = time.perf_counter()
        ent = evaluate_entities(self.entities, self.ground_truth)
        ret = evaluate_retrieval(self.engine, self.benchmark)
        clf = self.classification()
        cl = self.clustering()
        out = {"entity": ent, "retrieval": ret, "classification": clf, "clustering": cl,
               "metrics_table": flatten_metrics(ent, ret, clf, cl), "evaluation_runtime_s": time.perf_counter() - t0}
        self._cache["eval"] = out
        return out

    # ---------- dashboard statistics ----------------------------------------------------------
    def stats(self) -> dict:
        m, _ = self.maintenance()
        comp = self.compliance()
        return {"documents": len(self.documents), "chunks": len(self.chunks), "entities": len(self.entities),
                "assets": len(self.assets()), "incidents": len(m), "compliance_gaps": summarize(comp)["gaps"]}

    def preprocessing_stats(self) -> dict:
        if self.documents.empty:
            return {}
        d = self.documents
        return {"documents": len(d), "by_type": d["file_type"].value_counts().to_dict(), "raw_chars": int(d["raw_chars"].sum()),
                "clean_chars": int(d["text_length"].sum()), "chunks": int(d["n_chunks"].sum()),
                "avg_chunk_words": round(float(self.chunks["n_words"].mean()), 1) if not self.chunks.empty else 0,
                "status_counts": d["status"].value_counts().to_dict()}
