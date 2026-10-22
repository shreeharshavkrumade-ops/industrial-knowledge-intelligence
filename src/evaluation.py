"""Evaluation utilities. Every number is computed from actual predictions vs. labelled data.

Entity extraction : set-based precision / recall / F1 against data/entity_ground_truth.csv
Retrieval         : Precision@K, Recall@K, MRR (document level) against data/retrieval_benchmark.csv
Classification    : see maintenance_analyzer.train_risk_model (metrics computed there from held-out rows)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .retrieval_engine import MODE_LABELS

CASE_INSENSITIVE_TYPES = {"component", "fault", "maintenance_action"}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 with the convention 0.0 when the denominator is 0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _norm(entity_type: str, value: str) -> str:
    v = " ".join(str(value).split())
    return v.upper() if entity_type in ("asset_id", "regulatory_reference") else v.lower()


def evaluate_entities(extracted: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Document-level set comparison, restricted to the documents and entity types that were labelled."""
    if ground_truth is None or ground_truth.empty:
        return {"available": False, "reason": "No ground-truth file."}
    docs = set(ground_truth["document"]) & set(extracted["document"]) if not extracted.empty else set()
    if not docs:
        return {"available": False, "reason": "Labelled documents are not in the current knowledge base."}
    types = sorted(ground_truth["entity_type"].unique())
    gt = {(r.document, r.entity_type, _norm(r.entity_type, r.value)) for r in ground_truth[ground_truth["document"].isin(docs)].itertuples()}
    ex_df = extracted[extracted["document"].isin(docs) & extracted["entity_type"].isin(types)]
    ex = {(r.document, r.entity_type, _norm(r.entity_type, r.value)) for r in ex_df.itertuples()}
    rows, tp_all, fp_all, fn_all = [], 0, 0, 0
    for t in types:
        g, e = {x for x in gt if x[1] == t}, {x for x in ex if x[1] == t}
        tp, fp, fn = len(g & e), len(e - g), len(g - e)
        p, r, f = prf(tp, fp, fn)
        rows.append({"entity_type": t, "support (gold)": len(g), "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f})
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    per_type = pd.DataFrame(rows)
    mp, mr, mf = prf(tp_all, fp_all, fn_all)
    errors = pd.DataFrame([{"error": "false positive", "document": d, "entity_type": t, "value": v} for d, t, v in sorted(ex - gt)] +
                          [{"error": "false negative (missed)", "document": d, "entity_type": t, "value": v} for d, t, v in sorted(gt - ex)])
    return {"available": True, "per_type": per_type, "documents_evaluated": sorted(docs), "n_gold": len(gt), "n_predicted": len(ex),
            "micro": {"precision": mp, "recall": mr, "f1": mf}, "macro": {"precision": per_type["precision"].mean(),
            "recall": per_type["recall"].mean(), "f1": per_type["f1"].mean()}, "errors": errors}


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k if k else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(engine, benchmark: pd.DataFrame, modes: list[str] | None = None, ks=(1, 3, 5)) -> dict:
    if benchmark is None or benchmark.empty or engine is None or engine.chunks.empty:
        return {"available": False, "reason": "No benchmark or no indexed documents."}
    corpus = set(engine.chunks["document"])
    avail = engine.availability()
    modes = [m for m in (modes or ["tfidf", "lsa", "semantic"]) if avail[m][0]]
    per_q, summary, skipped = [], [], []
    for mode in modes:
        lat, rr, pk, rk = [], [], {k: [] for k in ks}, {k: [] for k in ks}
        for q in benchmark.itertuples(index=False):
            rel = {d.strip() for d in str(q.relevant_documents).split(";") if d.strip()} & corpus
            if not rel:
                skipped.append(q.query)
                continue
            t0 = time.perf_counter()
            ranked = engine.rank_documents(q.query, mode)
            lat.append((time.perf_counter() - t0) * 1000)
            rr.append(reciprocal_rank(ranked, rel))
            row = {"mode": MODE_LABELS[mode], "query": q.query, "relevant": len(rel), "first_relevant_rank": next((i for i, d in enumerate(ranked, 1) if d in rel), None), "top1": ranked[0] if ranked else ""}
            for k in ks:
                pk[k].append(precision_at_k(ranked, rel, k)); rk[k].append(recall_at_k(ranked, rel, k))
                row[f"P@{k}"], row[f"R@{k}"] = pk[k][-1], rk[k][-1]
            row["RR"] = rr[-1]
            per_q.append(row)
        if rr:
            s = {"mode": MODE_LABELS[mode], "queries": len(rr), "MRR": float(np.mean(rr)), "avg_query_ms": float(np.mean(lat))}
            for k in ks:
                s[f"P@{k}"], s[f"R@{k}"] = float(np.mean(pk[k])), float(np.mean(rk[k]))
            summary.append(s)
    return {"available": True, "summary": pd.DataFrame(summary), "per_query": pd.DataFrame(per_q), "modes": modes,
            "unavailable": {MODE_LABELS[m]: v[1] for m, v in avail.items() if not v[0]}, "skipped_queries": sorted(set(skipped)), "ks": list(ks)}


def flatten_metrics(entity: dict, retrieval: dict, classification: dict, clustering: dict | None) -> pd.DataFrame:
    """One tidy table of every computed metric, for CSV export."""
    rows = []
    if entity.get("available"):
        for r in entity["per_type"].itertuples(index=False):
            for m in ("precision", "recall", "f1"):
                rows.append(("entity_extraction", r.entity_type, m, getattr(r, m)))
        for avg in ("micro", "macro"):
            for m, v in entity[avg].items():
                rows.append(("entity_extraction", avg, m, v))
    if retrieval.get("available"):
        for r in retrieval["summary"].to_dict("records"):
            for k, v in r.items():
                if k not in ("mode", "queries"):
                    rows.append(("retrieval", r["mode"], k, v))
    if classification.get("available"):
        for r in classification["metrics"].to_dict("records"):
            for k, v in r.items():
                if k != "model":
                    rows.append(("classification", r["model"], k, v))
    if clustering:
        for k, v in clustering["metrics"].items():
            rows.append(("clustering", "KMeans+TF-IDF", k, v))
    return pd.DataFrame(rows, columns=["category", "model_or_scope", "metric", "value"])
