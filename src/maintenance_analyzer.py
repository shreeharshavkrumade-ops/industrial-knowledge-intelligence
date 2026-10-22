"""ML-assisted maintenance analysis: fault clustering, recurrence analysis, evidence-backed
recommendations and a baseline failure-risk classifier.

Important: all results are *observed patterns* in the supplied (synthetic) records, never proven
root causes, and the classifier only learns the synthetic labelling rule.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
                             normalized_mutual_info_score, precision_score, recall_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .text_preprocessor import clean_dataframe

LOG = logging.getLogger("iki.maintenance")
SEED = 42
REQUIRED = ["work_order_id", "date", "equipment_id", "fault_description"]
NUMERIC_FEATURES = ["operating_hours_since_overhaul", "days_since_last_pm", "vibration_mm_s", "temperature_c"]
CAT_FEATURES = ["equipment_type"]
TARGET = "failure_within_30d"
_TYPE = {"P": "Pump", "M": "Motor", "C": "Compressor", "B": "Boiler", "V": "Valve", "T": "Tank"}


def is_maintenance_frame(df: pd.DataFrame) -> bool:
    return not df.empty and all(c in df.columns for c in REQUIRED)


def prepare_maintenance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a raw work-order table. Returns (clean_df, preprocessing_stats)."""
    if df is None or df.empty:
        return pd.DataFrame(), {"rows_in": 0, "rows_out": 0}
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Maintenance table is missing required columns: {missing}")
    out, stats = clean_dataframe(df, text_cols=["fault_description", "action_taken"], date_cols=["date"],
                                 key_cols=["work_order_id"])
    n0 = len(out)
    out = out.dropna(subset=["date", "equipment_id"])
    out = out[out["fault_description"].astype(str).str.len() > 0]
    stats["rows_dropped_invalid"] = n0 - len(out)
    if "equipment_type" not in out:
        out["equipment_type"] = out["equipment_id"].astype(str).str[0].map(_TYPE).fillna("Unknown")
    if "downtime_hours" in out:
        out["downtime_hours"] = pd.to_numeric(out["downtime_hours"], errors="coerce")
        stats["downtime_missing_imputed"] = int(out["downtime_hours"].isna().sum())
        out["downtime_hours"] = out["downtime_hours"].fillna(out["downtime_hours"].median())  # median is robust to skew
    for c in ("action_taken", "component"):
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    return out.reset_index(drop=True), stats


def _top_terms(vec: TfidfVectorizer, centroid: np.ndarray, n: int = 4) -> list[str]:
    terms = np.array(vec.get_feature_names_out())
    return terms[np.argsort(-centroid)[:n]].tolist()


def cluster_faults(df: pd.DataFrame, k: int | None = None, k_range=(2, 10), seed: int = SEED) -> dict:
    """TF-IDF + K-Means over fault descriptions. k is chosen by silhouette score unless given."""
    t0 = time.perf_counter()
    texts = df["fault_description"].astype(str)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(texts)
    n = X.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 maintenance records to cluster.")
    sil = []
    for kk in range(k_range[0], min(k_range[1], n - 1) + 1):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(X)
        if len(set(lab)) > 1:
            sil.append({"k": kk, "silhouette": float(silhouette_score(X, lab, random_state=seed))})
    sil_df = pd.DataFrame(sil)
    chosen = int(k) if k else int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=chosen, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    names = {}
    for c in range(chosen):
        top = _top_terms(vec, km.cluster_centers_[c], 3)
        names[c] = f"C{c}: " + ", ".join(top)
    out = df.copy()
    out["cluster"] = labels
    out["fault_category"] = out["cluster"].map(names)
    rows = []
    for c in range(chosen):
        sub = out[out["cluster"] == c]
        rows.append({"cluster": c, "fault_category": names[c], "records": len(sub), "share_pct": round(100 * len(sub) / n, 1),
                     "top_terms": ", ".join(_top_terms(vec, km.cluster_centers_[c], 6)),
                     "top_equipment": ", ".join(sub["equipment_id"].value_counts().head(3).index),
                     "avg_downtime_h": round(float(sub["downtime_hours"].mean()), 1) if "downtime_hours" in sub else np.nan,
                     "example": sub["fault_description"].iloc[0]})
    metrics = {"k": chosen, "silhouette": float(silhouette_score(X, labels, random_state=seed)), "inertia": float(km.inertia_),
               "n_records": n, "runtime_s": time.perf_counter() - t0}
    if "fault_label" in df.columns and df["fault_label"].nunique() > 1:
        # External validity check only: 'fault_label' is the template category used to generate the
        # synthetic data. It is never an input to the clustering.
        metrics["adjusted_rand_index"] = float(adjusted_rand_score(df["fault_label"], labels))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(df["fault_label"], labels))
    return {"data": out, "summary": pd.DataFrame(rows), "silhouette_by_k": sil_df, "metrics": metrics}


def fault_frequency(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["equipment_id", "fault_category"]).size().reset_index(name="count")
            if "fault_category" in df else df.groupby("equipment_id").size().reset_index(name="count"))


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df:
        return pd.DataFrame(columns=["month", "work_orders"])
    m = df.dropna(subset=["date"]).groupby(df["date"].dt.to_period("M")).size()
    return m.rename("work_orders").reset_index().assign(month=lambda d: d["date"].dt.to_timestamp())[["month", "work_orders"]]


def recurring_issues(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """Equipment/fault-category pairs seen at least min_count times, with the average gap between events."""
    rows = []
    for (eq, cat), g in df.groupby(["equipment_id", "fault_category"]):
        if len(g) < min_count:
            continue
        d = g["date"].sort_values()
        rows.append({"equipment_id": eq, "fault_category": cat, "occurrences": len(g),
                     "first_seen": d.iloc[0].date(), "last_seen": d.iloc[-1].date(),
                     "mean_gap_days": round(float(d.diff().dt.days.dropna().mean()), 1),
                     "total_downtime_h": round(float(g["downtime_hours"].sum()), 1) if "downtime_hours" in g else np.nan,
                     "work_orders": ", ".join(g["work_order_id"].astype(str).head(4)) + (" ..." if len(g) > 4 else "")})
    cols = ["equipment_id", "fault_category", "occurrences", "first_seen", "last_seen", "mean_gap_days", "total_downtime_h", "work_orders"]
    return pd.DataFrame(rows, columns=cols).sort_values("occurrences", ascending=False).reset_index(drop=True)


def recommendations(recurring: pd.DataFrame, df: pd.DataFrame, engine=None, top_n: int = 8, min_evidence_score: float = 0.15) -> pd.DataFrame:
    """Evidence-backed suggestions: the most frequent recorded action plus the best-matching manual passage."""
    rows = []
    for r in recurring.head(top_n).itertuples(index=False):
        g = df[(df["equipment_id"] == r.equipment_id) & (df["fault_category"] == r.fault_category)]
        acts = g["action_taken"].replace("", np.nan).dropna()
        top_action = acts.value_counts().index[0] if len(acts) else "no action recorded"
        ev_doc = ev_txt = ""
        ev_score = 0.0
        if engine is not None and not engine.chunks.empty:
            terms = r.fault_category.split(": ", 1)[-1].replace(",", " ")
            hit = engine.search(f"{r.equipment_id} {terms}", "tfidf", top_k=1)
            hit = hit[hit["document"] != "maintenance_records.csv"] if not hit.empty else hit
            if hit.empty:  # fall back to the best non-record document
                allhits = engine.search(f"{r.equipment_id} {terms}", "tfidf", top_k=8)
                allhits = allhits[~allhits["document"].str.endswith(".csv")]
                hit = allhits.head(1)
            if not hit.empty and hit.iloc[0]["score"] >= min_evidence_score:
                ev_doc, ev_txt, ev_score = hit.iloc[0]["document"], hit.iloc[0]["excerpt"][:260], float(hit.iloc[0]["score"])
        rows.append({
            "equipment_id": r.equipment_id, "fault_category": r.fault_category, "occurrences": r.occurrences,
            "observed_pattern": f"{r.equipment_id} had {r.occurrences} '{r.fault_category}' records (mean gap {r.mean_gap_days} days). "
                                "This is an observed pattern, not a proven root cause.",
            "suggested_review": f"Review recurring cause; most frequent recorded action: '{top_action}'. Consider verifying whether the repair addresses the underlying cause.",
            "supporting_work_orders": r.work_orders,
            "manual_evidence_document": ev_doc or "none found above threshold",
            "manual_evidence_excerpt": ev_txt, "evidence_score": round(ev_score, 3)})
    return pd.DataFrame(rows)


def can_train_classifier(df: pd.DataFrame) -> tuple[bool, str]:
    need = NUMERIC_FEATURES + CAT_FEATURES + [TARGET]
    miss = [c for c in need if c not in df.columns]
    if miss:
        return False, f"Missing columns for risk classification: {miss}"
    y = df[TARGET].dropna()
    if y.nunique() < 2 or y.value_counts().min() < 8 or len(y) < 40:
        return False, "Need >=40 labelled rows with >=8 examples of each class."
    return True, ""


def train_risk_model(df: pd.DataFrame, seed: int = SEED, test_size: float = 0.25) -> dict:
    """Baseline failure-risk classifier with a stratified split; compares to a majority-class dummy.

    The scaler/encoder live inside the Pipeline so they are fit on training rows only (no leakage).
    """
    ok, why = can_train_classifier(df)
    if not ok:
        return {"available": False, "reason": why}
    t0 = time.perf_counter()
    d = df.dropna(subset=NUMERIC_FEATURES + CAT_FEATURES + [TARGET])
    X, y = d[NUMERIC_FEATURES + CAT_FEATURES], d[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = lambda: ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES),
                                     ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])
    models = {"Majority-class baseline": DummyClassifier(strategy="most_frequent"),
              "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
              "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=seed)}
    results, cms, fitted = [], {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre()), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1")  # CV on TRAIN only; test set stays untouched
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results.append({"model": name, "accuracy": accuracy_score(yte, pred),
                        "precision": precision_score(yte, pred, zero_division=0), "recall": recall_score(yte, pred, zero_division=0),
                        "f1": f1_score(yte, pred, zero_division=0), "cv_f1_mean_train": float(cv_f1.mean()), "cv_f1_std_train": float(cv_f1.std())})
        cms[name] = confusion_matrix(yte, pred, labels=[0, 1])
        fitted[name] = pipe
    rf = fitted["Random Forest"]
    names = NUMERIC_FEATURES + list(rf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))
    imp = pd.DataFrame({"feature": names, "importance": rf.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False)
    return {"available": True, "metrics": pd.DataFrame(results), "confusion_matrices": cms, "feature_importance": imp,
            "features": NUMERIC_FEATURES + CAT_FEATURES, "target": TARGET, "test_size": test_size,
            "n_train": len(Xtr), "n_test": len(Xte), "train_positive_rate": float(ytr.mean()), "test_positive_rate": float(yte.mean()),
            "seed": seed, "runtime_s": time.perf_counter() - t0}
