"""Retrieval models over document chunks.

Three modes, reported honestly in the UI:
  tfidf     - TF-IDF (uni+bi-grams, sublinear tf, light stemming) + cosine similarity. Always available.
  lsa       - Latent Semantic Analysis: TruncatedSVD on the TF-IDF matrix. A lightweight, fully local
              *latent-semantic* method. It is NOT a neural sentence-embedding model.
  semantic  - Sentence-Transformers embeddings (all-MiniLM-L6-v2) with cosine similarity. Only used if the
              package is installed AND the model could be loaded (needs a one-off download). If not,
              the mode reports itself unavailable; it is never faked.

Vector search is an exact NumPy dot product on L2-normalised vectors, which is fast enough for
thousands of chunks and avoids a FAISS dependency.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

LOG = logging.getLogger("iki.retrieval")
SEED = 42
MODES = ("tfidf", "lsa", "semantic")
MODE_LABELS = {"tfidf": "Keyword (TF-IDF)", "lsa": "Latent semantic (LSA)", "semantic": "Neural semantic (SBERT)"}
_TOKEN = re.compile(r"[a-z0-9]+")
_ID = re.compile(r"\b[a-z]{1,4}-\d{2,4}\b")
_STOP = set(ENGLISH_STOP_WORDS) - {"no", "not"}


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'lubricated'/'lubricate' and 'failures'/'failure' match."""
    for suf in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def analyzer(text: str) -> list[str]:
    """Tokenise; keep asset/work-order IDs (e.g. 'p-101') as single extra tokens; add bigrams."""
    t = str(text).lower()
    ids = _ID.findall(t)
    toks = [_stem(w) for w in _TOKEN.findall(t) if w not in _STOP and len(w) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams + ids


class RetrievalEngine:
    def __init__(self, chunks: pd.DataFrame, lsa_components: int = 60, seed: int = SEED):
        self.chunks = chunks.reset_index(drop=True) if chunks is not None else pd.DataFrame()
        self.seed = seed
        self.build_seconds: dict[str, float] = {}
        self.semantic_error: str | None = None
        self._sbert = None
        self._sbert_matrix: np.ndarray | None = None
        self.vectorizer = self.tfidf = self.svd = self.lsa = None
        if self.chunks.empty:
            return
        t0 = time.perf_counter()
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(self.chunks["text"].astype(str))
        self.build_seconds["tfidf"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        max_comp = min(lsa_components, self.tfidf.shape[0] - 1, self.tfidf.shape[1] - 1)
        if max_comp >= 2:
            self.svd = TruncatedSVD(n_components=max_comp, random_state=seed)
            self.lsa = normalize(self.svd.fit_transform(self.tfidf))
        self.build_seconds["lsa"] = time.perf_counter() - t0

    # ---- availability -------------------------------------------------------------------
    def load_semantic(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
        """Try to load SBERT and embed all chunks. Returns False (and stores the reason) on failure."""
        if self._sbert_matrix is not None:
            return True
        if self.chunks.empty:
            return False
        try:
            t0 = time.perf_counter()
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(model_name)
            emb = self._sbert.encode(self.chunks["text"].astype(str).tolist(), batch_size=32,
                                     show_progress_bar=False, normalize_embeddings=True)
            self._sbert_matrix = np.asarray(emb)
            self.build_seconds["semantic"] = time.perf_counter() - t0
            self.semantic_error = None
            return True
        except Exception as exc:
            self.semantic_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("Semantic model unavailable: %s", self.semantic_error)
            return False

    def availability(self) -> dict[str, tuple[bool, str]]:
        if self.chunks.empty:
            return {m: (False, "No documents indexed") for m in MODES}
        sem = self._sbert_matrix is not None
        note = "Loaded" if sem else (self.semantic_error or "Not loaded - click 'Load neural model' (needs sentence-transformers + one-off model download)")
        return {"tfidf": (True, "Always available"),
                "lsa": (self.lsa is not None, "Local latent-semantic fallback (not a neural embedding)" if self.lsa is not None else "Too few chunks"),
                "semantic": (sem, note)}

    # ---- scoring ------------------------------------------------------------------------
    def scores(self, query: str, mode: str = "tfidf") -> np.ndarray:
        if self.chunks.empty or not str(query).strip():
            return np.zeros(len(self.chunks))
        if mode == "tfidf":
            q = self.vectorizer.transform([query])
            return (self.tfidf @ q.T).toarray().ravel()  # tf-idf rows are L2-normalised => dot = cosine
        if mode == "lsa":
            if self.lsa is None:
                raise RuntimeError("LSA unavailable (too few chunks).")
            q = normalize(self.svd.transform(self.vectorizer.transform([query])))
            return (self.lsa @ q.T).ravel()
        if mode == "semantic":
            if self._sbert_matrix is None:
                raise RuntimeError(f"Semantic model not available: {self.semantic_error or 'not loaded'}")
            q = self._sbert.encode([query], normalize_embeddings=True, show_progress_bar=False)
            return (self._sbert_matrix @ np.asarray(q).T).ravel()
        raise ValueError(f"Unknown mode {mode}")

    def search(self, query: str, mode: str = "tfidf", top_k: int = 5, min_score: float = 0.0) -> pd.DataFrame:
        cols = ["rank", "document", "location", "score", "excerpt", "chunk_id", "doc_id"]
        s = self.scores(query, mode)
        if s.size == 0:
            return pd.DataFrame(columns=cols)
        order = np.argsort(-s)[:top_k]
        rows = []
        for r, i in enumerate(order, start=1):
            if s[i] < min_score or s[i] <= 0:
                continue
            c = self.chunks.iloc[i]
            rows.append({"rank": r, "document": c["document"], "location": c["location"], "score": round(float(s[i]), 4),
                         "excerpt": c["text"], "chunk_id": c["chunk_id"], "doc_id": c["doc_id"]})
        return pd.DataFrame(rows, columns=cols)

    def rank_documents(self, query: str, mode: str = "tfidf") -> list[str]:
        """Document ranking for evaluation: each document is scored by its best chunk."""
        s = self.scores(query, mode)
        if s.size == 0:
            return []
        df = pd.DataFrame({"document": self.chunks["document"], "score": s})
        best = df.groupby("document")["score"].max().sort_values(ascending=False)
        return best.index.tolist()
