"""Evidence-grounded EXTRACTIVE question answering.

No generative model is used here. The answer is assembled from sentences that already exist in the
retrieved chunks, each tagged with a citation. If evidence is weak, or the asset ID asked about
does not occur in any retrieved text, the assistant says it cannot answer instead of guessing.

Improvements over the original:
- top_k default raised to 8, max_sentences to 6
- Deduplication of near-identical sentences
- Structured output grouping evidence by source document
- Better "insufficient evidence" explanation
"""
from __future__ import annotations

import re

from .entity_extractor import ASSET_RE
from .text_preprocessor import split_sentences

# chunk-score thresholds differ by scoring scale (cosine on TF-IDF vs LSA vs SBERT)
MIN_SCORE = {"tfidf": 0.08, "lsa": 0.50, "semantic": 0.30}
SENT_MIN  = 0.03
HEADING_FOLLOW = 3   # a matched section heading is expanded with up to N following sentences


def _is_heading(sent: str) -> bool:
    w = sent.split()
    return len(w) <= 8 and not sent.rstrip().endswith((".", ";", "!", "?")) and ":" not in sent


def _is_title(sent: str) -> bool:
    return sent.isupper() or bool(
        re.match(r"^(Document No|Department|Date|Reported by|Reviewed by|Prepared by|Inspector)\b", sent)
    )


def _near_dup(a: str, b: str) -> bool:
    """Return True if a and b share > 70 % of their words (simple Jaccard-ish)."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) / max(len(wa), len(wb)) > 0.70


def answer_question(
    engine,
    question: str,
    mode: str = "tfidf",
    top_k: int = 8,
    max_sentences: int = 6,
) -> dict:
    res = {
        "question":  question,
        "mode":      mode,
        "sufficient": False,
        "answer":    "",
        "citations": [],
        "notes":     [],
    }
    if engine is None or engine.chunks.empty:
        res["answer"] = "The knowledge base is empty. Load the sample dataset or upload documents first."
        return res

    asked_assets = [a.upper() for a in ASSET_RE.findall(question.upper())]
    hits = engine.search(question, mode, top_k=50 if asked_assets else top_k)

    # --- asset-specific filtering ----------------------------------------
    if asked_assets and not hits.empty:
        mask = hits["excerpt"].str.upper().apply(lambda t: any(a in t for a in asked_assets))
        hits = hits[mask].head(top_k)
        if hits.empty:
            res["answer"] = (
                f"I found no document text that mentions **{', '.join(asked_assets)}**, "
                "so I cannot answer from the available evidence. "
                "Try uploading documents that reference this equipment."
            )
            return res

    thr = MIN_SCORE.get(mode, 0.08)
    if hits.empty or hits.iloc[0]["score"] < thr:
        best = hits.iloc[0]["score"] if not hits.empty else 0
        res["answer"] = (
            "The available documents do not contain enough evidence to answer this question "
            f"(best relevance score: **{best:.2f}**, threshold: **{thr}**). "
            "Try rephrasing your query, adding more keywords, or uploading relevant documents."
        )
        return res

    hits = hits[hits["score"] >= thr]

    # --- sentence scoring ------------------------------------------------
    qv   = engine.vectorizer.transform([question])
    cand = []
    for ci, h in enumerate(hits.itertuples(index=False), start=1):
        is_row = str(h.location).startswith("row")
        sents  = [h.excerpt] if is_row else split_sentences(h.excerpt)
        if not sents:
            continue
        sv = engine.vectorizer.transform(sents)
        sc = (sv @ qv.T).toarray().ravel()
        for si, (s, v) in enumerate(zip(sents, sc)):
            if _is_title(s) and not is_row:
                continue
            if asked_assets and not any(a in s.upper() for a in asked_assets) and len(sents) > 1:
                v *= 0.6
            if v < SENT_MIN:
                continue
            w = v * (1.0 + 0.1 * (len(hits) - ci))
            if not is_row and _is_heading(s):
                follow = [t for t in sents[si + 1: si + 1 + HEADING_FOLLOW] if not _is_heading(t) and not _is_title(t)]
                if not follow:
                    continue
                cand.append((w, ci, s + " " + " ".join(follow), h.document))
            else:
                cand.append((w, ci, s, h.document))

    cand.sort(key=lambda x: -x[0])

    # --- select best, deduplicate ----------------------------------------
    chosen: list[tuple[int, str, str]] = []   # (ci, sentence, document)
    for v, ci, s, doc in cand:
        if any(_near_dup(s, c) for _, c, _ in chosen):
            continue
        chosen = [(cj, c, d) for cj, c, d in chosen if not _near_dup(c, s)]
        chosen.append((ci, s, doc))
        if len(chosen) >= max_sentences:
            break

    # --- format output ---------------------------------------------------
    if not chosen:
        res["answer"] = (
            "Documents were retrieved, but no individual sentence scored high enough "
            "to quote as evidence. See the Retrieved Evidence section for the full passages."
        )
    else:
        chosen_sorted = sorted(chosen, key=lambda x: x[0])

        # Group by source document
        by_doc: dict[str, list[tuple[int, str]]] = {}
        for ci, sent, doc in chosen_sorted:
            by_doc.setdefault(doc, []).append((ci, sent))

        lines = []
        if len(by_doc) > 1:
            lines.append(
                f"Based on **{len(chosen_sorted)}** evidence passage(s) "
                f"from **{len(by_doc)}** document(s):\n"
            )
        else:
            lines.append(
                f"Based on **{len(chosen_sorted)}** evidence passage(s):\n"
            )

        for doc, passages in by_doc.items():
            if len(by_doc) > 1:
                lines.append(f"**{doc}**")
            for ci, sent in passages:
                lines.append(f"- {sent} *[{ci}]*")
            if len(by_doc) > 1:
                lines.append("")

        res["answer"]    = "\n".join(lines).strip()
        res["sufficient"] = True

    used = {ci for ci, _, _ in chosen}
    for ci, h in enumerate(hits.itertuples(index=False), start=1):
        res["citations"].append({
            "n":        ci,
            "document": h.document,
            "location": h.location,
            "score":    h.score,
            "excerpt":  h.excerpt,
            "quoted":   ci in used,
            "chunk_id": h.chunk_id,
        })
    if any(c["document"].endswith(".csv") for c in res["citations"]):
        res["notes"].append("Some evidence comes from structured CSV records (one row per citation).")
    return res
