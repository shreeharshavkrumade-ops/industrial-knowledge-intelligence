"""Text cleaning, sentence splitting and overlapping chunking."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: object) -> str:
    """Normalise unicode, remove control characters, repair wrapped lines, collapse whitespace.

    Newlines between paragraphs are kept (they help sentence splitting); newlines that merely
    wrap a sentence (next line starts lowercase / indented) are joined.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    t = _CTRL.sub(" ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)             # re-join words hyphenated at line end
    t = re.sub(r"\n[ \t]+(?=[a-z(])", " ", t)           # indented lowercase continuation line
    t = re.sub(r"(?<![.!?:;\n])\n(?=[a-z])", " ", t)    # wrapped line that continues a sentence
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (lines first, then terminal punctuation)."""
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", line):
            s = s.strip()
            if len(re.findall(r"[A-Za-z]", s)) >= 3:   # drop bare list numbers such as "1."
                out.append(s)
    return out


def chunk_text(text: str, chunk_words: int = 90, overlap_words: int = 20) -> list[str]:
    """Sentence-aware chunking with overlap.

    Sentences are accumulated until ~chunk_words; the next chunk starts with the trailing
    sentences of the previous one (up to overlap_words) so context at boundaries is not lost.
    Very long sentences are hard-split by words.
    """
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be larger than overlap_words")
    sentences: list[str] = []
    for s in split_sentences(text):
        w = s.split()
        if len(w) > chunk_words:
            sentences += [" ".join(w[i:i + chunk_words]) for i in range(0, len(w), chunk_words)]
        else:
            sentences.append(s)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        n = len(s.split())
        if cur and cur_len + n > chunk_words:
            chunks.append("\n".join(cur))
            keep: list[str] = []
            kl = 0
            for prev in reversed(cur):
                pl = len(prev.split())
                if kl + pl > overlap_words:
                    break
                keep.insert(0, prev)
                kl += pl
            cur, cur_len = keep, kl
        cur.append(s)
        cur_len += n
    if cur and (not chunks or "\n".join(cur) != chunks[-1]):
        chunks.append("\n".join(cur))
    return chunks


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_dataframe(df: pd.DataFrame, text_cols: list[str] | None = None, date_cols: list[str] | None = None,
                    key_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Generic tabular cleaning: strip strings, parse dates, drop duplicates, report what changed."""
    stats = {"rows_in": len(df)}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("string").str.strip()
    stats["missing_cells"] = int(df.isna().sum().sum() + (df.astype(str) == "").sum().sum())
    for c in text_cols or []:
        if c in df:
            df[c] = df[c].fillna("").astype(str).map(clean_text)
    for c in date_cols or []:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    before = len(df)
    df = df.drop_duplicates(subset=key_cols) if key_cols else df.drop_duplicates()
    stats["duplicates_removed"] = before - len(df)
    if date_cols:
        stats["unparseable_dates"] = int(sum(df[c].isna().sum() for c in date_cols if c in df))
    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats
