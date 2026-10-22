"""Document ingestion: PDF / DOCX / TXT / CSV -> cleaned text pages -> overlapping chunks.

Every file yields a ProcessedDocument record (status, sizes, errors) so failures are visible
in the UI instead of being silently dropped. Page numbers are preserved for PDFs; CSV rows are
treated as individual locations ("row N").
"""
from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .text_preprocessor import chunk_text, clean_text, content_hash

LOG = logging.getLogger("iki.ingest")
SUPPORTED = {".pdf", ".txt", ".docx", ".csv"}
MIN_PAGE_CHARS = 20  # a PDF page with less text than this is treated as scanned -> OCR attempt


@dataclass
class ProcessedDocument:
    doc_id: str
    name: str
    file_type: str
    source: str = "upload"            # "sample" or "upload"
    status: str = "processed"         # processed | warning | error | empty | unsupported
    n_pages: int = 0
    raw_chars: int = 0
    text_length: int = 0
    n_chunks: int = 0
    n_words: int = 0
    message: str = ""
    raw_preview: str = ""
    clean_preview: str = ""
    pages: list = field(default_factory=list, repr=False)  # [(location_label, page_no|None, text)]

    def record(self) -> dict:
        d = asdict(self)
        d.pop("pages")
        return d


# ---- OCR helpers -----------------------------------------------------------

_easyocr_reader = None   # module-level singleton; initialised on first use
_easyocr_init_failed = False  # set True after a failed init so we don't retry every page


def _get_easyocr():
    """Return a cached EasyOCR Reader, or None if the package is unavailable or init fails.

    Init is deferred to first use (not at import time) because loading the
    CRAFT + recognition models takes 30–120 s on CPU and would freeze the UI.
    After the first successful init the reader is reused for all subsequent calls.
    """
    global _easyocr_reader, _easyocr_init_failed
    if _easyocr_reader is not None:
        return _easyocr_reader
    if _easyocr_init_failed:
        return None
    try:
        import easyocr  # noqa: PLC0415
        LOG.info("Initialising EasyOCR reader (first use — may take 30-120s on CPU)…")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        LOG.info("EasyOCR reader ready.")
        return _easyocr_reader
    except Exception as exc:
        LOG.warning("EasyOCR init failed: %s", exc)
        _easyocr_init_failed = True
        return None


def easyocr_available() -> bool:
    """True if EasyOCR can be imported (no Tesseract binary required)."""
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def tesseract_available() -> bool:
    """True if Tesseract binary is present and reachable via pytesseract."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_available() -> bool:
    """True if ANY OCR backend is available (Tesseract OR EasyOCR)."""
    return tesseract_available() or easyocr_available()


def _ocr_image_bytes(png_bytes: bytes) -> str:
    """Run OCR on raw PNG bytes. Prefers Tesseract; falls back to EasyOCR.

    Returns empty string if both fail — callers should check and report."""
    # --- try Tesseract first (installed binary) ----------------------------
    if tesseract_available():
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes))
            return pytesseract.image_to_string(img)
        except Exception as exc:
            LOG.warning("Tesseract OCR failed: %s", exc)

    # --- fall back to EasyOCR (pure-Python, no binary required) -----------
    reader = _get_easyocr()
    if reader is not None:
        try:
            import numpy as np
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.array(img)  # shape (H, W, 3) — required by EasyOCR
            result = reader.readtext(arr, detail=0)
            return " ".join(result)
        except Exception as exc:
            LOG.warning("EasyOCR readtext failed: %s", exc)

    return ""


def _extract_txt(data: bytes) -> list[tuple[str, int | None, str]]:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return [("document", None, data.decode(enc))]
        except UnicodeDecodeError:
            continue
    return []


def _extract_docx(data: bytes) -> list[tuple[str, int | None, str]]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:  # tables carry hazard/control pairs, so keep them
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(": ".join(cells) + ".")
    return [("document", None, "\n".join(parts))]


def _extract_pdf(data: bytes) -> tuple[list[tuple[str, int | None, str]], str]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:
        import fitz  # older PyMuPDF releases
    pages, notes = [], []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            if len(text.strip()) < MIN_PAGE_CHARS:
                if ocr_available():
                    try:
                        pix = page.get_pixmap(dpi=200)
                        text = _ocr_image_bytes(pix.tobytes("png"))
                        backend = "Tesseract" if tesseract_available() else "EasyOCR"
                        notes.append(f"page {i}: OCR used ({backend})")
                    except Exception as exc:
                        notes.append(f"page {i}: OCR failed ({exc})")
                else:
                    notes.append(f"page {i}: no text layer; install EasyOCR or Tesseract to OCR scanned pages (page skipped)")
            pages.append((f"page {i}", i, text))
    return pages, "; ".join(notes)


def _extract_csv(data: bytes) -> list[tuple[str, int | None, str]]:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding_errors="replace")
    if df.empty:
        return []
    rows = []
    for i, rec in enumerate(df.to_dict("records"), start=1):
        text = "; ".join(f"{k}: {v}" for k, v in rec.items() if str(v).strip())
        rows.append((f"row {i}", i, text))
    return rows


def process_bytes(name: str, data: bytes, source: str = "upload", chunk_words: int = 90,
                  overlap_words: int = 20) -> tuple[ProcessedDocument, list[dict]]:
    """Process one file. Never raises: problems are reported through the returned record."""
    ext = Path(name).suffix.lower()
    doc = ProcessedDocument(doc_id=content_hash(data)[:12], name=Path(name).name,
                            file_type=ext.lstrip(".") or "unknown", source=source)
    if ext not in SUPPORTED:
        doc.status, doc.message = "unsupported", f"Unsupported file type '{ext}'. Allowed: PDF, TXT, DOCX, CSV."
        return doc, []
    if not data:
        doc.status, doc.message = "empty", "File is empty (0 bytes)."
        return doc, []
    try:
        if ext == ".txt":
            pages = _extract_txt(data)
        elif ext == ".docx":
            pages = _extract_docx(data)
        elif ext == ".pdf":
            pages, note = _extract_pdf(data)
            doc.message = note
        else:
            pages = _extract_csv(data)
    except Exception as exc:  # malformed / encrypted / corrupt file
        LOG.exception("Extraction failed for %s", name)
        doc.status, doc.message = "error", f"Extraction failed: {type(exc).__name__}: {exc}"
        return doc, []

    doc.raw_chars = sum(len(p[2] or "") for p in pages)
    doc.raw_preview = (pages[0][2] or "")[:600]
    cleaned = [(loc, no, clean_text(txt)) for loc, no, txt in pages]
    cleaned = [p for p in cleaned if p[2]]
    doc.pages = cleaned
    doc.clean_preview = cleaned[0][2][:600] if cleaned else ""
    doc.n_pages = len(pages)
    doc.text_length = sum(len(p[2]) for p in cleaned)
    doc.n_words = sum(len(p[2].split()) for p in cleaned)
    if not cleaned:
        doc.status = "empty"
        doc.message = (doc.message + " " if doc.message else "") + "No extractable text found."
        return doc, []
    if doc.message:
        doc.status = "warning"

    chunks: list[dict] = []
    for loc, no, text in cleaned:
        # CSV rows are already atomic records: one chunk per row, no overlap needed.
        parts = [text] if ext == ".csv" else chunk_text(text, chunk_words, overlap_words)
        for part in parts:
            chunks.append({"doc_id": doc.doc_id, "document": doc.name, "file_type": doc.file_type,
                           "chunk_id": f"{doc.doc_id}-{len(chunks):03d}", "page": no if no is not None else pd.NA,
                           "location": loc if loc != "document" else "-", "text": part, "n_words": len(part.split())})
    doc.n_chunks = len(chunks)
    return doc, chunks
