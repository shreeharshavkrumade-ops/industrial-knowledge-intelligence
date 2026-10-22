"""NVIDIA NIM API integration for the AI Knowledge Assistant.

Uses the NVIDIA-hosted NIM endpoint (OpenAI-compatible).
Reads NVIDIA_API_KEY from environment (via python-dotenv .env file).
Falls back to local extractive QA when the key is absent, invalid, or
when the account does not have inference access.

Key design rules
----------------
- status_message() NEVER claims "connected" based on key format alone.
  It only claims connected after a live ping has succeeded this session.
- 403 PermissionDeniedError is handled separately from 401 AuthenticationError.
  403 = account/key lacks inference access (playground key, TOS not accepted, etc.)
  401 = key itself is wrong or expired.
- The API key is NEVER logged, printed, or exposed to the browser.
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval_engine import RetrievalEngine

LOG = logging.getLogger("iki.nvidia_assistant")

# ---------------------------------------------------------------------------
# NVIDIA NIM configuration
# ---------------------------------------------------------------------------
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = "nvidia/llama-3.1-nemotron-70b-instruct"

# Retrieval settings
TOP_K_CHUNKS = 8
MIN_SCORE    = {"tfidf": 0.05, "lsa": 0.40, "semantic": 0.22}

# Prompt
MAX_CONTEXT_CHARS = 8000
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert industrial knowledge assistant embedded in a plant operations platform.
    Your answers are ALWAYS grounded in the provided document excerpts — never invent
    specifications, maintenance records, inspection findings, equipment details, or regulatory
    requirements not present in the evidence.

    ## Response style
    - Write in clear, professional prose. Use **headings** for multi-part answers.
    - For procedures or ordered steps, use **numbered lists**. For lists of items or findings,
      use bullet points.
    - Aim for 3–6 concise paragraphs for technical questions. For simple factual questions,
      a short direct answer is better than a padded long one.
    - Do NOT pad your answer with generic disclaimers or repeat the question back.

    ## Citations
    - Cite each key claim with its source reference inline (e.g. [pump_manual.pdf, p.3] or [2]).
    - If evidence is insufficient, explicitly state what information is missing and why you
      cannot fully answer. Then give any general guidance that is well-supported by the evidence.

    ## Critical rules
    - Never fabricate equipment IDs, failure modes, work orders, inspection dates, or
      regulatory clause numbers that are not present in the excerpts.
    - If the evidence is contradictory, note the discrepancy.
    - If uncertain, say so. Do NOT claim certainty you do not have.
""")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Load NVIDIA_API_KEY from .env and return it, or None if absent.

    Uses an explicit path so the key is found regardless of the working
    directory the app was launched from.
    """
    try:
        import pathlib
        from dotenv import load_dotenv
        env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    return os.environ.get("NVIDIA_API_KEY", "").strip() or None


# ---------------------------------------------------------------------------
# Connection test — cached per process
# ---------------------------------------------------------------------------

_conn_cache: dict | None = None   # None = not yet tested


def test_connection(force: bool = False) -> dict:
    """Send a minimal inference request to verify key and model access.

    Returns
    -------
    {
      "ok":      bool,
      "status":  "connected"|"missing"|"forbidden"|"auth_error"|
                 "rate_limit"|"network"|"unknown",
      "message": str,   # human-readable; never contains the key value
      "model":   str,
    }

    The result is cached (one real call per process lifetime).
    Pass force=True to re-test.
    """
    global _conn_cache
    if _conn_cache is not None and not force:
        return _conn_cache

    key = _get_api_key()
    if not key:
        r = {
            "ok": False, "status": "missing", "model": NIM_MODEL,
            "message": (
                "NVIDIA_API_KEY is not set. "
                "Create a .env file in the project root with NVIDIA_API_KEY=nvapi-…"
            ),
        }
        _conn_cache = r
        return r

    if not key.startswith("nvapi-"):
        r = {
            "ok": False, "status": "auth_error", "model": NIM_MODEL,
            "message": (
                "NVIDIA_API_KEY does not start with 'nvapi-'. "
                "Check that you copied the full key from build.nvidia.com."
            ),
        }
        _conn_cache = r
        return r

    try:
        from openai import (
            OpenAI, APIConnectionError, AuthenticationError,
            PermissionDeniedError, RateLimitError,
        )
        client = OpenAI(api_key=key, base_url=NIM_BASE_URL)
        client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=8,
            temperature=0,
        )
        r = {
            "ok": True, "status": "connected", "model": NIM_MODEL,
            "message": f"NVIDIA NIM connected · model: {NIM_MODEL}",
        }

    except PermissionDeniedError:
        r = {
            "ok": False, "status": "forbidden", "model": NIM_MODEL,
            "message": (
                "403 Forbidden — your API key is recognised but does not have "
                "inference access for this endpoint. "
                "Possible causes: (1) this is a playground-only key, "
                "(2) NVIDIA account terms of service have not been accepted, or "
                "(3) the key was generated for a different NVIDIA service (e.g. NGC) "
                "rather than the NIM inference API. "
                "Visit build.nvidia.com → your account → API Keys to check and "
                "generate a new key with inference permissions."
            ),
        }

    except AuthenticationError:
        r = {
            "ok": False, "status": "auth_error", "model": NIM_MODEL,
            "message": (
                "401 Unauthorized — API key rejected. "
                "Check that NVIDIA_API_KEY in your .env is the full key "
                "and has not expired."
            ),
        }

    except RateLimitError:
        r = {
            "ok": False, "status": "rate_limit", "model": NIM_MODEL,
            "message": (
                "429 Rate limit — key is valid but the account quota is exhausted. "
                "Wait a moment and retry, or check your NIM usage at build.nvidia.com."
            ),
        }

    except APIConnectionError as exc:
        r = {
            "ok": False, "status": "network", "model": NIM_MODEL,
            "message": f"Cannot reach NVIDIA NIM — check internet connection. ({exc})",
        }

    except Exception as exc:  # noqa: BLE001
        r = {
            "ok": False, "status": "unknown", "model": NIM_MODEL,
            "message": f"Unexpected error ({type(exc).__name__}): {str(exc)[:200]}",
        }

    _conn_cache = r
    LOG.info("NIM connection test: status=%s", r["status"])
    return r


def reset_connection_cache() -> None:
    """Invalidate cached connection result so next call re-tests."""
    global _conn_cache
    _conn_cache = None


# ---------------------------------------------------------------------------
# Public status helpers (called by the page)
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True only when a live connection test has succeeded this session."""
    return test_connection().get("ok", False)


def status_message() -> tuple[str, str]:
    """Return (css_class, human-readable message) for the UI status banner.

    css_class is one of: 'ready' | 'missing' | 'forbidden' | 'error'

    This performs a live test on first call — never assumes "ready" from
    key format alone.
    """
    r = test_connection()
    s = r["status"]
    if s == "connected":
        return ("ready",     r["message"])
    if s == "missing":
        return ("missing",   r["message"])
    if s == "forbidden":
        return ("forbidden", r["message"])
    return ("error", r["message"])


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------

def _build_context(hits) -> tuple[str, list[dict]]:
    citations: list[dict] = []
    parts: list[str] = []
    total = 0
    for rank, row in enumerate(hits.itertuples(index=False), start=1):
        excerpt = (row.excerpt or "").strip()
        if not excerpt:
            continue
        snippet = excerpt[:800]
        block   = f"[{rank}] Source: {row.document} | Location: {row.location}\n{snippet}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
        citations.append({
            "n":        rank,
            "document": row.document,
            "location": row.location,
            "score":    float(row.score),
            "excerpt":  excerpt,
            "chunk_id": row.chunk_id,
        })
    return "\n\n---\n\n".join(parts), citations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_nvidia(
    question: str,
    engine: "RetrievalEngine",
    mode: str = "tfidf",
    conversation_history: list[dict] | None = None,
) -> dict:
    """Query NVIDIA NIM with evidence from the local knowledge base.

    Always falls back gracefully to local extractive QA on any API error.
    The fallback is clearly labelled — never silently presented as NIM output.
    """
    base = {
        "question":   question,
        "answer":     "",
        "citations":  [],
        "sufficient": False,
        "notes":      [],
        "error":      None,
        "mode":       "nvidia_nim",
    }

    # 1. Retrieve evidence ------------------------------------------------
    if engine is None or engine.chunks.empty:
        base.update(answer="The knowledge base is empty. Upload documents first.",
                    mode="extractive_fallback")
        return base

    hits = engine.search(question, mode, top_k=TOP_K_CHUNKS)
    thr  = MIN_SCORE.get(mode, 0.05)
    if hits.empty or hits.iloc[0]["score"] < thr:
        base["answer"] = (
            "I could not find relevant evidence in the loaded documents. "
            f"Best relevance score: {hits.iloc[0]['score']:.3f} (threshold {thr}). "
            "Try rephrasing, or upload more relevant documents."
        )
        return base

    context_text, citations = _build_context(hits)
    base["citations"] = citations

    # 2. Check connection (cached) ----------------------------------------
    conn = test_connection()
    if not conn["ok"]:
        from .knowledge_assistant import answer_question
        result          = answer_question(engine, question, mode)
        result["mode"]  = "extractive_fallback"
        result["error"] = conn["message"]
        result["notes"] = (result.get("notes") or []) + [
            "NVIDIA NIM unavailable — showing local extractive answer. "
            f"Reason: {conn['status']}."
        ]
        return result

    # 3. Build prompt -----------------------------------------------------
    user_msg = (
        "Use ONLY the following document excerpts to answer the question. "
        "Cite each excerpt by its number [N].\n\n"
        f"QUESTION: {question}\n\nEVIDENCE:\n{context_text}"
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    # 4. Call NIM ---------------------------------------------------------
    try:
        from openai import (
            OpenAI, APIConnectionError, AuthenticationError,
            PermissionDeniedError, RateLimitError,
        )
        client   = OpenAI(api_key=_get_api_key(), base_url=NIM_BASE_URL)
        response = client.chat.completions.create(
            model=NIM_MODEL, messages=messages, temperature=0.20, max_tokens=2048,
        )
        answer = response.choices[0].message.content.strip()
        base["answer"]    = answer
        base["sufficient"] = bool(answer and len(answer) > 40)
        base["mode"]      = "nvidia_nim"
        if any(c["document"].endswith(".csv") for c in citations):
            base["notes"].append(
                "Some evidence comes from structured CSV records (one row per citation)."
            )
        return base

    except PermissionDeniedError:
        err = conn["message"]  # use the detailed message from the test
        reset_connection_cache()
    except AuthenticationError:
        err = "401 Unauthorized — API key rejected. Check NVIDIA_API_KEY in .env."
        reset_connection_cache()
    except RateLimitError:
        err = "429 Rate limit — wait a moment and try again."
    except APIConnectionError as exc:
        err = f"Cannot reach NVIDIA NIM — check internet connection. ({exc})"
    except Exception as exc:  # noqa: BLE001
        err = f"NVIDIA NIM error ({type(exc).__name__}): {str(exc)[:200]}"

    LOG.error("NIM call failed: %s", err)
    from .knowledge_assistant import answer_question
    fallback          = answer_question(engine, question, mode)
    fallback["error"] = err
    fallback["mode"]  = "extractive_fallback"
    fallback["notes"] = (fallback.get("notes") or []) + [
        "NIM API unavailable — showing local extractive answer.",
    ]
    return fallback
