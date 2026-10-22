
"""Google Gemini API integration for the AI Knowledge Assistant.

Reads GEMINI_API_KEY from the project-root .env file.
Uses Gemini only; no NVIDIA provider.
"""

from __future__ import annotations

import logging
import os
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval_engine import RetrievalEngine

LOG = logging.getLogger("iki.gemini_assistant")

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_BACKUP_MODEL = "gemini-3.5-flash-lite"

MAX_OUT_TOKENS = 1500
TOP_K_CHUNKS = 8
MIN_SCORE = {"tfidf": 0.05, "lsa": 0.40, "semantic": 0.22}
MAX_CONTEXT_CHARS = 8000

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert industrial knowledge assistant embedded in a plant operations platform.
    Your answers must always be grounded in the provided document excerpts.
    Never invent specifications, maintenance records, inspection findings, equipment details,
    or regulatory requirements that are not present in the evidence.

    Response style:
    - Write in clear, professional prose.
    - Use headings for multi-part answers.
    - Use numbered lists for procedures and bullet points for findings.
    - Give direct answers to simple factual questions.
    - Do not pad answers with generic disclaimers.

    Citations:
    - Cite key claims using the source reference or excerpt number, such as [1].
    - If evidence is insufficient, explain what information is missing.

    Critical rules:
    - Never fabricate equipment IDs, failure modes, work orders, inspection dates,
      or regulatory clause numbers.
    - If evidence is contradictory, note the discrepancy.
    - If uncertain, say so.
""")


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Load the API key from the project-root .env file."""
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass

    return os.environ.get("GEMINI_API_KEY", "").strip() or None


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

_conn_cache: dict | None = None


def test_connection(force: bool = False) -> dict:
    """Test the primary Gemini model and cache the result."""
    global _conn_cache

    if _conn_cache is not None and not force:
        return _conn_cache

    key = _get_api_key()

    if not key:
        result = {
            "ok": False,
            "status": "missing",
            "model": GEMINI_MODEL,
            "message": (
                "GEMINI_API_KEY is missing. Add GEMINI_API_KEY=your-key "
                "to the .env file in the project root."
            ),
        }
        _conn_cache = result
        return result

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: OK",
            config={"max_output_tokens": 8},
        )

        if not (response.text or "").strip():
            raise RuntimeError("Gemini returned an empty response.")

        result = {
            "ok": True,
            "status": "connected",
            "model": GEMINI_MODEL,
            "message": f"Gemini connected · model: {GEMINI_MODEL}",
        }

    except ImportError:
        result = {
            "ok": False,
            "status": "missing_sdk",
            "model": GEMINI_MODEL,
            "message": (
                "google-genai is not installed. Run: "
                "pip install google-genai"
            ),
        }

    except Exception as exc:
        message_lower = str(exc).lower()

        if any(word in message_lower for word in (
            "429", "quota", "resource_exhausted", "resource exhausted"
        )):
            status = "quota"
            message = "Gemini quota or rate limit reached. Try again later."
        elif any(word in message_lower for word in (
            "503", "unavailable", "high demand", "overloaded",
            "service unavailable"
        )):
            status = "unavailable"
            message = (
                "Gemini is temporarily unavailable. The assistant will "
                "retry Gemini and try the backup Gemini model."
            )
        elif any(word in message_lower for word in (
            "api_key", "api key", "401", "403",
            "permission denied", "invalid_argument"
        )):
            status = "auth_error"
            message = (
                "Google rejected the Gemini request. Check GEMINI_API_KEY "
                "and whether the selected model is available to your key."
            )
        elif any(word in message_lower for word in (
            "network", "connect", "timeout", "name or service"
        )):
            status = "network"
            message = "Cannot reach Gemini. Check your internet connection."
        else:
            status = "unknown"
            message = (
                f"Gemini connection failed ({type(exc).__name__}): "
                f"{str(exc)[:200]}"
            )

        result = {
            "ok": False,
            "status": status,
            "model": GEMINI_MODEL,
            "message": message,
        }

    _conn_cache = result
    LOG.info(
        "Gemini connection test: status=%s model=%s",
        result["status"],
        result["model"],
    )
    return result


def reset_connection_cache() -> None:
    """Clear the cached connection result."""
    global _conn_cache
    _conn_cache = None


def is_available() -> bool:
    """Return whether the primary Gemini model passed its connection test."""
    return test_connection().get("ok", False)


def status_message() -> tuple[str, str]:
    """Return the CSS class and human-readable connection message."""
    result = test_connection()
    status = result["status"]

    if status == "connected":
        return "ready", result["message"]
    if status == "missing":
        return "missing", result["message"]
    if status == "quota":
        return "quota", result["message"]

    return "error", result["message"]


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------

def _build_context(hits) -> tuple[str, list[dict]]:
    """Build numbered evidence excerpts and citation metadata."""
    citations: list[dict] = []
    parts: list[str] = []
    total_chars = 0

    for rank, row in enumerate(hits.itertuples(index=False), start=1):
        excerpt = (row.excerpt or "").strip()

        if not excerpt:
            continue

        snippet = excerpt[:800]
        block = (
            f"[{rank}] Source: {row.document} | "
            f"Location: {row.location}\n{snippet}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total_chars += len(block)

        citations.append({
            "n": rank,
            "document": row.document,
            "location": row.location,
            "score": float(row.score),
            "excerpt": excerpt,
            "chunk_id": row.chunk_id,
        })

    return "\n\n---\n\n".join(parts), citations


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------

def _local_fallback(
    engine,
    question: str,
    mode: str,
    base: dict,
) -> dict:
    """Use local evidence only if Gemini generation fails."""
    from .knowledge_assistant import answer_question

    result = answer_question(engine, question, mode)
    result["mode"] = "extractive_fallback"
    result["error"] = base.get("error")
    result["notes"] = (
        (result.get("notes") or []) + (base.get("notes") or [])
    )
    result["citations"] = (
        result.get("citations") or base.get("citations", [])
    )

    return result


# ---------------------------------------------------------------------------
# Main Gemini question-answering function
# ---------------------------------------------------------------------------

def ask_gemini(
    question: str,
    engine: "RetrievalEngine",
    mode: str = "tfidf",
    conversation_history: list[dict] | None = None,
) -> dict:
    """Answer with Gemini, retrying failures and trying a backup Gemini model."""

    base: dict = {
        "question": question,
        "answer": "",
        "citations": [],
        "sufficient": False,
        "notes": [],
        "error": None,
        "mode": "gemini",
    }

    # 1. Retrieve relevant evidence.
    if engine is None or engine.chunks.empty:
        base.update(
            answer="The knowledge base is empty. Upload documents first.",
            mode="extractive_fallback",
        )
        return base

    hits = engine.search(question, mode, top_k=TOP_K_CHUNKS)
    threshold = MIN_SCORE.get(mode, 0.05)

    if hits.empty or hits.iloc[0]["score"] < threshold:
        best_score = (
            float(hits.iloc[0]["score"]) if not hits.empty else 0.0
        )
        base["answer"] = (
            "I could not find enough relevant evidence in the loaded "
            f"documents. Best relevance score: {best_score:.3f} "
            f"(threshold {threshold}). Try rephrasing the question or "
            "uploading relevant documents."
        )
        base["mode"] = "extractive_fallback"
        return base

    context_text, citations = _build_context(hits)
    base["citations"] = citations

    # 2. Read the API key. Do not let a failed connection-test request
    # prevent the actual answer-generation request from being attempted.
    api_key = _get_api_key()

    if not api_key:
        base["error"] = (
            "GEMINI_API_KEY is missing. Add it to the project-root .env file."
        )
        base["notes"].append(
            "Gemini is not configured; showing local evidence."
        )
        return _local_fallback(engine, question, mode, base)

    # 3. Prepare the question and evidence.
    user_message = (
        "Answer the question using ONLY the document excerpts below. "
        "Do not invent maintenance actions, dates, measurements, "
        "specifications, or other facts. If the excerpts do not answer "
        "the question, explain what is missing. Cite claims by excerpt "
        "number, for example [1].\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{context_text}"
    )

    contents: list[dict] = []

    if conversation_history:
        for turn in conversation_history[-6:]:
            role = turn.get("role", "")
            content = turn.get("content", "")

            if role == "user" and content:
                contents.append({
                    "role": "user",
                    "parts": [{"text": str(content)}],
                })
            elif role in ("assistant", "model") and content:
                contents.append({
                    "role": "model",
                    "parts": [{"text": str(content)}],
                })

    contents.append({
        "role": "user",
        "parts": [{"text": user_message}],
    })

    # 4. Import Gemini SDK.
    try:
        from google import genai
    except ImportError:
        base["error"] = (
            "The google-genai package is missing. Install it using "
            "the project's virtual environment: pip install google-genai"
        )
        base["notes"].append(
            "Gemini SDK is unavailable; showing local evidence."
        )
        return _local_fallback(engine, question, mode, base)

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        base["error"] = (
            f"Could not initialize Gemini ({type(exc).__name__}): "
            f"{str(exc)[:200]}"
        )
        return _local_fallback(engine, question, mode, base)

    # 5. Try the primary Gemini model, then a backup Gemini model.
    primary_model = GEMINI_MODEL
    backup_model = GEMINI_BACKUP_MODEL

    models_to_try = [primary_model]
    if backup_model and backup_model != primary_model:
        models_to_try.append(backup_model)

    errors: list[str] = []

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "max_output_tokens": MAX_OUT_TOKENS,
                        "temperature": 0.2,
                    },
                )

                answer = (response.text or "").strip()

                if not answer:
                    raise RuntimeError(
                        f"{model_name} returned an empty answer."
                    )

                base["answer"] = answer
                base["sufficient"] = len(answer) > 40
                base["mode"] = "gemini"

                if model_name != primary_model:
                    base["notes"].append(
                        "Primary model failed; answer generated by backup "
                        f"Gemini model {model_name}."
                    )

                if any(
                    str(citation.get("document", "")).lower().endswith(".csv")
                    for citation in citations
                ):
                    base["notes"].append(
                        "Some evidence comes from structured CSV records."
                    )

                return base

            except Exception as exc:
                error_text = str(exc)
                errors.append(
                    f"{model_name}: {type(exc).__name__}: {error_text[:250]}"
                )

                LOG.warning(
                    "Gemini generation failed for %s, attempt %s: %s",
                    model_name,
                    attempt + 1,
                    error_text[:250],
                )

                lower_message = error_text.lower()
                transient = any(
                    marker in lower_message
                    for marker in (
                        "429", "500", "502", "503", "504",
                        "quota", "resource_exhausted", "overloaded",
                        "unavailable", "timeout", "timed out",
                        "connection",
                    )
                )

                # Retry temporary errors once, then try the backup model.
                if attempt == 0 and transient:
                    time.sleep(2)
                    continue

                # Authentication, invalid model, and other non-transient
                # errors move directly to the next Gemini model.
                break

    # 6. Both Gemini models failed. Never call NVIDIA or another provider.
    base["error"] = "Gemini generation failed: " + " | ".join(errors)
    base["notes"].append(
        "Both Gemini model attempts failed. Showing local document evidence; "
        "no other AI provider was used."
    )

    return _local_fallback(engine, question, mode, base)
