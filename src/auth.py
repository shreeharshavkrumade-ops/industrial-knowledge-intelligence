"""Secure authentication module for Industrial Knowledge Intelligence.

Architecture
------------
- SQLite database (data/auth.db) — separate from knowledge.db
- Argon2id password hashing via argon2-cffi
- Server-side session tokens (UUID4, stored in auth.db sessions table)
- Rate limiting: per-identifier failed-attempt tracking with exponential lockout
- email-validator for RFC-5322 email normalisation + DNS check (optional)

Security notes
--------------
- Passwords are never stored, logged, or exposed. Only the Argon2id hash is kept.
- Session tokens are stored server-side; their presence in session_state is ONLY
  an index into the DB. Every page validates the token against the DB on each run.
- Username/email existence is never revealed through login error messages.
- Parameterised SQL throughout — no string interpolation into queries.
- Auth DB lives in data/ (excluded from git via .gitignore).
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("iki.auth")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
AUTH_DB  = ROOT / "data" / "auth.db"

# ---------------------------------------------------------------------------
# Security parameters
# ---------------------------------------------------------------------------
SESSION_TTL_HOURS   = 8          # sessions expire after 8 h of inactivity
SESSION_MAX_HOURS   = 24         # hard cap: re-login required after 24 h
TOKEN_BYTES         = 32         # 256-bit session token
MAX_FAILED          = 5          # lockout after N consecutive failures
LOCKOUT_SECONDS     = 300        # 5-minute lockout (first trigger)
LOCKOUT_MULTIPLIER  = 2.0        # each subsequent lock doubles
MAX_LOCKOUT_SECONDS = 3600       # cap at 1 hour

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
PASSWORD_MIN_LEN = 12
PASSWORD_MAX_LEN = 512           # bcrypt/argon2 truncation guard

# ---------------------------------------------------------------------------
# Common passwords (top 30 — extend as needed)
# ---------------------------------------------------------------------------
_COMMON_PASSWORDS = {
    "password123456", "password1234567", "123456789012", "qwertyuiop12",
    "letmein123456", "welcome123456", "monkey123456", "dragon123456",
    "master123456", "superman12345", "batman123456", "football12345",
    "iloveyou12345", "trustno11234", "sunshine12345", "princess12345",
    "admin123456789", "login123456789", "test1234567890", "hello123456789",
    "changeme123456", "secret123456789", "passw0rd123456", "qwerty123456789",
    "abcd1234567890", "1234567890ab", "password!12345", "pass1234567890",
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _connect():
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create auth tables if they do not exist. Safe to call multiple times."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                username_norm TEXT NOT NULL UNIQUE,
                email         TEXT NOT NULL UNIQUE,
                email_norm    TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token         TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                created_at    TEXT NOT NULL,
                last_seen     TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                hard_expires  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rate_limit (
                identifier    TEXT PRIMARY KEY,
                attempts      INTEGER NOT NULL DEFAULT 0,
                first_attempt TEXT NOT NULL,
                last_attempt  TEXT NOT NULL,
                locked_until  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
        """)
    LOG.debug("Auth DB initialised at %s", AUTH_DB)


# ---------------------------------------------------------------------------
# Argon2id — password hashing
# ---------------------------------------------------------------------------

def _hasher():
    from argon2 import PasswordHasher
    # OWASP-recommended Argon2id parameters (2024):
    # time_cost=3, memory_cost=65536 (64 MiB), parallelism=4
    return PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4,
                          hash_len=32, salt_len=16)


def hash_password(plaintext: str) -> str:
    return _hasher().hash(plaintext)


def verify_password(plaintext: str, stored_hash: str) -> bool:
    try:
        return _hasher().verify(stored_hash, plaintext)
    except Exception:
        return False


def needs_rehash(stored_hash: str) -> bool:
    return _hasher().check_needs_rehash(stored_hash)


# ---------------------------------------------------------------------------
# Email normalisation
# ---------------------------------------------------------------------------

def normalize_email(raw: str) -> str:
    """Validate and normalise an email address. Raises ValueError on bad input."""
    from email_validator import validate_email, EmailNotValidError
    try:
        info = validate_email(raw.strip(), check_deliverability=False)
        return info.normalized
    except EmailNotValidError as exc:
        raise ValueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Username normalisation
# ---------------------------------------------------------------------------

def normalize_username(raw: str) -> str:
    """NFKC normalise and lower-case a username for uniqueness checks."""
    return unicodedata.normalize("NFKC", raw.strip()).lower()


# ---------------------------------------------------------------------------
# Password strength
# ---------------------------------------------------------------------------

def check_password_strength(password: str) -> tuple[int, list[str]]:
    """Return (score 0-4, list-of-feedback-messages).

    Score: 0=very weak, 1=weak, 2=fair, 3=strong, 4=very strong.
    """
    issues: list[str] = []
    score = 0

    if len(password) < PASSWORD_MIN_LEN:
        issues.append(f"At least {PASSWORD_MIN_LEN} characters required.")
    if len(password) > PASSWORD_MAX_LEN:
        issues.append(f"Password too long (max {PASSWORD_MAX_LEN} characters).")
        return 0, issues

    if password.lower() in _COMMON_PASSWORDS:
        issues.append("This is a very common password — please choose something unique.")
        return 0, issues

    has_upper   = bool(re.search(r"[A-Z]",  password))
    has_lower   = bool(re.search(r"[a-z]",  password))
    has_digit   = bool(re.search(r"\d",     password))
    has_special = bool(re.search(r"[^A-Za-z0-9]", password))

    types = sum([has_upper, has_lower, has_digit, has_special])
    length = len(password)

    if length >= PASSWORD_MIN_LEN:
        score += 1
    if types >= 2:
        score += 1
    if types >= 3 or length >= 20:
        score += 1
    if types == 4 or length >= 30:
        score += 1

    if types < 2:
        missing = []
        if not (has_upper or has_lower): missing.append("letters")
        elif not has_upper:              missing.append("uppercase letters")
        elif not has_lower:              missing.append("lowercase letters")
        if not has_digit:                missing.append("numbers")
        if not has_special:              missing.append("special characters (e.g. !@#$)")
        if missing:
            issues.append(f"Add {' and '.join(missing)} to strengthen your password.")

    return score, issues


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_rate_limit(identifier: str) -> tuple[bool, str]:
    """Return (is_allowed, reason_if_blocked)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT attempts, locked_until FROM rate_limit WHERE identifier=?",
            (identifier,)
        ).fetchone()

    if row is None:
        return True, ""

    attempts, locked_until = row
    if locked_until:
        lock_dt = datetime.fromisoformat(locked_until)
        now_dt  = datetime.now(timezone.utc)
        if now_dt < lock_dt:
            wait = int((lock_dt - now_dt).total_seconds())
            return False, f"Too many failed attempts. Please wait {wait} second(s) before trying again."

    return True, ""


def _record_failure(identifier: str) -> None:
    now = _now_iso()
    with _connect() as conn:
        row = conn.execute(
            "SELECT attempts, locked_until FROM rate_limit WHERE identifier=?",
            (identifier,)
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO rate_limit(identifier,attempts,first_attempt,last_attempt) VALUES(?,1,?,?)",
                (identifier, now, now)
            )
            return

        attempts = row[0] + 1
        # Calculate lockout duration with multiplier
        lock_secs = min(
            LOCKOUT_SECONDS * (LOCKOUT_MULTIPLIER ** max(0, (attempts - MAX_FAILED))),
            MAX_LOCKOUT_SECONDS,
        )
        locked_until = None
        if attempts >= MAX_FAILED:
            locked_until = (datetime.now(timezone.utc) + timedelta(seconds=lock_secs)).isoformat()
        conn.execute(
            "UPDATE rate_limit SET attempts=?, last_attempt=?, locked_until=? WHERE identifier=?",
            (attempts, now, locked_until, identifier)
        )


def _clear_rate_limit(identifier: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM rate_limit WHERE identifier=?", (identifier,))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_user(username: str, email: str, password: str) -> dict[str, Any]:
    """Register a new user.

    Returns {"ok": True, "user_id": str} or {"ok": False, "error": str}.
    Never reveals whether a username/email already exists in a distinguishable way
    for callers that care, but the UI shows validation feedback to assist the user.
    """
    # --- validate username ---------------------------------------------------
    raw_username = (username or "").strip()
    if not USERNAME_RE.match(raw_username):
        return {"ok": False, "error":
                "Username must be 3–30 characters and contain only letters, numbers, and underscores."}
    username_norm = normalize_username(raw_username)

    # --- validate email ------------------------------------------------------
    try:
        email_norm = normalize_email(email or "")
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    # --- validate password ---------------------------------------------------
    if not password:
        return {"ok": False, "error": "Password is required."}
    score, issues = check_password_strength(password)
    if score == 0 or issues:
        return {"ok": False, "error": " ".join(issues) if issues else "Password is too weak."}

    # --- hash password -------------------------------------------------------
    pw_hash = hash_password(password)

    # --- persist -------------------------------------------------------------
    user_id  = secrets.token_hex(16)
    now      = _now_iso()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users(user_id,username,username_norm,email,email_norm,password_hash,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (user_id, raw_username, username_norm, email_norm, email_norm, pw_hash, now)
            )
        LOG.info("User registered: username_norm=%s", username_norm)
        return {"ok": True, "user_id": user_id}
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "username" in msg:
            return {"ok": False, "error": "That username is already taken. Please choose a different one."}
        if "email" in msg:
            return {"ok": False, "error": "That email address is already registered."}
        return {"ok": False, "error": "Registration failed due to a conflict. Please try again."}
    except Exception as exc:
        LOG.exception("Registration failed for username_norm=%s", username_norm)
        return {"ok": False, "error": f"Registration failed: {type(exc).__name__}"}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login_user(identifier: str, password: str, ip: str = "unknown") -> dict[str, Any]:
    """Authenticate a user by username-or-email + password.

    Returns {"ok": True, "token": str, "username": str, "user_id": str}
         or {"ok": False, "error": str}.

    IMPORTANT: error messages never reveal whether the identifier exists.
    """
    init_db()
    identifier = (identifier or "").strip()
    rate_key   = f"{ip}:{identifier.lower()}"

    # --- rate limit check ----------------------------------------------------
    allowed, reason = _check_rate_limit(rate_key)
    if not allowed:
        return {"ok": False, "error": reason}

    # --- look up user --------------------------------------------------------
    with _connect() as conn:
        # Try username match first, then email
        row = conn.execute(
            "SELECT user_id, username, password_hash, is_active "
            "FROM users WHERE username_norm=? OR email_norm=?",
            (normalize_username(identifier), identifier.lower())
        ).fetchone()

    if row is None:
        # Still do a dummy hash to prevent timing attacks
        _dummy_verify()
        _record_failure(rate_key)
        return {"ok": False, "error": "Invalid credentials. Please check your username/email and password."}

    user_id, username, pw_hash, is_active = row

    if not is_active:
        _record_failure(rate_key)
        return {"ok": False, "error": "Invalid credentials. Please check your username/email and password."}

    if not verify_password(password, pw_hash):
        _record_failure(rate_key)
        return {"ok": False, "error": "Invalid credentials. Please check your username/email and password."}

    # --- success -------------------------------------------------------------
    _clear_rate_limit(rate_key)
    token = _create_session(user_id)

    # Re-hash if parameters have been upgraded
    if needs_rehash(pw_hash):
        try:
            new_hash = hash_password(password)
            with _connect() as conn:
                conn.execute("UPDATE users SET password_hash=? WHERE user_id=?", (new_hash, user_id))
        except Exception:
            pass  # non-critical

    LOG.info("Login successful: username=%s", username)
    return {"ok": True, "token": token, "username": username, "user_id": user_id}


def _dummy_verify() -> None:
    """Perform a dummy Argon2 verification to prevent timing-based user enumeration."""
    try:
        _hasher().verify(
            "$argon2id$v=19$m=65536,t=3,p=4$aaaaaaaaaaaaaaaa$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "dummy"
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _create_session(user_id: str) -> str:
    token       = secrets.token_urlsafe(TOKEN_BYTES)
    now         = datetime.now(timezone.utc)
    expires_at  = now + timedelta(hours=SESSION_TTL_HOURS)
    hard_expires= now + timedelta(hours=SESSION_MAX_HOURS)
    now_iso     = now.isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(token,user_id,created_at,last_seen,expires_at,hard_expires) "
            "VALUES(?,?,?,?,?,?)",
            (token, user_id, now_iso, now_iso, expires_at.isoformat(), hard_expires.isoformat())
        )
    return token


def validate_session(token: str | None) -> dict[str, Any] | None:
    """Validate a session token against the database.

    Returns user info dict if the token is valid and not expired, else None.
    Updates last_seen and extends the inactivity window on each valid call.
    """
    if not token:
        return None
    now = datetime.now(timezone.utc)
    with _connect() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.expires_at, s.hard_expires, u.username "
            "FROM sessions s JOIN users u ON s.user_id=u.user_id "
            "WHERE s.token=? AND u.is_active=1",
            (token,)
        ).fetchone()
        if row is None:
            return None

        user_id, expires_at_str, hard_expires_str, username = row
        expires_at   = datetime.fromisoformat(expires_at_str)
        hard_expires = datetime.fromisoformat(hard_expires_str)

        if now >= hard_expires or now >= expires_at:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            return None

        # Extend inactivity window
        new_expires = now + timedelta(hours=SESSION_TTL_HOURS)
        if new_expires > hard_expires:
            new_expires = hard_expires
        conn.execute(
            "UPDATE sessions SET last_seen=?, expires_at=? WHERE token=?",
            (now.isoformat(), new_expires.isoformat(), token)
        )

    return {"user_id": user_id, "username": username, "token": token}


def logout(token: str | None) -> None:
    """Invalidate a session token server-side."""
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    LOG.info("Session logged out")


def purge_expired_sessions() -> None:
    """Delete all sessions past their hard expiry. Call periodically."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE hard_expires<=? OR expires_at<=?", (now, now))


# ---------------------------------------------------------------------------
# Auth guard — call at the top of every protected page
# ---------------------------------------------------------------------------

import streamlit as _st  # noqa: E402 — imported here to keep module importable without Streamlit


def require_auth() -> dict[str, Any]:
    """Enforce authentication on the calling page.

    - Validates the session token from st.session_state against the DB.
    - If invalid/missing: clears session_state and stops the page (shows nothing).
    - Returns the validated user dict on success.

    Usage (top of every protected page, before any other st.* call):
        user = require_auth()
    """
    init_db()
    token = _st.session_state.get("_auth_token")
    user  = validate_session(token)

    if user is None:
        # Wipe any stale session data
        for key in ("_auth_token", "_auth_user", "_auth_username"):
            _st.session_state.pop(key, None)
        _st.error("🔒 You must be logged in to view this page.")
        _st.info("Please use the **Login** option in the sidebar to sign in.")
        _st.stop()

    # Keep session_state in sync (username may have changed on rehash etc.)
    _st.session_state["_auth_user"]     = user["user_id"]
    _st.session_state["_auth_username"] = user["username"]
    return user
