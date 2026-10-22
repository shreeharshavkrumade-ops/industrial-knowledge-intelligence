"""Automated tests for the authentication module.

Tests cover: registration, login, session management, rate limiting,
password hashing, and security properties.
Run: pytest tests/test_auth.py -v
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

# Use an isolated test DB
TEST_DB = Path(__file__).parent / "test_auth.db"


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    """Redirect auth DB to a fresh tmp file for each test."""
    db = tmp_path / "auth_test.db"
    import src.auth as auth_mod
    monkeypatch.setattr(auth_mod, "AUTH_DB", db)
    auth_mod.init_db()
    yield db
    # cleanup handled by tmp_path fixture


# ---------------------------------------------------------------------------
# 1. Registration — valid account
# ---------------------------------------------------------------------------
def test_register_valid():
    from src.auth import register_user
    r = register_user("TestUser1", "test@example.com", "MyStr0ng!Password99")
    assert r["ok"], r.get("error")
    assert "user_id" in r


# ---------------------------------------------------------------------------
# 2. Registration — invalid username
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_name", [
    "ab",                   # too short
    "a" * 31,               # too long
    "bad name",             # space
    "user@name",            # special char
    "user-name",            # hyphen
    "",                     # empty
])
def test_register_invalid_username(bad_name):
    from src.auth import register_user
    r = register_user(bad_name, "valid@example.com", "MyStr0ng!Password99")
    assert not r["ok"]
    assert "error" in r


# ---------------------------------------------------------------------------
# 3. Registration — bad email
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_email", [
    "notanemail",
    "missing@",
    "@nodomain.com",
    "spaces in@email.com",
    "",
])
def test_register_invalid_email(bad_email):
    from src.auth import register_user
    r = register_user("gooduser1", bad_email, "MyStr0ng!Password99")
    assert not r["ok"]


# ---------------------------------------------------------------------------
# 4. Registration — weak password
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_pw", [
    "short",                # under min length
    "password123456",       # common password
    "12345678901234",       # all digits
])
def test_register_weak_password(bad_pw):
    from src.auth import register_user
    r = register_user("gooduser2", "good@example.com", bad_pw)
    assert not r["ok"]


# ---------------------------------------------------------------------------
# 5. Registration — duplicate username / email
# ---------------------------------------------------------------------------
def test_register_duplicate_username():
    from src.auth import register_user
    register_user("dupuser", "first@example.com", "MyStr0ng!Password99")
    r = register_user("dupuser", "second@example.com", "MyStr0ng!Password99")
    assert not r["ok"]
    assert "username" in r["error"].lower() or "taken" in r["error"].lower()


def test_register_duplicate_email():
    from src.auth import register_user
    register_user("user_a", "shared@example.com", "MyStr0ng!Password99")
    r = register_user("user_b", "shared@example.com", "MyStr0ng!Password99")
    assert not r["ok"]
    assert "email" in r["error"].lower()


def test_register_username_case_insensitive():
    """Username uniqueness check must be case-insensitive."""
    from src.auth import register_user
    register_user("CaseUser", "case1@example.com", "MyStr0ng!Password99")
    r = register_user("caseuser", "case2@example.com", "MyStr0ng!Password99")
    assert not r["ok"]  # same username, different case


# ---------------------------------------------------------------------------
# 6. Login — valid credentials
# ---------------------------------------------------------------------------
def test_login_valid():
    from src.auth import login_user, register_user
    register_user("loginuser", "login@example.com", "MyStr0ng!Password99")
    r = login_user("loginuser", "MyStr0ng!Password99")
    assert r["ok"], r.get("error")
    assert r["username"] == "loginuser"
    assert "token" in r


def test_login_via_email():
    from src.auth import login_user, register_user
    register_user("emaillogin", "emaillogin@example.com", "MyStr0ng!Password99")
    r = login_user("emaillogin@example.com", "MyStr0ng!Password99")
    assert r["ok"]
    assert r["username"] == "emaillogin"


# ---------------------------------------------------------------------------
# 7. Login — wrong password
# ---------------------------------------------------------------------------
def test_login_wrong_password():
    from src.auth import login_user, register_user
    register_user("wrongpw", "wrongpw@example.com", "MyStr0ng!Password99")
    r = login_user("wrongpw", "WrongPassword!!!")
    assert not r["ok"]
    # Error must not reveal password is wrong specifically
    assert "invalid" in r["error"].lower()


# ---------------------------------------------------------------------------
# 8. Login — unknown user — no account-existence leak
# ---------------------------------------------------------------------------
def test_login_unknown_user():
    from src.auth import login_user
    r = login_user("doesnotexist", "SomePassword123!")
    assert not r["ok"]
    # Should say same generic message as wrong password
    assert "invalid" in r["error"].lower()


# ---------------------------------------------------------------------------
# 9. Session — validate and invalidate
# ---------------------------------------------------------------------------
def test_session_valid():
    from src.auth import login_user, register_user, validate_session
    register_user("sessuser", "sess@example.com", "MyStr0ng!Password99")
    tok = login_user("sessuser", "MyStr0ng!Password99")["token"]
    user = validate_session(tok)
    assert user is not None
    assert user["username"] == "sessuser"


def test_session_invalid_token():
    from src.auth import validate_session
    assert validate_session("completelyfaketoken") is None


def test_session_none():
    from src.auth import validate_session
    assert validate_session(None) is None


def test_logout_invalidates_session():
    from src.auth import login_user, logout, register_user, validate_session
    register_user("logoutuser", "logout@example.com", "MyStr0ng!Password99")
    tok = login_user("logoutuser", "MyStr0ng!Password99")["token"]
    assert validate_session(tok) is not None
    logout(tok)
    assert validate_session(tok) is None


# ---------------------------------------------------------------------------
# 10. Password stored as hash, never plaintext
# ---------------------------------------------------------------------------
def test_password_not_stored_plaintext(fresh_db):
    from src.auth import register_user
    pw = "MyStr0ng!Password99"
    register_user("hashcheck", "hashcheck@example.com", pw)
    conn = sqlite3.connect(str(fresh_db))
    row  = conn.execute("SELECT password_hash FROM users WHERE username='hashcheck'").fetchone()
    conn.close()
    assert row is not None
    assert pw not in row[0]           # plaintext NOT in hash column
    assert row[0].startswith("$argon2id")  # Argon2id hash format


# ---------------------------------------------------------------------------
# 11. Rate limiting — excessive failures cause lockout
# ---------------------------------------------------------------------------
def test_rate_limit_lockout():
    from src.auth import login_user, register_user
    register_user("rluser", "rl@example.com", "MyStr0ng!Password99")
    # Exhaust allowed attempts
    for _ in range(5):
        login_user("rluser", "BadPassword!!!!", ip="testip")
    # Next attempt should be rate-limited
    r = login_user("rluser", "BadPassword!!!!", ip="testip")
    assert not r["ok"]
    assert "too many" in r["error"].lower() or "wait" in r["error"].lower()


# ---------------------------------------------------------------------------
# 12. Session expiry
# ---------------------------------------------------------------------------
def test_session_expiry(monkeypatch, fresh_db):
    """Sessions past their hard expiry are rejected."""
    from src import auth as auth_mod
    from src.auth import login_user, register_user, validate_session
    # Monkey-patch so sessions expire immediately (hard_expires in the past)
    original_create = auth_mod._create_session

    def instant_expire(user_id):
        import secrets
        from datetime import datetime, timedelta, timezone
        token = secrets.token_urlsafe(32)
        now   = datetime.now(timezone.utc)
        past  = (now - timedelta(seconds=1)).isoformat()
        with auth_mod._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token,user_id,created_at,last_seen,expires_at,hard_expires)"
                " VALUES(?,?,?,?,?,?)",
                (token, user_id, now.isoformat(), now.isoformat(), past, past)
            )
        return token

    register_user("expuser", "exp@example.com", "MyStr0ng!Password99")
    result = login_user("expuser", "MyStr0ng!Password99")
    # Replace the token with an already-expired one
    monkeypatch.setattr(auth_mod, "_create_session", instant_expire)
    r2 = login_user("expuser", "MyStr0ng!Password99")
    expired_tok = r2.get("token") if r2.get("ok") else instant_expire(result["user_id"])
    assert validate_session(expired_tok) is None


# ---------------------------------------------------------------------------
# 13. DB persistence across re-initialisation
# ---------------------------------------------------------------------------
def test_persistence(fresh_db):
    from src import auth as auth_mod
    import importlib
    # Register and log out
    auth_mod.register_user("persist1", "persist@example.com", "MyStr0ng!Password99")
    # Re-initialise DB (simulates app restart)
    auth_mod.init_db()
    # Login should still work
    r = auth_mod.login_user("persist1", "MyStr0ng!Password99")
    assert r["ok"]


# ---------------------------------------------------------------------------
# 14. Password strength scoring
# ---------------------------------------------------------------------------
def test_password_strength():
    from src.auth import check_password_strength
    score, _ = check_password_strength("short")
    assert score == 0
    score, _ = check_password_strength("MyStr0ng!Password99XXXXXXXXXX")
    assert score >= 3
