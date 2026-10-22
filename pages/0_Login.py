"""Login page — Industrial Knowledge Intelligence.

This is the entry point for unauthenticated users.
No protected data or functions are accessible here.
"""
from __future__ import annotations

import streamlit as st

from src.auth import init_db, login_user
from src.ui import inject_css, inject_auth_bg

inject_css()
inject_auth_bg()
init_db()

# ---------------------------------------------------------------------------
# If already authenticated, redirect to dashboard immediately
# ---------------------------------------------------------------------------
from src.auth import validate_session
if validate_session(st.session_state.get("_auth_token")):
    st.switch_page("app.py")

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.markdown("""
<div class="auth-page-wrapper">
  <div class="auth-form-card">
    <div class="auth-logo">⚙️</div>
    <h2 class="auth-title">Industrial Knowledge Intelligence</h2>
    <p class="auth-subtitle">Sign in to your account</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.form("login_form", clear_on_submit=False):
            identifier = st.text_input(
                "Username or email",
                placeholder="Enter your username or email",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

        if submitted:
            if not identifier or not password:
                st.error("Please enter both your username/email and password.")
            else:
                with st.spinner("Authenticating…"):
                    result = login_user(
                        identifier=identifier,
                        password=password,
                        ip=st.context.headers.get("X-Forwarded-For", "local") if hasattr(st, "context") else "local",
                    )
                if result["ok"]:
                    st.session_state["_auth_token"]    = result["token"]
                    st.session_state["_auth_user"]     = result["user_id"]
                    st.session_state["_auth_username"] = result["username"]
                    st.success(f"Welcome back, **{result['username']}**! Redirecting…")
                    st.rerun()
                else:
                    st.error(result["error"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align:center;color:#68758A;font-size:14px;">'
            "Don't have an account? Use <b>Sign Up</b> in the sidebar.</p>",
            unsafe_allow_html=True,
        )
