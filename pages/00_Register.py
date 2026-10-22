"""Sign Up page — Industrial Knowledge Intelligence.

Accessible only to unauthenticated users.
"""
from __future__ import annotations

import streamlit as st

from src.auth import (
    check_password_strength,
    init_db,
    normalize_email,
    register_user,
    validate_session,
)
from src.ui import inject_css, inject_auth_bg

inject_css()
inject_auth_bg()
init_db()

# Redirect authenticated users away
if validate_session(st.session_state.get("_auth_token")):
    st.switch_page("app.py")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="auth-page-wrapper">
  <div class="auth-form-card">
    <div class="auth-logo">⚙️</div>
    <h2 class="auth-title">Create Account</h2>
    <p class="auth-subtitle">Join Industrial Knowledge Intelligence</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Registration form
# ---------------------------------------------------------------------------
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    with st.form("register_form", clear_on_submit=False):
        username = st.text_input(
            "Username",
            placeholder="3–30 chars, letters/numbers/underscores",
            autocomplete="username",
            help="3–30 characters. Letters, numbers, and underscores only.",
        )
        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
            autocomplete="email",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 12 characters",
            autocomplete="new-password",
            help="Minimum 12 characters. Mix of uppercase, lowercase, numbers, and symbols recommended.",
        )
        password2 = st.text_input(
            "Confirm password",
            type="password",
            placeholder="Re-enter your password",
            autocomplete="new-password",
        )
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    # --- Live password strength indicator (outside form, reactive) ----------
    if password:
        score, issues = check_password_strength(password)
        labels  = ["Very weak", "Weak", "Fair", "Strong", "Very strong"]
        colours = ["#D96A8A", "#E8834A", "#C99A2E", "#52B788", "#4A90D9"]
        label   = labels[score]
        colour  = colours[score]
        bar_pct = max(10, score * 25)
        st.markdown(
            f'<div style="margin:6px 0 2px;font-size:13px;color:#68758A;">Password strength: '
            f'<b style="color:{colour};">{label}</b></div>'
            f'<div style="height:6px;border-radius:4px;background:rgba(180,210,240,.35);overflow:hidden;">'
            f'<div style="width:{bar_pct}%;height:100%;background:{colour};'
            f'border-radius:4px;transition:width .4s ease;"></div></div>',
            unsafe_allow_html=True,
        )
        if issues:
            for msg in issues:
                st.caption(f"⚠️ {msg}")

    # --- Form submission logic ----------------------------------------------
    if submitted:
        errors = []
        if not username:
            errors.append("Username is required.")
        if not email:
            errors.append("Email address is required.")
        if not password:
            errors.append("Password is required.")
        elif password != password2:
            errors.append("Passwords do not match.")
        else:
            score, pw_issues = check_password_strength(password)
            if score == 0 or pw_issues:
                errors.extend(pw_issues or ["Password is too weak."])

        if not errors and email:
            try:
                normalize_email(email)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Creating your account…"):
                result = register_user(
                    username=username,
                    email=email,
                    password=password,
                )
            if result["ok"]:
                st.success(
                    "✅ Account created successfully! "
                    "Go to **Login** in the sidebar to sign in."
                )
            else:
                st.error(result["error"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#68758A;font-size:14px;">'
        "Already have an account? Use <b>Login</b> in the sidebar.</p>",
        unsafe_allow_html=True,
    )
