"""
app.py — Streamlit entry point for the RAG chatbot.

Run with:
    streamlit run main/llm/app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import cfg

import ui
from llm import gpt_request

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.app.get("title", "RAG Assistant"),
    page_icon=cfg.app.get("icon", "📄"),
    layout="wide",
)

# ── Apply CSS styling ────────────────────────────────────────────────────────
ui.apply_style()

# ── Sidebar — user authentication ───────────────────────────────────────────
with st.sidebar:
    # Logo (optional — set app.logo_url in config.yaml to show it)
    logo_url = cfg.app.get("logo_url", "").strip()
    if logo_url:
        st.image(logo_url, width=200)

    st.markdown(f"### {cfg.app.get('icon', '📄')} {cfg.app.get('title', 'RAG Assistant')}")
    st.markdown(
        cfg.app.get("welcome_message", "Sign in to start chatting with the document assistant.")
    )
    st.markdown("---")
    st.markdown("### 🔐 User Access")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # NOT authenticated → Login / Register form
    if not st.session_state.authenticated:
        action = st.radio("Select action:", ["Sign in", "Register"], horizontal=True)

        with st.form(key="auth_form", clear_on_submit=False):
            username_input = st.text_input("👤 Username", key="username_input")
            password_input = st.text_input("🔑 Password", type="password", key="password_input")
            submit = st.form_submit_button("Login" if action == "Sign in" else "Create account")

        if action == "Sign in" and submit:
            if ui.authenticate_user(username_input, password_input):
                st.session_state.authenticated = True
                st.session_state.username = username_input
                st.session_state.messages = ui.load_chat_history(username_input)
                if not st.session_state.messages:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "How can I help you? 👇"}
                    ]
                st.success(f"Welcome, **{username_input}** 👋")
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        elif action == "Register" and submit:
            if not username_input or not password_input:
                st.warning("Please enter both username and password.")
            elif ui.register_user(username_input, password_input):
                st.success("Registration complete! You can now sign in.")
            else:
                st.error("This username already exists.")
        st.stop()

    # Authenticated → user options
    else:
        username = st.session_state.username
        st.success(f"✅ Signed in as **{username}**")

        if st.button("🔄 Reset chat"):
            ui.reset_chat_history(username)
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat reset. How can I help you? 👇"}
            ]
            ui.save_chat_history(username, st.session_state.messages)
            st.success("💬 Chat reset successfully!")
            st.rerun()

        if st.button("🚪 Sign out"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

# ── Initialise or restore chat history ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = ui.load_chat_history(username)
    if not st.session_state.messages:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    f"You are a document assistant for {cfg.domain.get('name', 'the knowledge base')}. "
                    "Answer clearly and based only on the documents or information already provided."
                ),
            },
            {"role": "assistant", "content": "How can I help you? 👇"},
        ]

# ── Main chat header ─────────────────────────────────────────────────────────
ui.load_testata()

# ── Render existing chat history ─────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📄"):
        st.markdown(message["content"])

# ── User input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Generating answer..."):
            full_response = ""
            placeholder = st.empty()

            for token in gpt_request(st.session_state.messages):
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            ui.save_chat_history(username, st.session_state.messages)
