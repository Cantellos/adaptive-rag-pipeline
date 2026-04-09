"""
ui.py — Streamlit UI helpers (styling, auth, chat persistence).
"""

import streamlit as st
import json
import os
import hashlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_loader import cfg


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def apply_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
        min-width: 200px;
        max-width: 700px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        max-width: 70%;
        word-wrap: break-word;
    }

    .stChatInputContainer {
        background-color: white !important;
        border-radius: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        max-width: 800px;
    }

    div[data-baseweb="input"] > div {
        background-color: #f8f9fa !important;
        border: 1.5px solid #dee2e6 !important;
        box-shadow: 0 0 6px rgba(0,123,255,0.25);
        border-radius: 0.5rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease-in-out;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 100%;
            max-width: none;
            border-right: none;
            border-bottom: 1px solid #dee2e6;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat header
# ---------------------------------------------------------------------------

def load_testata():
    title    = cfg.app.get("title", "RAG Assistant")
    icon     = cfg.app.get("icon", "📄")
    welcome  = cfg.app.get("welcome_message", "Welcome! I am your document assistant.")
    disclaimer = cfg.app.get(
        "disclaimer",
        "Always verify the results — check the source documents referenced in each answer.",
    )

    st.markdown(f"# {icon} {title}", unsafe_allow_html=True)
    st.markdown(" ")
    st.write(welcome)
    st.caption(disclaimer)
    st.divider()


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def _get_user_filename(username: str) -> str:
    return os.path.join("chat_history", f"chat_history_{username}.json")


def load_chat_history(username: str) -> list:
    filename = _get_user_filename(username)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save_chat_history(username: str, messages: list) -> None:
    os.makedirs("chat_history", exist_ok=True)
    filename = _get_user_filename(username)
    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(messages, fh, ensure_ascii=False, indent=2)


def reset_chat_history(username: str) -> None:
    filename = _get_user_filename(username)
    if os.path.exists(filename):
        os.remove(filename)


# ---------------------------------------------------------------------------
# User authentication
# ---------------------------------------------------------------------------

USERS_FILE = "users.json"


def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(users, fh, ensure_ascii=False, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str) -> bool:
    """Register a new user. Returns False if the username already exists."""
    users = _load_users()
    if username in users:
        return False
    users[username] = _hash_password(password)
    _save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    """Return True if credentials are valid."""
    users = _load_users()
    return username in users and users[username] == _hash_password(password)
