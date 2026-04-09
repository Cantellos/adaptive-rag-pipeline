"""
config_loader.py
================
Loads and exposes the top-level config.yaml as a simple Python object.

Usage
-----
    from config_loader import cfg

    print(cfg.domain.name)          # "Your Software / System"
    print(cfg.app.title)            # "Document RAG Assistant"
    print(cfg.retrieval.default_top_k)  # 15

The module resolves the config file by walking up from this file's
directory until it finds config.yaml — so it works regardless of the
current working directory when a script is launched.
"""

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    """A dict subclass that allows attribute-style access (cfg.key.subkey)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"Config key '{name}' not found") from None
        return _AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        value = super().get(key, default)
        return _AttrDict(value) if isinstance(value, dict) else value


def _find_config_file() -> Path:
    """Walk up the directory tree from this file to find config.yaml."""
    current = Path(__file__).resolve().parent
    for _ in range(6):  # max 6 levels up
        candidate = current / "config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "config.yaml not found. Make sure it exists in the project root."
    )


def _load(path: Path) -> _AttrDict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _AttrDict(raw or {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

cfg: _AttrDict = _load(_find_config_file())


def reload() -> None:
    """Re-read config.yaml from disk (useful during development)."""
    global cfg
    cfg = _load(_find_config_file())


def fmt(template: str) -> str:
    """
    Substitute config tokens in *template* and return the result.

    Supported tokens:
        {domain_name}          → cfg.domain.name
        {domain_description}   → cfg.domain.description
        {document_type}        → cfg.domain.document_type
        {out_of_scope}         → cfg.domain.out_of_scope_description
    """
    return template.format(
        domain_name=cfg.domain.name,
        domain_description=cfg.domain.description,
        document_type=cfg.domain.document_type,
        out_of_scope=cfg.domain.out_of_scope_description,
    )
