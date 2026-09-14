"""Loads .env (API keys) and config.yaml (experiment settings).

Import `CFG` and `PATHS` anywhere: `from src.config import CFG`.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = parent of this file's folder (…/AI-Economics)
ROOT = Path(__file__).resolve().parent.parent

# 1) Load API keys from .env into the environment (LiteLLM reads them from there).
load_dotenv(ROOT / ".env")

# 2) Load experiment configuration.
with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CFG: dict = yaml.safe_load(f)


def _abs(p: str) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


PATHS = {
    "root": ROOT,
    "data": _abs(CFG["paths"]["data_dir"]),
    "results": _abs(CFG["paths"]["results_dir"]),
    "chroma": _abs(CFG["paths"]["chroma_dir"]),
}

# Make sure the output folders exist.
PATHS["data"].mkdir(parents=True, exist_ok=True)
PATHS["results"].mkdir(parents=True, exist_ok=True)


def require_keys(*names: str) -> None:
    """Raise a clear error if a needed API key is missing from .env."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            "Missing API key(s) in your .env file: "
            + ", ".join(missing)
            + "\nOpen  .env  and paste the key(s), then re-run."
        )


def provider_key_for(model: str) -> str:
    """Return the env-var name of the key a given model needs (for checks)."""
    if model.startswith("groq/"):
        return "GROQ_API_KEY"
    if model.startswith("openai/"):
        return "OPENAI_API_KEY"
    if model.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    return ""
