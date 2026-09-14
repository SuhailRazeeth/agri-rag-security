"""List the model IDs your API keys can actually use, then suggest a config block.

Providers rename/retire models often, so run this whenever you get a
'model does not exist / no endpoints found' error, and paste the results
into config.yaml -> models.

Run:  python -m src.list_models
"""
from __future__ import annotations

import os

import httpx

from src.config import require_keys


def _get(url: str, key: str | None = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def list_groq() -> list[str]:
    models = _get("https://api.groq.com/openai/v1/models", os.getenv("GROQ_API_KEY"))
    ids = sorted(m["id"] for m in models)
    # keep chat models; drop whisper/tts/guard-only if you like
    return [f"groq/{i}" for i in ids]


def list_openrouter() -> list[str]:
    models = _get("https://openrouter.ai/api/v1/models")  # public, no key needed
    ids = sorted(m["id"] for m in models)
    return [f"openrouter/{i}" for i in ids]


def _pick(candidates: list[str], keywords: list[str]) -> str | None:
    for kw in keywords:
        for c in candidates:
            if kw in c.lower():
                return c
    return candidates[0] if candidates else None


def main():
    require_keys("GROQ_API_KEY")  # OpenRouter list is public

    print("=" * 70)
    print("GROQ — models available to your key:")
    print("=" * 70)
    groq = []
    try:
        groq = list_groq()
        for m in groq:
            print("  ", m)
    except Exception as e:
        print("  [error]", e)

    print("\n" + "=" * 70)
    print("OPENROUTER — free models (id ends with ':free') you can use now:")
    print("=" * 70)
    orouter_all = []
    try:
        orouter_all = list_openrouter()
        free = [m for m in orouter_all if m.endswith(":free")]
        for m in free[:40]:
            print("  ", m)
        if not free:
            print("  (no free models listed; any id below works if your account has credit)")
    except Exception as e:
        print("  [error]", e)

    # ---- suggest a ready-to-paste config block ----
    print("\n" + "=" * 70)
    print("SUGGESTED  config.yaml -> models:  (copy this in, adjust to taste)")
    print("=" * 70)
    suggestions = []
    g_big = _pick(groq, ["70b", "versatile", "gpt-oss", "llama-3"])
    g_small = _pick([m for m in groq if m != g_big], ["8b", "instant", "1b", "3b"])
    orouter_free = [m for m in orouter_all if m.endswith(":free")]
    o1 = _pick(orouter_free, ["llama", "mistral", "qwen", "gemma"])
    for s in [g_big, g_small, o1]:
        if s and s not in suggestions:
            suggestions.append(s)
    suggestions.append("openai/gpt-4o-mini")  # already confirmed working for you

    print("models:")
    for s in suggestions:
        print(f"  - {s}")
    print("\njudge_model: openai/gpt-4o-mini")
    print("\n(Open config.yaml and replace the models: list with the lines above.)")


if __name__ == "__main__":
    main()
