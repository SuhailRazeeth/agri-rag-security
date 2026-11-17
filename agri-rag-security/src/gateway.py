"""LLM Gateway — a thin wrapper over LiteLLM so every model (Groq / OpenAI /
OpenRouter) is called the same way, and we log latency + cost for RQ3.

LiteLLM automatically reads GROQ_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
from the environment (loaded from .env by src.config).
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass

import litellm

from src.config import CFG

# Quieten LiteLLM's verbose logging; keep our output clean.
litellm.drop_params = True          # ignore params a provider doesn't support
litellm.suppress_debug_info = True

# --- Rate limiting -----------------------------------------------------------
# Free tiers cap BOTH requests-per-minute (RPM) and tokens-per-minute (TPM).
#   * RPM  -> minimum seconds between calls (config: rate_limits).
#   * TPM  -> rolling 60s token budget      (config: tpm_limits, per model or provider).
# We stay under both so we never trigger a 429; a retry loop is the safety net.
_DEFAULT_INTERVALS = {"groq": 2.2, "openai": 0.0, "openrouter": 1.5}
_last_call: dict[str, float] = {}
_tok_history: dict[str, list[tuple[float, int]]] = {}   # provider -> [(ts, tokens)]


def _provider(model: str) -> str:
    return model.split("/", 1)[0]


def _tpm_for(model: str) -> float:
    limits = CFG.get("tpm_limits") or {}
    if model in limits:
        return float(limits[model])
    return float(limits.get(_provider(model), 0.0))


def _estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // 4 + int(max_tokens)   # ~4 chars/token, plus room for the reply


def _throttle(model: str, est_tokens: int) -> None:
    prov = _provider(model)
    intervals = {**_DEFAULT_INTERVALS, **CFG.get("rate_limits", {})}
    min_gap = float(intervals.get(prov, 0.0))
    tpm = _tpm_for(model)

    while True:
        now = time.time()
        # (a) requests-per-minute spacing
        rpm_wait = (min_gap - (now - _last_call.get(prov, 0.0))) if min_gap > 0 else 0.0
        # (b) tokens-per-minute rolling budget
        tpm_wait = 0.0
        if tpm > 0:
            hist = _tok_history.setdefault(prov, [])
            cutoff = now - 60
            while hist and hist[0][0] < cutoff:
                hist.pop(0)
            used = sum(t for _, t in hist)
            if used + est_tokens > tpm and hist:
                tpm_wait = 60 - (now - hist[0][0]) + 0.2   # wait for oldest to expire
        wait = max(rpm_wait, tpm_wait)
        if wait <= 0:
            break
        time.sleep(min(wait, 65))
    _last_call[prov] = time.time()


def _record_tokens(model: str, tokens: int) -> None:
    _tok_history.setdefault(_provider(model), []).append((time.time(), tokens))


def _parse_retry_seconds(err: str, default: float = 8.0) -> float:
    m = re.search(r"try again in ([\d.]+)s", err)
    return (float(m.group(1)) + 1.0) if m else default


# --- End-to-end call accounting ------------------------------------------------
# Inside `with meter() as calls:` every chat() call (generation, guardrails,
# Prompt Guard) is appended to `calls`, tagged with the role set by call_role().
_METER: list | None = None
_ROLE = ["generation"]


@contextmanager
def meter():
    global _METER
    prev, _METER = _METER, []
    try:
        yield _METER
    finally:
        _METER = prev


@contextmanager
def call_role(name: str):
    prev, _ROLE[0] = _ROLE[0], name
    try:
        yield
    finally:
        _ROLE[0] = prev


def list_cost(model: str, p_tok: int, c_tok: int) -> float:
    """Cost at published paid-tier list prices (config: list_prices, USD per 1M
    tokens), so free-tier calls are still priced realistically."""
    price = (CFG.get("list_prices") or {}).get(model)
    if not price:
        return 0.0
    return (p_tok * float(price["input"]) + c_tok * float(price["output"])) / 1e6


@dataclass
class LLMResult:
    text: str
    latency_s: float
    cost_usd: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def chat(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> LLMResult:
    """Single chat completion through the gateway. Returns text + telemetry."""
    temperature = CFG["temperature"] if temperature is None else temperature
    max_tokens = CFG["max_tokens"] if max_tokens is None else max_tokens

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:     # e.g. {"type": "json_object"} (OpenAI)
        kwargs["response_format"] = response_format
    # gpt-oss / reasoning models spend tokens "thinking" before answering.
    # Keep reasoning light so the token budget goes to the actual answer.
    if "gpt-oss" in model or "reasoning" in model:
        kwargs["reasoning_effort"] = "low"

    est = _estimate_tokens(messages, max_tokens)
    max_attempts = int(CFG.get("num_retries", 8)) + 1

    # latency = time inside the successful API call only. Client-side rate-limit
    # waits and retry back-off are excluded: they are artefacts of free-tier
    # quotas, not of the defense being measured.
    latency = 0.0
    resp = None
    for attempt in range(max_attempts):
        _throttle(model, est)           # respect RPM + TPM before every attempt
        try:
            t_call = time.perf_counter()
            resp = litellm.completion(**kwargs)
            latency = time.perf_counter() - t_call
            break
        except litellm.RateLimitError as e:      # our own backoff (more reliable here)
            emsg = str(e).lower()
            wait = _parse_retry_seconds(str(e), default=8.0)
            # A daily cap (TPD) or a long wait can't be retried away — fail fast so
            # the caller can skip this model and resume tomorrow.
            # An exhausted balance/quota is reported as 429 too, but never recovers.
            no_credit = ("insufficient_quota" in emsg or "credit_balance" in emsg
                         or "no credits" in emsg)
            if (no_credit or "per day" in emsg or "tpd" in emsg or wait > 40
                    or attempt == max_attempts - 1):
                raise
            time.sleep(wait)

    msg = resp.choices[0].message
    text = msg.content or ""
    # Some reasoning models put the reply in reasoning_content if content is empty.
    if not text:
        text = getattr(msg, "reasoning_content", "") or ""

    # Cost + token accounting (best-effort; some providers omit fields).
    try:
        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:
        cost = 0.0
    usage = getattr(resp, "usage", None)
    p_tok = getattr(usage, "prompt_tokens", 0) or 0
    c_tok = getattr(usage, "completion_tokens", 0) or 0
    _record_tokens(model, (p_tok + c_tok) or est)   # feed the TPM budget tracker

    if _METER is not None:                          # end-to-end accounting
        _METER.append({"role": _ROLE[0], "model": model, "latency_s": latency,
                       "prompt_tokens": p_tok, "completion_tokens": c_tok,
                       "provider_cost_usd": cost,
                       "list_cost_usd": list_cost(model, p_tok, c_tok)})

    return LLMResult(
        text=text.strip(),
        latency_s=latency,
        cost_usd=cost,
        model=model,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
    )


def simple(model: str, prompt: str, **kw) -> str:
    """Convenience: send one user prompt, get back just the text."""
    return chat(model, [{"role": "user", "content": prompt}], **kw).text
