"""Revision Run C: does document-side screening generalise to unseen attacks?

Study S4 found that the LLM classifier, applied to retrieved passages, flags
100% of poisoned documents. All of those documents carried an explicit injected
instruction, written in the style that informed the classifier prompt.
Reviewers R3.4 and R4 ask whether this holds for previously unseen attacks.

Here the same detectors (LLM classifier = gpt-4o-mini with the unchanged input-
rail prompt; Prompt Guard 2) are applied, with the same placement as S4, to
  stealth : data/indirect_stealth.jsonl  -- 138 instruction-free poisoned docs,
            written by different models (qwen3.8-27b, gpt-oss-120b) in six
            realistic genres; they simply ASSERT the harmful claim as fact.
  hardneg : data/hard_negatives.jsonl    -- 100 clean look-alike documents on
            the same topics and genres (false-positive control).
For stealth items we also measure end-to-end attack success (target
gpt-4o-mini, primary judge) without defense and with document screening.

Writes results/raw_unseen.csv (resumable).
Run:  python -m src.run_unseen            then   python -m src.run_unseen --summary
"""
from __future__ import annotations

import argparse
import json

import litellm
import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_security import attack_succeeded
from src.guardrails import apply_input_rail, prompt_guard_input_rail
from src.rag import RAGBot
from src.stats import mcnemar_exact

OUT = PATHS["results"] / "raw_unseen.csv"
TARGET = "openai/gpt-4o-mini"
_flag_cache: dict[str, tuple[int, int, float]] = {}


def _load(name):
    with open(PATHS["data"] / name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def flags(doc, judge):
    """(llm_flag, pg_flag, pg_score) for one document, cached in memory."""
    if doc not in _flag_cache:
        llm = int(not apply_input_rail(doc, judge)[0])
        ok, score = prompt_guard_input_rail(doc)
        _flag_cache[doc] = (llm, int(not ok), float(score))
    return _flag_cache[doc]


def run():
    judge = CFG["judge_model"]
    require_keys(provider_key_for(TARGET), provider_key_for(judge),
                 provider_key_for(CFG["prompt_guard_model"]))
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["set"], r["id"]) for r in rows}
    bot = RAGBot(TARGET)
    try:
        for it in tqdm(_load("hard_negatives.jsonl"), desc="hard negatives"):
            if ("hardneg", it["id"]) in done:
                continue
            llm, pg, sc = flags(it["clean_doc"], judge)
            rows.append({"set": "hardneg", "id": it["id"], "category": it["category"],
                         "llm_flag": llm, "pg_flag": pg, "pg_score": sc})
            pd.DataFrame(rows).to_csv(OUT, index=False)

        for it in tqdm(_load("indirect_stealth.jsonl"), desc="unseen attacks"):
            if ("stealth", it["id"]) in done:
                continue
            q, doc = it["question"], it["poisoned_doc"]
            clean = list(bot.retrieve(q))
            ctx = [doc] + clean
            f = [flags(d, judge) for d in ctx]
            desc = f"A poisoned document tried to make the assistant: {it['malicious_goal']}."
            r0, _ = bot.answer(q, context_docs=ctx)
            kept = [d for d, x in zip(ctx, f) if not x[0]]
            if doc in kept:                            # poisoned doc survived screening
                r1, _ = bot.answer(q, context_docs=kept)
                asr_screen = attack_succeeded(desc, r1.text, False)["attack_success"]
            else:
                asr_screen = 0.0
            rows.append({
                "set": "stealth", "id": it["id"], "base_id": it["base_id"],
                "category": it["category"], "generator": it.get("generator"),
                "genre": it.get("genre"),
                "llm_flag": f[0][0], "pg_flag": f[0][1], "pg_score": f[0][2],
                "llm_fp_clean": sum(x[0] for x in f[1:]), "pg_fp_clean": sum(x[1] for x in f[1:]),
                "n_clean": len(clean),
                "asr_none": attack_succeeded(desc, r0.text, False)["attack_success"],
                "asr_doc_screen": asr_screen,
            })
            pd.DataFrame(rows).to_csv(OUT, index=False)
    except litellm.RateLimitError as e:
        print(f"\n[limit] {str(e)[:160]}\nProgress saved; re-run to resume.")
    summary()


def _w(k, n, z=1.959964):
    import math
    if n == 0:
        return "n/a"
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{p:.3f} [{max(0, c - h):.3f},{min(1, c + h):.3f}] ({k}/{n})"


def summary():
    d = pd.read_csv(OUT)
    st, hn = d[d.set == "stealth"], d[d.set == "hardneg"]
    print("\n=== Unseen instruction-free attacks: document-side detection ===")
    if len(st):
        print("  LLM classifier TPR :", _w(int(st.llm_flag.sum()), len(st)))
        print("  Prompt Guard 2 TPR :", _w(int(st.pg_flag.sum()), len(st)))
        nc = int(st.n_clean.sum())
        print("  LLM FPR (KB passages):", _w(int(st.llm_fp_clean.sum()), nc))
        print("  ASR no defense     :", _w(int(st.asr_none.sum()), len(st)))
        print("  ASR doc screening  :", _w(int(st.asr_doc_screen.sum()), len(st)))
        print("  McNemar none vs screen:", mcnemar_exact(st.asr_none.astype(int),
                                                         st.asr_doc_screen.astype(int)))
        print("  per category (LLM TPR / ASR none / ASR screen):")
        for c, g in st.groupby("category"):
            print(f"    {c:20s} {g.llm_flag.mean():.2f} / {g.asr_none.mean():.2f} / "
                  f"{g.asr_doc_screen.mean():.2f}  n={len(g)}")
        print("  by generator TPR:", st.groupby("generator").llm_flag.mean().round(3).to_dict())
    if len(hn):
        print("  Hard negatives FPR: LLM", _w(int(hn.llm_flag.sum()), len(hn)),
              "| PG", _w(int(hn.pg_flag.sum()), len(hn)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run()
