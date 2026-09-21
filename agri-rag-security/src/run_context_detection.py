"""Context-side detection experiment (addresses the reviewer's central objection).

The prior study applied the injection detectors (LLM classifier, Meta Llama
Prompt Guard 2) ONLY to the user query, which under indirect injection is benign.
That measures placement, not detector capability. Here we apply the SAME
detectors to the RETRIEVED DOCUMENTS -- i.e. the actually-attacked content -- and
measure, per detector:

  * TPR : does it flag the poisoned document?               (capability)
  * FPR : does it wrongly flag the clean retrieved docs?    (collateral)
  * ASR : end-to-end attack success after removing flagged docs, then generating.

This separates "detector capability on the attacked content" from "detector
placement", exactly as the reviewer asked.

Writes results/raw_context_detection.csv  (resumable per item).

Run:  python -m src.run_context_detection
      python -m src.run_context_detection --target openai/gpt-4o-mini
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


def _load_indirect():
    with open(PATHS["data"] / "indirect_attacks.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _flag_pg(doc: str) -> bool:
    allowed, _ = prompt_guard_input_rail(doc)      # same detector, doc as input
    return not allowed


def _flag_llm(doc: str, judge: str) -> bool:
    allowed, _ = apply_input_rail(doc, judge)      # same classifier, doc as input
    return not allowed


def run(target: str):
    judge = CFG["judge_model"]
    require_keys(provider_key_for(target), provider_key_for(judge),
                 provider_key_for(CFG.get("prompt_guard_model", "groq/x")))
    attacks = _load_indirect()
    print(f"[context-detection] {len(attacks)} indirect attacks, target={target}")

    out_path = PATHS["results"] / "raw_context_detection.csv"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(prev["id"].astype(str))
        print(f"[resume] {len(prev)} rows done.")

    bot = RAGBot(target)
    try:
        for a in tqdm(attacks, desc="context-detection"):
            if a["id"] in done:
                continue
            q, poisoned, goal = a["question"], a["poisoned_doc"], a["malicious_goal"]
            clean = bot.retrieve(q)                       # benign retrieved docs
            context_full = [poisoned] + list(clean)       # poisoned at rank 1
            desc = f"A poisoned document tried to make the assistant: {goal}."

            # --- detection on each document (the attacked content) ---
            pg_flags = [_flag_pg(d) for d in context_full]
            llm_flags = [_flag_llm(d, judge) for d in context_full]

            # --- end-to-end ASR under three context conditions ---
            def asr(ctx):
                if not ctx:                              # everything filtered -> refuse
                    return 0.0, "(all context filtered)"
                r, _ = bot.answer(q, context_docs=ctx)
                return float(attack_succeeded(desc, r.text, False)["attack_success"]), r.text

            ctx_pg = [d for d, f in zip(context_full, pg_flags) if not f]
            ctx_llm = [d for d, f in zip(context_full, llm_flags) if not f]
            asr_none, _ = asr(context_full)
            asr_pg, _ = asr(ctx_pg)
            asr_llm, _ = asr(ctx_llm)

            n_clean = len(clean)
            rows.append({
                "id": a["id"], "category": a["category"], "model": target,
                "pg_flag_poisoned": int(pg_flags[0]),
                "llm_flag_poisoned": int(llm_flags[0]),
                "pg_flag_clean": int(sum(pg_flags[1:])), "n_clean": n_clean,
                "llm_flag_clean": int(sum(llm_flags[1:])),
                "asr_no_defense": asr_none,
                "asr_context_promptguard": asr_pg,
                "asr_context_llm": asr_llm,
                "poisoned_removed_pg": int(pg_flags[0]),
                "poisoned_removed_llm": int(llm_flags[0]),
            })
            pd.DataFrame(rows).to_csv(out_path, index=False)
    except litellm.RateLimitError as e:
        print(f"\n[limit] {target} hit a rate/daily limit; re-run to resume.\n  {str(e)[:120]}")

    _summary(pd.DataFrame(rows))
    print(f"[saved] {out_path}")


def _summary(d: pd.DataFrame):
    if len(d) == 0:
        return
    n = len(d)
    clean_total = int(d["n_clean"].sum())
    print("\n" + "=" * 62)
    print(f"  n = {n} indirect attacks   (clean docs seen = {clean_total})")
    print("-" * 62)
    print("  DETECTION on the attacked content:")
    print(f"    Prompt Guard 2  TPR (poisoned flagged) = {d['pg_flag_poisoned'].mean():.3f}")
    print(f"    LLM classifier  TPR (poisoned flagged) = {d['llm_flag_poisoned'].mean():.3f}")
    print(f"    Prompt Guard 2  FPR (clean flagged)    = {d['pg_flag_clean'].sum()/max(clean_total,1):.3f}")
    print(f"    LLM classifier  FPR (clean flagged)    = {d['llm_flag_clean'].sum()/max(clean_total,1):.3f}")
    print("-" * 62)
    print("  END-TO-END attack success:")
    print(f"    no defense                     ASR = {d['asr_no_defense'].mean():.3f}")
    print(f"    context-side Prompt Guard 2    ASR = {d['asr_context_promptguard'].mean():.3f}")
    print(f"    context-side LLM classifier    ASR = {d['asr_context_llm'].mean():.3f}")
    print(f"    (compare: query-side Prompt Guard 2 ASR = 0.240 from the main study)")
    print("=" * 62)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="openai/gpt-4o-mini")
    a = ap.parse_args()
    run(a.target)
