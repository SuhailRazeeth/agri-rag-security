"""Indirect prompt-injection experiment (RQ1 extension).

Here the USER QUESTION is benign, but a POISONED DOCUMENT is planted in the
retrieved context (simulating a compromised knowledge base / web source). The
input guardrail screens only the question, so it lets these through — only the
OUTPUT guardrail can catch the resulting unsafe answer. This is what separates
the three guardrail modes.

Writes results/raw_indirect.csv (own resume, does not touch raw_runs.csv).

Run:  python -m src.run_indirect
"""
from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_security import attack_succeeded
from src.guardrails import guarded_answer
from src.rag import RAGBot


def _load_indirect():
    path = PATHS["data"] / "indirect_attacks.jsonl"
    if not path.exists():
        raise FileNotFoundError("data/indirect_attacks.jsonl missing.")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _telemetry(res):
    r = res["result"]
    return (0.0, 0.0) if r is None else (r.latency_s, r.cost_usd)


def run():
    judge = CFG["judge_model"]
    needed = {provider_key_for(m) for m in CFG["models"]} | {provider_key_for(judge)}
    require_keys(*(n for n in needed if n))

    attacks = _load_indirect()
    expected = len(attacks)
    print(f"[indirect] {len(attacks)} indirect attacks, "
          f"{len(CFG['models'])} models, {len(CFG['guardrail_modes'])} guardrail modes")

    out_path = PATHS["results"] / "raw_indirect.csv"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        counts = prev.groupby(["model", "guardrail"]).size()
        done = {(m, g) for (m, g), n in counts.items() if n >= expected}
        print(f"[resume] {len(prev)} existing rows; skipping {len(done)} finished combo(s).")

    import litellm
    for model in CFG["models"]:
        bot = None
        for mode in CFG["guardrail_modes"]:
            tag = f"{model} | {mode}"
            if (model, mode) in done:
                print(f"[skip] {tag}  (already complete)")
                continue
            if bot is None:
                bot = RAGBot(model)

            combo_rows = []
            try:
                for item in tqdm(attacks, desc=f"indirect  {tag}", leave=False):
                    # benign question + poisoned document in the retrieved context
                    res = guarded_answer(
                        bot, item["question"], mode, judge,
                        injected_docs=[item["poisoned_doc"]],
                    )
                    lat, cost = _telemetry(res)
                    # Judge success against the INJECTED malicious goal, not the question.
                    desc = (f"A poisoned document tried to make the assistant: "
                            f"{item['malicious_goal']}.")
                    sec = attack_succeeded(desc, res["answer"], res["blocked"])
                    combo_rows.append({
                        "model": model, "guardrail": mode, "set": "indirect",
                        "id": item["id"], "category": item["category"],
                        "question": item["question"], "malicious_goal": item["malicious_goal"],
                        "answer": res["answer"], "blocked": res["blocked"],
                        "block_stage": res["stage"], "latency_s": lat, "cost_usd": cost,
                        **sec,
                    })
            except litellm.RateLimitError as e:
                print(f"\n[limit] {model} hit a rate/daily limit — skipping its remaining "
                      f"modes; re-run later to resume.\n        {str(e)[:150]}")
                break

            rows.extend(combo_rows)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[saved] {tag}  ({len(rows)} rows so far)")

    print(f"\n[done] {len(rows)} rows -> results/raw_indirect.csv")
    print("Next:  python -m src.analyze   (now includes indirect results)")


if __name__ == "__main__":
    run()
