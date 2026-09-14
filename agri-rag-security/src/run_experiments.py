"""The main experiment grid.

For every (model x guardrail_mode):
  - Benign set  -> faithfulness, answer_relevancy, refusal rate, latency, cost
  - Attack set  -> attack-success-rate, block rate, latency, cost

Writes results/raw_runs.csv (one row per question run).

Run:  python -m src.run_experiments
"""
from __future__ import annotations

import json

import litellm
import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_faithfulness import score_faithfulness
from src.eval_security import attack_succeeded
from src.guardrails import REFUSAL_MESSAGE, guarded_answer
from src.rag import RAGBot


def _load_jsonl(name):
    path = PATHS["data"] / name
    if not path.exists():
        raise FileNotFoundError(f"{name} missing. Run  python -m src.data_prep  first.")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _telemetry(res):
    r = res["result"]
    if r is None:
        return 0.0, 0.0
    return r.latency_s, r.cost_usd


def run():
    judge = CFG["judge_model"]

    # Verify keys up front.
    needed = {provider_key_for(m) for m in CFG["models"]} | {provider_key_for(judge)}
    require_keys(*(n for n in needed if n))

    benign = _load_jsonl("benign_qa.jsonl")
    attacks = _load_jsonl("attacks.jsonl")
    expected = len(benign) + len(attacks)
    print(f"[run] {len(benign)} benign, {len(attacks)} attacks, "
          f"{len(CFG['models'])} models, {len(CFG['guardrail_modes'])} guardrail modes")

    # ----- RESUME: keep rows already saved, skip finished (model, guardrail) combos -----
    out_path = PATHS["results"] / "raw_runs.csv"
    rows = []
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        counts = prev.groupby(["model", "guardrail"]).size()
        done = {(m, g) for (m, g), n in counts.items() if n >= expected}
        print(f"[resume] found {len(prev)} existing rows; "
              f"{len(done)} finished combo(s) will be skipped.")

    for model in CFG["models"]:
        bot = None  # built lazily so skipped models don't load the embedder
        for mode in CFG["guardrail_modes"]:
            tag = f"{model} | {mode}"
            if (model, mode) in done:
                print(f"[skip] {tag}  (already complete)")
                continue
            if bot is None:
                bot = RAGBot(model)

            combo_rows = []
            try:
                # ---------- benign (faithfulness) ----------
                for item in tqdm(benign, desc=f"benign  {tag}", leave=False):
                    res = guarded_answer(bot, item["question"], mode, judge)
                    lat, cost = _telemetry(res)
                    refused = res["blocked"] or res["answer"].strip() == REFUSAL_MESSAGE
                    fa = (
                        {"faithfulness": None, "answer_relevancy": None, "faith_backend": "refused"}
                        if refused
                        else score_faithfulness(
                            item["question"], res["answer"], res["contexts"], item["ground_truth"]
                        )
                    )
                    combo_rows.append({
                        "model": model, "guardrail": mode, "set": "benign",
                        "id": item["id"], "question": item["question"],
                        "answer": res["answer"], "blocked": res["blocked"],
                        "refused": refused, "latency_s": lat, "cost_usd": cost,
                        "attack_success": None, **fa,
                    })

                # ---------- attacks (security) ----------
                for item in tqdm(attacks, desc=f"attack  {tag}", leave=False):
                    res = guarded_answer(bot, item["attack"], mode, judge)
                    lat, cost = _telemetry(res)
                    sec = attack_succeeded(item["attack"], res["answer"], res["blocked"])
                    combo_rows.append({
                        "model": model, "guardrail": mode, "set": "attack",
                        "id": item["id"], "question": item["attack"],
                        "answer": res["answer"], "blocked": res["blocked"],
                        "refused": res["blocked"], "latency_s": lat, "cost_usd": cost,
                        "faithfulness": None, "answer_relevancy": None,
                        "faith_backend": None, **sec,
                    })
            except litellm.RateLimitError as e:
                print(f"\n[limit] {model} hit a rate/daily limit — skipping its remaining "
                      f"modes; re-run later to resume.\n        {str(e)[:150]}")
                break

            # Save incrementally so a crash never loses completed work.
            rows.extend(combo_rows)
            pd.DataFrame(rows).to_csv(PATHS["results"] / "raw_runs.csv", index=False)
            print(f"[saved] {tag}  ({len(rows)} rows so far)")

    print(f"\n[done] {len(rows)} rows -> results/raw_runs.csv")
    print("Next:  python -m src.analyze")


if __name__ == "__main__":
    run()
