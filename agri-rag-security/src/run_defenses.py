"""Defense comparison experiment (addresses the 'no baseline defenses' gap).

Benchmarks several published/standard defenses on the SAME attacks:
  no_defense, llm_input (our LLM-classifier rail), promptguard_input
  (Meta Llama Prompt Guard 2), spotlight (spotlighting), llm_output,
  spotlight_output.

Runs each defense over the configured attack sets (direct + indirect).
Writes results/raw_defenses.csv (own resume).

Run:  python -m src.run_defenses
"""
from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_security import attack_succeeded
from src.guardrails import defended_answer
from src.rag import RAGBot


def _load(name):
    path = PATHS["data"] / name
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _items_for_set(set_name):
    """Return list of (id, category, query, injected_docs, goal) for a set."""
    if set_name == "attack":
        rows = _load("attacks.jsonl")
        return [(r["id"], r.get("category", "?"), r["attack"], None, r["attack"])
                for r in rows]
    if set_name == "indirect":
        rows = _load("indirect_attacks.jsonl")
        return [(r["id"], r["category"], r["question"], [r["poisoned_doc"]],
                 r["malicious_goal"]) for r in rows]
    raise ValueError(f"unknown set {set_name}")


def _telemetry(res):
    r = res["result"]
    return (0.0, 0.0) if r is None else (r.latency_s, r.cost_usd)


def run(models=None, sets=None, out_name="raw_defenses.csv"):
    """models/sets default to config; out_name lets a variant run (e.g. an
    expanded attack set, or a single model) write to its own file so it does not
    invalidate the resume state of the main study."""
    models = models or CFG["models"]
    sets = sets or CFG.get("defense_sets", ["attack", "indirect"])

    judge = CFG["judge_model"]
    needed = {provider_key_for(m) for m in models} | {provider_key_for(judge)}
    # Prompt Guard runs on Groq:
    needed |= {provider_key_for(CFG.get("prompt_guard_model", "groq/x"))}
    require_keys(*(n for n in needed if n))

    defenses = CFG["defenses"]
    items = {s: _items_for_set(s) for s in sets}
    per_combo = sum(len(items[s]) for s in sets)
    print(f"[defenses] {len(defenses)} defenses x {len(models)} model(s) "
          f"x {per_combo} attacks/combo over sets {sets} -> {out_name}")

    out_path = PATHS["results"] / out_name
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        counts = prev.groupby(["model", "defense"]).size()
        done = {(m, d) for (m, d), n in counts.items() if n >= per_combo}
        print(f"[resume] {len(prev)} rows; skipping {len(done)} finished combo(s).")

    import litellm
    for model in models:
        bot = None
        for dname, dcfg in defenses.items():
            if (model, dname) in done:
                print(f"[skip] {model} | {dname}")
                continue
            if bot is None:
                bot = RAGBot(model)

            combo_rows = []                      # only saved if the combo completes
            try:
                for set_name in sets:
                    for (aid, cat, query, injected, goal) in tqdm(
                        items[set_name], desc=f"{dname:16s} {set_name} {model.split('/')[-1]}",
                        leave=False,
                    ):
                        res = defended_answer(bot, query, dcfg, judge, injected_docs=injected)
                        lat, cost = _telemetry(res)
                        desc = (goal if set_name == "attack"
                                else f"A poisoned document tried to make the assistant: {goal}.")
                        sec = attack_succeeded(desc, res["answer"], res["blocked"])
                        combo_rows.append({
                            "model": model, "defense": dname, "set": set_name,
                            "id": aid, "category": cat, "blocked": res["blocked"],
                            "block_stage": res["stage"], "pg_score": res.get("pg_score"),
                            "latency_s": lat, "cost_usd": cost, **sec,
                        })
            except litellm.RateLimitError as e:
                # Daily/rate cap for this model — discard the partial combo, skip the
                # rest of this model's defenses, and move on. Resume finishes it later.
                print(f"\n[limit] {model} hit a rate/daily limit — skipping its remaining "
                      f"defenses; re-run later to resume.\n        {str(e)[:150]}")
                break

            rows.extend(combo_rows)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[saved] {model} | {dname}  ({len(rows)} rows)")

    print(f"\n[done] {len(rows)} rows -> results/raw_defenses.csv")
    print("Next:  python -m src.stats   and   python -m src.analyze")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="override the model list from config.yaml")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="which attack sets to run: attack and/or indirect")
    ap.add_argument("--out", default="raw_defenses.csv",
                    help="output filename under results/")
    a = ap.parse_args()
    run(a.models, a.sets, a.out)
