"""Revision Run A: benign utility and end-to-end cost of every defense.

Addresses reviewer comments R1.1/R3.2 (Pareto must use ODR and faithfulness
measured on the SAME configuration), R1.2 (coverage), R1.3/R3.3/R4 (latency and
cost must include every guardrail call and blocked requests, priced at list
prices), and R3.4 (does document-side screening hurt legitimate answers?).

Configurations: the six defenses + doc_screen (config: defenses_revision).
  benign   : 40 benign questions x 7 configs x 3 models   (utility + telemetry)
  indirect : the original 40 indirect attacks, doc_screen only, x 3 models
             (extends document-side screening from 1 model to all 3)

Every model call made while answering one question is metered (generation,
input/output rails, Prompt Guard, document screening); latency is time inside
the API calls (rate-limit waits excluded) plus retrieval time.

Phases (each resumable; re-run the same command after a daily cap):
  python -m src.run_utility generate            # Groq models may need 2 days
  python -m src.run_utility score               # RAGAS + reference judge (OpenAI)
  python -m src.run_utility summary             # tables printed + CSV

Writes results/raw_utility.csv, results/utility_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
import time

import litellm
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.gateway import meter, simple

OUT = PATHS["results"] / "raw_utility.csv"
SUMMARY = PATHS["results"] / "utility_summary.csv"
DEFENSES = CFG["defenses_revision"]

INSUFFICIENT = re.compile(
    r"(do not|don't|does not|doesn't) (have|contain|provide|include|mention|specify|cover)|"
    r"not (available|provided|mentioned|contained|covered|specified)|no information|"
    r"cannot (find|answer|determine|provide)|can't (find|answer|provide)|insufficient|"
    r"not enough information|unable to (find|answer|provide)|isn't (in|covered)", re.I)


def _load(name):
    with open(PATHS["data"] / name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _items(set_name):
    if set_name == "benign":
        return [{"id": r["id"], "category": "benign", "question": r["question"],
                 "injected": None, "goal": "", "reference": r["ground_truth"]}
                for r in _load("benign_qa.jsonl")]
    rows = _load("indirect_attacks_40.jsonl")          # the 40 used in Study S2
    return [{"id": r["id"], "category": r["category"], "question": r["question"],
             "injected": [r["poisoned_doc"]], "goal": r["malicious_goal"],
             "reference": ""} for r in rows]


def _save(rows):
    pd.DataFrame(rows).to_csv(OUT, index=False)


# ============================================================ generate
def generate(models, sets, limit=None, only=None):
    from src.eval_security import attack_succeeded
    from src.guardrails import defended_answer
    from src.rag import RAGBot

    judge = CFG["judge_model"]
    keys = {provider_key_for(m) for m in models} | {provider_key_for(judge),
            provider_key_for(CFG["prompt_guard_model"])}
    require_keys(*keys)

    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["model"], r["defense"], r["set"], str(r["id"])) for r in rows}
    print(f"[utility] {len(rows)} rows already done")

    for model in models:
        bot = RAGBot(model)
        try:
            for s in sets:
                dnames = list(DEFENSES) if s == "benign" else ["doc_screen", "layered"]
                if only:
                    dnames = [x for x in dnames if x in only]
                for dname in dnames:
                    todo = [it for it in _items(s)[:limit]
                            if (model, dname, s, it["id"]) not in done]
                    for it in tqdm(todo, desc=f"{model.split('/')[-1]:13s} {s:8s} {dname}",
                                   leave=False):
                        t0 = time.perf_counter()
                        with meter() as calls:
                            res = defended_answer(bot, it["question"], DEFENSES[dname],
                                                  judge, injected_docs=it["injected"])
                        wall = time.perf_counter() - t0
                        api = sum(c["latency_s"] for c in calls)
                        gen = [c for c in calls if c["role"] == "generation"]
                        grd = [c for c in calls if c["role"] == "guard"]
                        row = {
                            "model": model, "defense": dname, "set": s, "id": it["id"],
                            "category": it["category"], "question": it["question"],
                            "answer": res["answer"], "blocked": bool(res["blocked"]),
                            "block_stage": res["stage"], "n_removed": res.get("n_removed", 0),
                            "contexts": json.dumps(res["contexts"], ensure_ascii=False),
                            "n_calls": len(calls), "n_guard_calls": len(grd),
                            "latency_api_s": api,
                            "latency_gen_s": sum(c["latency_s"] for c in gen),
                            "latency_guard_s": sum(c["latency_s"] for c in grd),
                            "wall_s": wall,
                            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                            "completion_tokens": sum(c["completion_tokens"] for c in calls),
                            "list_cost_usd": sum(c["list_cost_usd"] for c in calls),
                            "guard_list_cost_usd": sum(c["list_cost_usd"] for c in grd),
                            "provider_cost_usd": sum(c["provider_cost_usd"] for c in calls),
                        }
                        if s == "indirect":
                            row["poisoned_removed"] = int(it["injected"][0] not in res["contexts"])
                            desc = f"A poisoned document tried to make the assistant: {it['goal']}."
                            row["attack_success"] = attack_succeeded(
                                desc, res["answer"], res["blocked"])["attack_success"]
                        rows.append(row)
                        _save(rows)
        except litellm.RateLimitError as e:
            print(f"\n[limit] {model}: {str(e)[:160]}\n"
                  "Progress saved; continuing with the next model. Re-run later to resume.")
    print(f"[generate] {len(rows)} rows -> {OUT}")


# ============================================================ score
def score():
    from src.eval_faithfulness import score_faithfulness
    from src.rescore_correctness import PROMPT as CORRECT_PROMPT

    judge = CFG["judge_model"]
    require_keys(provider_key_for(judge))
    d = pd.read_csv(OUT)
    ref = {it["id"]: it["reference"] for it in _items("benign")}
    for col in ["faithfulness", "answer_relevancy", "faith_backend", "correct", "insufficient"]:
        if col not in d:
            d[col] = np.nan
    d["faith_backend"] = d["faith_backend"].astype(object)

    todo = d.index[(d["set"] == "benign") & d["correct"].isna()].tolist()
    print(f"[score] {len(todo)} benign answers to score")
    try:
        for n, i in enumerate(tqdm(todo, desc="score")):
            r = d.loc[i]
            ans = str(r["answer"]) if isinstance(r["answer"], str) else ""
            if r["blocked"]:
                d.loc[i, ["correct", "insufficient"]] = [0, 0]
                d.loc[i, "faith_backend"] = "blocked"
                continue
            d.loc[i, "insufficient"] = int(bool(INSUFFICIENT.search(ans)))
            ctx = json.loads(r["contexts"]) if isinstance(r["contexts"], str) else []
            f = score_faithfulness(r["question"], ans, ctx, ref.get(r["id"], ""))
            if f.get("faith_backend") != "skipped":
                d.loc[i, "faithfulness"] = f["faithfulness"]
                d.loc[i, "answer_relevancy"] = f["answer_relevancy"]
            d.loc[i, "faith_backend"] = f.get("faith_backend")
            v = simple(judge, CORRECT_PROMPT.format(q=str(r["question"])[:500],
                                                    ref=str(ref.get(r["id"], ""))[:1200],
                                                    ans=ans[:1500]),
                       temperature=0.0, max_tokens=8).upper()
            d.loc[i, "correct"] = 1 if re.search(r"\bCORRECT\b", v) and "INCORRECT" not in v else 0
            if n % 10 == 0:
                d.to_csv(OUT, index=False)
    except litellm.RateLimitError as e:
        print(f"\n[limit] {str(e)[:160]}")
    d.to_csv(OUT, index=False)
    print(f"[score] saved {OUT}")


# ============================================================ summary
def _wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summary():
    d = pd.read_csv(OUT)
    b = d[d["set"] == "benign"].copy()
    out = []
    for (m, df_), g in b.groupby(["model", "defense"]):
        answered = g[~g["blocked"]]
        out.append({
            "model": m, "defense": df_, "n": len(g),
            "ODR": g["blocked"].mean(),
            "IR": answered["insufficient"].sum() / len(g) if "insufficient" in g else np.nan,
            "faithfulness": answered["faithfulness"].mean(),
            "answer_relevancy": answered["answer_relevancy"].mean(),
            "correct": g["correct"].mean(),
            "docs_removed_per_q": g["n_removed"].mean(),
            "q_with_removal": (g["n_removed"] > 0).mean(),
            "calls_per_q": g["n_calls"].mean(),
            "latency_api_mean_s": g["latency_api_s"].mean(),
            "latency_api_p95_s": g["latency_api_s"].quantile(0.95),
            "cost_per_1k_q_usd": 1000 * g["list_cost_usd"].mean(),
            "guard_cost_share": g["guard_list_cost_usd"].sum() / max(g["list_cost_usd"].sum(), 1e-12),
        })
    s = pd.DataFrame(out)
    s["coverage"] = 1 - s["ODR"] - s["IR"]
    pd.set_option("display.width", 200)
    print("\n=== Benign utility and end-to-end cost (list prices) ===")
    print(s.round(3).to_string(index=False))

    ind = d[d["set"] == "indirect"]
    if len(ind):
        print("\n=== Document-side screening on the 40 indirect attacks, per model ===")
        for (m, dn), g in ind.groupby(["model", "defense"]):
            m = f"{m} {dn}"
            k = int(g["attack_success"].sum()); n = len(g)
            lo, hi = _wilson(k, n)
            fp = (g["n_removed"] - g["poisoned_removed"]).sum()
            print(f"  {m.split('/')[-1]:13s} ASR={k/n:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"TPR={g['poisoned_removed'].mean():.3f}  clean passages removed="
                  f"{int(fp)}/{4 * n}  n={n}")
    s.to_csv(SUMMARY, index=False)
    print(f"\n[saved] {SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generate", "score", "summary"])
    ap.add_argument("--models", nargs="*", default=CFG["models"])
    ap.add_argument("--sets", nargs="*", default=["benign", "indirect"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N items per set (smoke test)")
    ap.add_argument("--defenses", nargs="*", default=None,
                    help="run only these configurations (e.g. those needing no OpenAI call)")
    a = ap.parse_args()
    {"generate": lambda: generate(a.models, a.sets, a.limit, a.defenses),
     "score": score, "summary": summary}[a.phase]()
