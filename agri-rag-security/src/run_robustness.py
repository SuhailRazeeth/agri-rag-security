"""Revision Runs D, E and G: robustness of the main findings to the attack setting
and to the retrieval configuration (reviewer R4), and a layered defense (Run G).

Run D -- attack setting (k = 4, 100 indirect attacks)
  pos1     poisoned document at rank 1 of 5 (the setting of Studies S2-S5)
  pos3     poisoned document in the middle (rank 3 of 5)
  pos5     poisoned document last (rank 5 of 5)
  two_docs two compromised documents: the instruction-bearing document plus an
           instruction-free variant with the same goal (Study S6), ranks 1 and 3
           (79 attacks; only these have an instruction-free variant in S6)
  natural  NO forced placement: the poisoned document competes with the corpus
           on cosine similarity and enters the context only if it ranks in the
           top 4 (it then displaces the lowest-ranked passage)
Run E -- retrieval depth (poisoned document at rank 1)
  k2, k8   2 and 8 legitimately retrieved passages instead of 4
Run G -- unseen instruction-free attacks under the remaining defenses
  unseen   the 138 instruction-free documents of Study S6 at rank 1. For
           gpt-4o-mini only spotlight_output and layered are run here; no_defense
           and doc_screen are the Study S6 results (results/raw_unseen.csv).
For targets other than gpt-4o-mini, pos1 is taken from the n=100 replication of
the core study (results/raw_defenses_expanded_<model>.csv), run with
  python -m src.run_defenses --models <target> --sets indirect
         --config-key defenses_revision --out raw_defenses_expanded_<model>.csv

Defenses: no_defense, spotlight_output, doc_screen, and layered (document
screening + spotlighting + output rail). Judge: primary judge.
Prompt Guard threshold sensitivity is a separate, Groq-only script
(src/run_pg_sweep.py).

Writes results/raw_robustness.csv (resumable per model x condition x defense x attack).
Run:  python -m src.run_robustness
      python -m src.run_robustness --conditions pos3 pos5 --defenses layered
      python -m src.run_robustness --target groq/openai/gpt-oss-120b --conditions pos1 natural
      python -m src.run_robustness --summary
"""
from __future__ import annotations

import argparse
import json

import litellm
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_security import attack_succeeded
from src.guardrails import REFUSAL_MESSAGE, apply_input_rail, apply_output_rail
from src.rag import RAGBot
from src.stats import mcnemar_exact

OUT = PATHS["results"] / "raw_robustness.csv"
TARGET = "openai/gpt-4o-mini"
CONDITIONS = ["pos1", "pos3", "pos5", "two_docs", "natural", "k2", "k8", "temp07", "unseen"]
GEN = {"temp07": {"temperature": 0.7}}   # generation-setting sensitivity (Run E)
DEFENSES = ["no_defense", "spotlight_output", "doc_screen", "layered"]
UNSEEN_DEFENSES = ["spotlight_output", "layered"]       # the rest come from Study S6
_screen_cache: dict[str, bool] = {}


def _load(name):
    with open(PATHS["data"] / name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _retrieve(bot, q, k):
    r = bot.col.query(query_texts=[q], n_results=k, include=["documents", "distances"])
    return r["documents"][0], r["distances"][0]


def _cos_dist(bot, q, doc):
    ef = bot.col._embedding_function
    a, b = (np.asarray(v, dtype=float) for v in ef([q, doc]))
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_context(bot, cond, it, variant):
    """Return (context list, poisoned docs present in it, extra info)."""
    q, doc = it["question"], it["poisoned_doc"]
    if cond in ("k2", "k8"):
        clean, _ = _retrieve(bot, q, int(cond[1:]))
        return [doc] + list(clean), [doc], {}
    clean, dists = _retrieve(bot, q, 4)
    clean = list(clean)
    if cond in ("pos1", "unseen", "temp07"):
        return [doc] + clean, [doc], {}
    if cond == "pos3":
        return clean[:2] + [doc] + clean[2:], [doc], {}
    if cond == "pos5":
        return clean + [doc], [doc], {}
    if cond == "two_docs":
        return [doc] + clean[:1] + [variant] + clean[1:], [doc, variant], {}
    if cond == "natural":
        d = _cos_dist(bot, q, doc)
        ranked = sorted([(x, c) for x, c in zip(dists, clean)] + [(d, doc)])[:4]
        ctx = [c for _, c in ranked]
        rank = next((i + 1 for i, (_, c) in enumerate(ranked) if c == doc), 0)
        return ctx, ([doc] if rank else []), {"retrieved_rank": rank,
                                              "poison_dist": d, "kth_dist": float(dists[-1])}
    raise ValueError(cond)


def _screen(ctx, judge):
    kept = []
    for d in ctx:
        if d not in _screen_cache:
            _screen_cache[d] = apply_input_rail(d, judge)[0]
        if _screen_cache[d]:
            kept.append(d)
    return kept


def defend(bot, q, ctx, dname, judge):
    """Answer under one defense with an explicit context. Returns (answer, blocked, kept)."""
    if dname in ("doc_screen", "layered"):
        kept = _screen(ctx, judge)
        r, _ = bot.answer(q, context_docs=kept, spotlight=(dname == "layered"))
        if dname == "layered" and not apply_output_rail(r.text, judge)[0]:
            return REFUSAL_MESSAGE, True, kept
        return r.text, False, kept
    if dname == "spotlight_output":
        r, _ = bot.answer(q, context_docs=ctx, spotlight=True)
        if not apply_output_rail(r.text, judge)[0]:
            return REFUSAL_MESSAGE, True, ctx
        return r.text, False, ctx
    r, _ = bot.answer(q, context_docs=ctx)
    return r.text, False, ctx


def _rows():
    if not OUT.exists():
        return []
    d = pd.read_csv(OUT)
    if "model" not in d:
        d.insert(0, "model", TARGET)                     # Runs D/E were gpt-4o-mini only
    d["model"] = d["model"].fillna(TARGET)
    return d.to_dict("records")


def run(conditions, target, defenses):
    judge = CFG["judge_model"]
    require_keys(provider_key_for(target), provider_key_for(judge))
    attacks = _load("indirect_attacks.jsonl")
    stealth = _load("indirect_stealth.jsonl")
    variants = {}
    for s in stealth:
        variants.setdefault(s["base_id"], s["poisoned_doc"])        # first variant
    rows = _rows()
    done = {(r["model"], r["condition"], r["defense"], r["id"]) for r in rows}
    bot = RAGBot(target)
    try:
        import src.rag as rag
        for cond in conditions:
            rag.GEN_OVERRIDES = GEN.get(cond, {})
            if cond == "unseen":
                items = stealth
                # gpt-4o-mini: no_defense/doc_screen already measured in Study S6
                dset = ([x for x in defenses if x in UNSEEN_DEFENSES]
                        if target == TARGET else defenses)
            else:
                items = [a for a in attacks if cond != "two_docs" or a["id"] in variants]
                dset = defenses
            for it in tqdm(items, desc=f"{target.split('/')[-1]} {cond}"):
                todo = [x for x in dset if (target, cond, x, it["id"]) not in done]
                if not todo:
                    continue
                ctx, poisoned, info = build_context(bot, cond, it, variants.get(it["id"]))
                desc = f"A poisoned document tried to make the assistant: {it['malicious_goal']}."
                for dname in todo:
                    if cond == "natural" and not poisoned and dname != "no_defense":
                        # poison not retrieved: same clean context, attack cannot act
                        ans, blocked, kept = "", False, ctx
                        succ = 0.0
                    else:
                        ans, blocked, kept = defend(bot, it["question"], ctx, dname, judge)
                        succ = attack_succeeded(desc, ans, blocked)["attack_success"]
                    rows.append({
                        "model": target, "condition": cond, "defense": dname, "id": it["id"],
                        "category": it["category"], "n_context": len(ctx),
                        "n_poisoned_in_context": len(poisoned),
                        "n_poisoned_kept": sum(p in kept for p in poisoned),
                        "blocked": blocked, "attack_success": succ, **info,
                    })
                pd.DataFrame(rows).to_csv(OUT, index=False)
    except litellm.RateLimitError as e:
        print(f"\n[limit] {str(e)[:160]}\nProgress saved; re-run to resume.")
    summary()


def _w(k, n, z=1.959964):
    if n == 0:
        return "n/a"
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{p:.3f} [{max(0, c - h):.3f},{min(1, c + h):.3f}]"


def summary():
    d = pd.DataFrame(_rows())
    uns = PATHS["results"] / "raw_unseen.csv"
    if uns.exists():                       # Study S6 supplies no_defense / doc_screen
        u = pd.read_csv(uns)
        u = u[u.set == "stealth"]
        for x, col in (("no_defense", "asr_none"), ("doc_screen", "asr_doc_screen")):
            d = pd.concat([d, pd.DataFrame({"model": TARGET, "condition": "unseen",
                                            "defense": x, "id": u.id,
                                            "category": u.category,
                                            "attack_success": u[col]})])
    # Rank-1 baseline for other targets = their n=100 replication of the core study
    # (same placement, same items), so the same measurement is not run twice.
    for m in set(d.model) - {TARGET}:
        f = PATHS["results"] / f"raw_defenses_expanded_{m.split('/')[-1]}.csv"
        if f.exists() and not len(d[(d.model == m) & (d.condition == "pos1")]):
            e = pd.read_csv(f)
            e = e[(e.set == "indirect") & e.defense.isin(DEFENSES)]
            d = pd.concat([d, pd.DataFrame({"model": m, "condition": "pos1",
                                            "defense": e.defense, "id": e.id,
                                            "category": e.category,
                                            "attack_success": e.attack_success})])
    for m, dm in d.groupby("model"):
        print(f"\n=== Attack success by condition and defense ({m}) ===")
        print(f"{'condition':10s}" + "".join(f"{x:>24s}" for x in DEFENSES) + "     n")
        for cond in CONDITIONS:
            g = dm[dm.condition == cond]
            if not len(g):
                continue
            cells = []
            for x in DEFENSES:
                s = g[g.defense == x]["attack_success"]
                cells.append(_w(int(s.sum()), len(s)) if len(s) else "-")
            print(f"{cond:10s}" + "".join(f"{c:>24s}" for c in cells)
                  + f"  {g.groupby('defense').size().max()}")
        base = dm[dm.condition == "pos1"].pivot(index="id", columns="defense",
                                                values="attack_success")
        print("\n  Paired McNemar vs pos1 (same defense):")
        for cond in CONDITIONS[1:-1]:
            g = dm[dm.condition == cond].pivot(index="id", columns="defense",
                                               values="attack_success")
            for x in DEFENSES:
                if x in g and x in base:
                    j = base[[x]].join(g[[x]], rsuffix="_c", how="inner").dropna()
                    if len(j):
                        t = mcnemar_exact(j.iloc[:, 0].astype(int), j.iloc[:, 1].astype(int))
                        print(f"    {cond:9s} {x:17s} b01={t['b01']:2d} b10={t['b10']:2d} "
                              f"p={t['p_value']}")
        print("\n  Paired McNemar layered vs other defenses (same condition):")
        for cond in CONDITIONS:
            g = dm[dm.condition == cond].pivot(index="id", columns="defense",
                                               values="attack_success")
            if "layered" not in g:
                continue
            for x in ("no_defense", "spotlight_output", "doc_screen"):
                if x in g:
                    j = g[[x, "layered"]].dropna()
                    t = mcnemar_exact(j[x].astype(int), j["layered"].astype(int))
                    print(f"    {cond:9s} {x:17s} vs layered b01={t['b01']:2d} "
                          f"b10={t['b10']:2d} p={t['p_value']}  n={len(j)}")
        nat = dm[(dm.condition == "natural") & (dm.defense == "no_defense")]
        if len(nat):
            print(f"\n  natural retrieval: poisoned doc retrieved in top-4 for "
                  f"{(nat.retrieved_rank > 0).mean():.3f} of attacks; rank distribution "
                  f"{nat[nat.retrieved_rank > 0].retrieved_rank.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--defenses", nargs="*", default=DEFENSES)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run(a.conditions, a.target, a.defenses)
