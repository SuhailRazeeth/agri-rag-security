"""Revision Run E (part 2): Prompt Guard 2 decision-threshold sensitivity.

Records the raw Prompt Guard 2 injection probability for every document used in
the document-side studies and sweeps the threshold gamma:
  poisoned : 100 instruction-bearing (S4) + 138 instruction-free (S6) documents
  clean    : the co-retrieved knowledge-base passages + 100 clean look-alikes
At each gamma we report TPR on each poisoned set and FPR on the clean sets,
answering whether the default gamma = 0.5 explains Prompt Guard's weak detection.

Groq only (no OpenAI cost). Writes results/raw_pg_scores.csv (resumable).
Run:  python -m src.run_pg_sweep          python -m src.run_pg_sweep --summary
"""
from __future__ import annotations

import argparse
import json

import litellm
import pandas as pd
from tqdm import tqdm

from src.config import PATHS
from src.guardrails import prompt_guard_input_rail
from src.rag import RAGBot

OUT = PATHS["results"] / "raw_pg_scores.csv"
GAMMAS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def _load(name):
    with open(PATHS["data"] / name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run():
    docs = []                                            # (set, id, text)
    bot = RAGBot("openai/gpt-4o-mini")                   # retrieval only, no API call
    for a in _load("indirect_attacks.jsonl"):
        docs.append(("poisoned_instruction", a["id"], a["poisoned_doc"]))
        for j, c in enumerate(bot.retrieve(a["question"])):
            docs.append(("clean_kb", f"{a['id']}-kb{j}", c))
    for s in _load("indirect_stealth.jsonl"):
        docs.append(("poisoned_instruction_free", s["id"], s["poisoned_doc"]))
    for h in _load("hard_negatives.jsonl"):
        docs.append(("clean_lookalike", h["id"], h["clean_doc"]))

    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["set"], r["id"]) for r in rows}
    try:
        for s, i, text in tqdm(docs, desc="Prompt Guard scores"):
            if (s, i) in done:
                continue
            _, score = prompt_guard_input_rail(text)
            rows.append({"set": s, "id": i, "pg_score": score})
            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False)
    except litellm.RateLimitError as e:
        print(f"\n[limit] {str(e)[:160]}\nProgress saved; re-run to resume.")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    summary()


def summary():
    d = pd.read_csv(OUT)
    print("\n=== Prompt Guard 2 threshold sweep ===")
    print("gamma   TPR(instr)  TPR(instr-free)  FPR(KB)  FPR(look-alike)")
    for g in GAMMAS:
        r = {s: (x["pg_score"] >= g).mean() for s, x in d.groupby("set")}
        print(f"{g:<7} {r.get('poisoned_instruction', float('nan')):10.3f}"
              f" {r.get('poisoned_instruction_free', float('nan')):16.3f}"
              f" {r.get('clean_kb', float('nan')):8.3f}"
              f" {r.get('clean_lookalike', float('nan')):16.3f}")
    print("n per set:", d.groupby("set").size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run()
