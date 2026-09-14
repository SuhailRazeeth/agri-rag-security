"""Re-score the already-generated benign answers with RAGAS — no re-generation.

Your first run scored faithfulness with the LLM-judge fallback. This re-scores the
SAME answers using the real RAGAS library so you can cite RAGAS in the paper.
Retrieval is deterministic and local (free); only the OpenAI judge is called, so
there are no Groq rate-limit issues and it costs only a few cents.

Reads  results/raw_runs.csv
Writes results/faithfulness_ragas.csv  (per-row RAGAS scores)

Run:  python -m src.rescore_faithfulness
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.config import CFG, PATHS, provider_key_for, require_keys
from src.eval_faithfulness import _RAGAS, _init_ragas
from src.rag import RAGBot


def main():
    require_keys(provider_key_for(CFG["judge_model"]))  # RAGAS judge = OpenAI

    raw = PATHS["results"] / "raw_runs.csv"
    if not raw.exists():
        raise FileNotFoundError("Run  python -m src.run_experiments  first.")
    df = pd.read_csv(raw)
    benign = df[(df["set"] == "benign") & (~df["refused"].astype(bool))].copy()
    print(f"[rescore] {len(benign)} answered benign rows to score with RAGAS")

    _init_ragas()
    if not _RAGAS["ready"]:
        raise RuntimeError(
            "RAGAS backend did not initialise. Ensure judge_model is an openai/* model "
            "and dependencies are installed (pip install -r requirements.txt)."
        )

    # One bot just for (local, free) retrieval — the LLM is never called here.
    bot = RAGBot(CFG["models"][0])

    # Ground-truth lookup by benign id (from data/benign_qa.jsonl via raw file).
    import json
    gt = {}
    with open(PATHS["data"] / "benign_qa.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                gt[d["id"]] = d["ground_truth"]

    out = []
    for _, r in tqdm(benign.iterrows(), total=len(benign), desc="RAGAS"):
        contexts = bot.retrieve(r["question"])   # same deterministic retrieval
        try:
            s = _RAGAS["score"](r["question"], str(r["answer"]), contexts,
                                gt.get(r["id"], ""))
        except Exception:
            s = {"faithfulness": None, "answer_relevancy": None}
        out.append({
            "model": r["model"], "guardrail": r["guardrail"], "id": r["id"],
            "ragas_faithfulness": s["faithfulness"],
            "ragas_answer_relevancy": s["answer_relevancy"],
        })

    res = pd.DataFrame(out)
    res.to_csv(PATHS["results"] / "faithfulness_ragas.csv", index=False)
    print("\n[saved] results/faithfulness_ragas.csv")

    print("\n=== Mean RAGAS scores by model x guardrail ===")
    summary = res.groupby(["model", "guardrail"])[
        ["ragas_faithfulness", "ragas_answer_relevancy"]
    ].mean().round(3)
    print(summary.to_string())
    print("\nCompare these to the LLM-judge columns in results/summary.csv "
          "for your methodology section.")


if __name__ == "__main__":
    main()
