"""Score the returned independent annotation (Munzeef_annotation_task.xlsx).

Reports
  Task 1: Cohen's kappa of annotator 2 vs the primary LLM judge, vs annotator 1
          (both passes), and vs the second LLM judge where available.
  Task 2: share of dataset reference answers rated Correct / Partly / Incorrect,
          and reference agreement recomputed on the expert-validated subset.

Run:  python -m src.score_annotation  [path\to\returned.xlsx]
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import PATHS
from src.stats import cohen_kappa

DEFAULT = PATHS["root"] / "paper" / "Final Paper" / "revision" / "Munzeef_annotation_task.xlsx"


def main(path):
    key = pd.read_csv(PATHS["results"] / "annotation_key.csv")
    hr = pd.read_csv(PATHS["results"] / "human_review.csv")

    # ---------------- Task 1
    s = pd.read_excel(path, sheet_name="Security_Labels")
    s.columns = ["item", "goal", "question", "answer", "label", "comment"]
    s = s.merge(key, on="item")
    s["a2"] = s["label"].astype(str).str.upper().map({"COMPLIED": 1, "RESISTED": 0})
    s = s.join(hr[["judge_label", "human_label", "human_label_pass1"]], on="src_row")
    s = s.dropna(subset=["a2"])
    print(f"=== Task 1: {len(s)} security items labelled by annotator 2 ===")
    for col, name in [("judge_label", "primary LLM judge"),
                      ("human_label", "annotator 1, pass 2"),
                      ("human_label_pass1", "annotator 1, pass 1")]:
        g = s.dropna(subset=[col])
        k = cohen_kappa(g["a2"].astype(int), g[col].astype(int))
        agree = (g["a2"].astype(int) == g[col].astype(int)).mean()
        print(f"  vs {name:22s} kappa={k:.3f}  agreement={agree:.3f}  n={len(g)}")
    print(pd.crosstab(s["judge_label"], s["a2"], rownames=["judge"], colnames=["annotator 2"]))

    # ---------------- Task 2
    r = pd.read_excel(path, sheet_name="Reference_Check")
    r.columns = ["item", "question", "reference", "rating", "regional", "comment"]
    r = r.merge(key[["item", "id"]], on="item").dropna(subset=["rating"])
    print(f"\n=== Task 2: {len(r)} reference answers rated ===")
    print(r["rating"].value_counts().to_string())
    print("region/season dependent:", r["regional"].value_counts().to_dict())

    corr = PATHS["results"] / "correctness.csv"
    if corr.exists():
        c = pd.read_csv(corr).merge(r[["id", "rating"]], on="id")
        valid = c[c["rating"] == "Correct"]
        print(f"\nReference agreement on all references       : {c['correct'].mean():.3f} (n={len(c)})")
        print(f"Reference agreement on expert-'Correct' only : {valid['correct'].mean():.3f} (n={len(valid)})")
        print(valid.groupby(["model", "guardrail"])["correct"].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
