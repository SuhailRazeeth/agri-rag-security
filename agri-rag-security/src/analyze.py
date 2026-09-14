"""Aggregate the experiment CSVs into the paper's tables + figures.

Reads:
  results/raw_runs.csv      (benign + direct attacks)
  results/raw_indirect.csv  (indirect injection attacks, if present)

Writes:
  results/summary.csv            - main table (per model x guardrail)
  results/category_breakdown.csv - attack success/block by attack category (RQ1)
  results/pareto.png             - security x faithfulness trade-off (RQ3)
  results/guardrail_effect.png   - direct vs indirect attack success by guardrail

Run:  python -m src.analyze
"""
from __future__ import annotations

import json

import pandas as pd
import matplotlib.pyplot as plt

from src.config import PATHS


# ---------------------------------------------------------------------------
def _load_attack_categories() -> dict:
    """id -> {source, category} for the DIRECT attack set."""
    path = PATHS["data"] / "attacks.jsonl"
    out = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    out[d["id"]] = {"source": d.get("source", "?"),
                                    "category": d.get("category", "?")}
    return out


def summarize(df: pd.DataFrame, indirect: pd.DataFrame | None) -> pd.DataFrame:
    out = []
    ind_by_combo = (
        indirect.groupby(["model", "guardrail"]) if indirect is not None else None
    )
    for (model, mode), g in df.groupby(["model", "guardrail"]):
        benign = g[g["set"] == "benign"]
        attack = g[g["set"] == "attack"]
        answered = benign[~benign["refused"]]

        row = {
            "model": model,
            "guardrail": mode,
            "faithfulness": round(answered["faithfulness"].astype(float).mean(), 3)
                if len(answered) else float("nan"),
            "answer_relevancy": round(answered["answer_relevancy"].astype(float).mean(), 3)
                if len(answered) else float("nan"),
            "benign_refusal_rate": round(benign["refused"].mean(), 3),
            "direct_attack_success": round(attack["attack_success"].astype(float).mean(), 3),
            "direct_block_rate": round(attack["blocked"].mean(), 3),
        }
        # indirect metrics (if that experiment was run)
        if ind_by_combo is not None and (model, mode) in ind_by_combo.groups:
            ig = ind_by_combo.get_group((model, mode))
            row["indirect_attack_success"] = round(ig["attack_success"].astype(float).mean(), 3)
            row["indirect_block_rate"] = round(ig["blocked"].mean(), 3)
        else:
            row["indirect_attack_success"] = float("nan")
            row["indirect_block_rate"] = float("nan")

        row["avg_latency_s"] = round(g["latency_s"].astype(float).mean(), 3)
        row["total_cost_usd"] = round(g["cost_usd"].astype(float).sum(), 5)
        out.append(row)
    return pd.DataFrame(out).sort_values(["model", "guardrail"]).reset_index(drop=True)


def category_breakdown(df: pd.DataFrame, indirect: pd.DataFrame | None) -> pd.DataFrame:
    """Attack success & block rate per attack category x guardrail (RQ1)."""
    cats = _load_attack_categories()
    direct = df[df["set"] == "attack"].copy()
    direct["source"] = direct["id"].map(lambda i: cats.get(i, {}).get("source", "?"))
    direct["category"] = direct["id"].map(lambda i: cats.get(i, {}).get("category", "?"))
    direct["attack_type"] = "direct"

    frames = [direct[["attack_type", "source", "category", "guardrail",
                      "attack_success", "blocked"]]]
    if indirect is not None:
        ind = indirect.copy()
        ind["attack_type"] = "indirect"
        ind["source"] = "domain"
        frames.append(ind[["attack_type", "source", "category", "guardrail",
                           "attack_success", "blocked"]])

    allrows = pd.concat(frames, ignore_index=True)
    g = allrows.groupby(["attack_type", "category", "guardrail"]).agg(
        n=("attack_success", "size"),
        attack_success_rate=("attack_success", lambda s: round(s.astype(float).mean(), 3)),
        block_rate=("blocked", lambda s: round(s.astype(float).mean(), 3)),
    ).reset_index()
    return g


# ---------------------------------------------------------------------------
def pareto_plot(summary: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"off": "o", "input": "s", "input_output": "^"}
    for mode, g in summary.groupby("guardrail"):
        ax.scatter(g["faithfulness"], 1 - g["direct_attack_success"],
                   label=f"guardrail={mode}", marker=markers.get(mode, "o"), s=90)
        for _, r in g.iterrows():
            ax.annotate(r["model"].split("/")[-1][:18],
                        (r["faithfulness"], 1 - r["direct_attack_success"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Faithfulness  (higher = better)")
    ax.set_ylabel("Security = 1 − Attack Success Rate  (higher = better)")
    ax.set_title("Security × Faithfulness trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


def guardrail_effect_plot(summary: pd.DataFrame, path):
    """Mean attack success (direct vs indirect) by guardrail mode — the key figure
    showing that the OUTPUT rail matters for indirect injection."""
    order = ["off", "input", "input_output"]
    g = (summary.groupby("guardrail")[["direct_attack_success", "indirect_attack_success"]]
         .mean().reindex(order))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(order))
    w = 0.35
    ax.bar([i - w / 2 for i in x], g["direct_attack_success"], w, label="direct attacks")
    ax.bar([i + w / 2 for i in x], g["indirect_attack_success"], w, label="indirect injection")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_xlabel("Guardrail mode")
    ax.set_ylabel("Mean attack success rate (lower = better)")
    ax.set_title("Guardrail effectiveness: direct vs indirect attacks")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


# ---------------------------------------------------------------------------
def main():
    raw = PATHS["results"] / "raw_runs.csv"
    if not raw.exists():
        raise FileNotFoundError("Run  python -m src.run_experiments  first.")
    df = pd.read_csv(raw)

    ind_path = PATHS["results"] / "raw_indirect.csv"
    indirect = pd.read_csv(ind_path) if ind_path.exists() else None
    if indirect is not None:
        print(f"[analyze] including {len(indirect)} indirect-injection rows.")

    summary = summarize(df, indirect)
    summary.to_csv(PATHS["results"] / "summary.csv", index=False)
    print("[summary] wrote results/summary.csv\n")
    print(summary.to_string(index=False))

    cats = category_breakdown(df, indirect)
    cats.to_csv(PATHS["results"] / "category_breakdown.csv", index=False)
    print("\n[category] wrote results/category_breakdown.csv\n")
    print(cats.to_string(index=False))

    pareto_plot(summary, PATHS["results"] / "pareto.png")
    if indirect is not None:
        guardrail_effect_plot(summary, PATHS["results"] / "guardrail_effect.png")

    print("\nDone. Tables + figures are in results/.")


if __name__ == "__main__":
    main()
