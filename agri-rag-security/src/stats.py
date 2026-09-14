"""Statistical rigor for the paper (addresses the 'no statistics' gap).

- Attack Success Rate with 95% bootstrap confidence intervals.
- McNemar's exact test comparing each defense to 'no_defense' (paired by attack id).
- Cohen's kappa helper for LLM-judge vs human agreement (gap #1).

Reads results/raw_defenses.csv (falls back to raw_indirect.csv).
Writes results/stats_asr_ci.csv and prints significance tests.

Run:  python -m src.stats
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import PATHS

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05):
    """Mean of a 0/1 vector with a percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(x.mean()), 3), round(float(lo), 3), round(float(hi), 3))


def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test on paired binary outcomes a,b (1=attack success).
    Returns discordant counts and a two-sided exact p-value."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    b01 = int(((a == 0) & (b == 1)).sum())   # a defended, b failed
    b10 = int(((a == 1) & (b == 0)).sum())   # a failed, b defended
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}
    k = min(b01, b10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b01": b01, "b10": b10, "p_value": round(min(1.0, p), 4)}


def cohen_kappa(y1, y2) -> float:
    """Cohen's kappa for two raters on binary labels (judge vs human)."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return float("nan")
    po = float((y1 == y2).mean())
    p1 = y1.mean() * y2.mean()
    p0 = (1 - y1.mean()) * (1 - y2.mean())
    pe = p1 + p0
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


# ---------------------------------------------------------------------------
def main():
    path = PATHS["results"] / "raw_defenses.csv"
    if not path.exists():
        alt = PATHS["results"] / "raw_indirect.csv"
        if not alt.exists():
            raise FileNotFoundError("Run  python -m src.run_defenses  first.")
        df = pd.read_csv(alt)
        df["defense"] = df["guardrail"] if "guardrail" in df else "n/a"
        df["set"] = "indirect"
    else:
        df = pd.read_csv(path)

    df["attack_success"] = df["attack_success"].astype(float)

    # ---- ASR with 95% CI per (set, model, defense) ----
    ci_rows = []
    for (set_name, model, defense), g in df.groupby(["set", "model", "defense"]):
        mean, lo, hi = bootstrap_ci(g["attack_success"].values)
        ci_rows.append({"set": set_name, "model": model, "defense": defense,
                        "n": len(g), "ASR": mean, "ci_low": lo, "ci_high": hi,
                        "block_rate": round(g["blocked"].astype(float).mean(), 3)})
    ci = pd.DataFrame(ci_rows).sort_values(["set", "model", "defense"])
    ci.to_csv(PATHS["results"] / "stats_asr_ci.csv", index=False)
    print("=== Attack Success Rate with 95% bootstrap CI ===")
    print(ci.to_string(index=False))

    # ---- McNemar: each defense vs no_defense, paired by attack id ----
    if "no_defense" in df["defense"].unique():
        print("\n=== McNemar exact test vs 'no_defense' (paired by attack id) ===")
        print(f"{'set':10s}{'model':28s}{'defense':18s}{'p_value':>9s}  signif")
        for set_name in df["set"].unique():
            for model in df["model"].unique():
                base = df[(df.set == set_name) & (df.model == model) &
                          (df.defense == "no_defense")].set_index("id")["attack_success"]
                for defense in df["defense"].unique():
                    if defense == "no_defense":
                        continue
                    cur = df[(df.set == set_name) & (df.model == model) &
                             (df.defense == defense)].set_index("id")["attack_success"]
                    ids = base.index.intersection(cur.index)
                    if len(ids) == 0:
                        continue
                    r = mcnemar_exact(base.loc[ids].values, cur.loc[ids].values)
                    sig = "***" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
                    print(f"{set_name:10s}{model.split('/')[-1][:26]:28s}"
                          f"{defense:18s}{r['p_value']:>9.4f}  {sig}")

    print("\n[saved] results/stats_asr_ci.csv")
    print("Report the ASR ± CI in tables and the McNemar p-values in the text.")


if __name__ == "__main__":
    main()
