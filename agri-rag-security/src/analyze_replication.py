"""Three-model replication of the 100-attack study (reviewers R2.4, R4).

gpt-4o-mini : Study S3 (six core configurations, results/raw_defenses_expanded.csv)
              + document screening and layered at rank one from Study S8
              (results/raw_robustness.csv, condition pos1)
gpt-oss-120b, gpt-oss-20b : all eight configurations in one run each
              (results/raw_defenses_expanded_<model>.csv, revision replication)

For every model: indirect ASR with 95% Wilson CI, block rate, and the exact McNemar
test of each defense against no_defense, Holm-adjusted over the seven comparisons.
Also pooled over the three models (n = 300 paired items per configuration).
Writes results/replication_n100.csv and LaTeX rows to results/replication_rows.tex.
Run:  python -m src.analyze_replication
"""
from __future__ import annotations

import pandas as pd

from src.analyze_revision import ORDER, holm, wilson
from src.config import PATHS
from src.stats import mcnemar_exact

R = PATHS["results"]
MODELS = {"openai/gpt-4o-mini": "gpt-4o-mini", "groq/openai/gpt-oss-120b": "gpt-oss-120b",
          "groq/openai/gpt-oss-20b": "gpt-oss-20b"}


def load():
    s3 = pd.read_csv(R / "raw_defenses_expanded.csv")
    s3 = s3[s3.set == "indirect"][["model", "defense", "id", "attack_success", "blocked"]]
    rb = pd.read_csv(R / "raw_robustness.csv")
    if "model" not in rb:
        rb["model"] = "openai/gpt-4o-mini"
    rb = rb[(rb.condition == "pos1") & rb.defense.isin(["doc_screen", "layered"])]
    parts = [s3, rb[["model", "defense", "id", "attack_success", "blocked"]]]
    for m, tag in MODELS.items():
        if tag == "gpt-4o-mini":
            continue
        # main file plus any *_partB.csv written when the run was split across keys
        for f in sorted(R.glob(f"raw_defenses_expanded_{tag}*.csv")):
            d = pd.read_csv(f)
            parts.append(d[d.set == "indirect"][["model", "defense", "id", "attack_success", "blocked"]])
    d = pd.concat(parts, ignore_index=True)
    d["blocked"] = d["blocked"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return d


def table(d):
    out = []
    groups = [(MODELS[m], g) for m, g in d.groupby("model")] + [("pooled", d)]
    for name, g in groups:
        idx = ["model", "id"] if name == "pooled" else ["id"]
        p = g.pivot_table(index=idx, columns="defense", values="attack_success", aggfunc="first")
        rows, ps = [], []
        for x in [c for c in ORDER if c in p]:
            s = g[g.defense == x]
            k, n = int(s.attack_success.sum()), len(s)
            lo, hi = wilson(k, n)
            r = {"model": name, "defense": x, "n": n, "asr": k / n, "lo": lo, "hi": hi,
                 "block_rate": s.blocked.mean()}
            if x != "no_defense" and "no_defense" in p:
                j = p[["no_defense", x]].dropna()
                t = mcnemar_exact(j.no_defense.astype(int), j[x].astype(int))
                r.update({"b01": t["b01"], "b10": t["b10"], "p": t["p_value"], "n_pairs": len(j)})
                ps.append(t["p_value"])
            rows.append(r)
        t = pd.DataFrame(rows)
        m = t.defense != "no_defense"
        if m.any():
            t.loc[m, "pH"] = holm(ps)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def main():
    d = load()
    t = table(d)
    t.to_csv(R / "replication_n100.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print("\nrows per model x defense:")
    print(d.groupby(["model", "defense"]).size().unstack(fill_value=0))
    # LaTeX rows: ASR [CI] per model, pooled ASR and pooled p_H
    fmt = lambda r: f"{r.asr:.3f}\\,[{r.lo:.3f},{r.hi:.3f}]"
    lines = []
    for x in ORDER:
        cells = []
        for name in ["gpt-4o-mini", "gpt-oss-120b", "gpt-oss-20b", "pooled"]:
            r = t[(t.model == name) & (t.defense == x)]
            cells.append(fmt(r.iloc[0]) if len(r) and r.iloc[0].n >= 100 else "--")
        r = t[(t.model == "pooled") & (t.defense == x)]
        ph = "---" if x == "no_defense" or not len(r) else (
            "$<10^{-4}$" if r.iloc[0].pH < 1e-4 else f"{r.iloc[0].pH:.3f}")
        lines.append("\\texttt{" + x.replace("_", "\\_") + "} & " + " & ".join(cells)
                     + f" & {ph}\\\\")
    (R / "replication_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
