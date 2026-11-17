"""Revision analysis: utility, cost and the security-utility trade-off (R1.1, R1.3,
R3.2, R3.3, R4), computed for the SAME configurations.

Inputs
  results/raw_utility.csv    Study S7: 3 models x 8 configurations x 40 benign questions
                             (+ doc_screen / layered on the 40 indirect attacks)
  results/raw_defenses.csv   Study S2: indirect ASR of the six core defenses on the
                             same 40 indirect attacks and the same three models
  results/reference_validation.csv   expert rating of the 40 references
Outputs
  results/revision_utility_table.csv   per model x configuration
  results/revision_pooled.csv          pooled over models, tests vs no_defense (Holm)
  results/revision_tradeoff.csv        indirect ASR + utility per model x configuration,
                                       with the Pareto flag
  paper/assets/fig3_pareto.png         redrawn Fig. 3
Run:  python -m src.analyze_revision
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import PATHS
from src.stats import mcnemar_exact

R = PATHS["results"]
ORDER = ["no_defense", "llm_input", "promptguard_input", "spotlight", "llm_output",
         "spotlight_output", "doc_screen", "layered"]
FAMILY = {"no_defense": "none", "llm_input": "query-side", "promptguard_input": "query-side",
          "spotlight": "generation/output-side", "llm_output": "generation/output-side",
          "spotlight_output": "generation/output-side", "doc_screen": "evidence-side",
          "layered": "evidence-side"}
SHORT = {"no_defense": "none", "llm_input": "LLM query", "promptguard_input": "PG2 query",
         "spotlight": "spotlight", "llm_output": "llm_output", "spotlight_output": "spot+out",
         "doc_screen": "doc screen", "layered": "layered"}
MODELS = ["groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b", "openai/gpt-4o-mini"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def per_model(b):
    out = []
    for (m, dn), g in b.groupby(["model", "defense"]):
        ans = g[~g.blocked]
        lo, hi = wilson(int(g.correct.sum()), len(g))
        out.append({"model": m, "defense": dn, "n": len(g), "ODR": g.blocked.mean(),
                    "IR": ans.insufficient.sum() / len(g),
                    "agreement": g.correct.mean(), "agree_lo": lo, "agree_hi": hi,
                    "faithfulness": ans.faithfulness.mean(),
                    "agree_substantive": ans[ans.insufficient == 0].correct.mean(),
                    "q_with_removal": (g.n_removed > 0).mean(),
                    "calls_per_q": g.n_calls.mean(),
                    "latency_mean_s": g.latency_api_s.mean(),
                    "latency_p95_s": g.latency_api_s.quantile(0.95),
                    "cost_per_1k_usd": 1000 * g.list_cost_usd.mean(),
                    "guard_cost_share": g.guard_list_cost_usd.sum() / max(g.list_cost_usd.sum(), 1e-12)})
    s = pd.DataFrame(out)
    s["coverage"] = 1 - s.ODR - s.IR
    return s


def pooled(b, expert_ok, region_no):
    p = b.pivot_table(index=["model", "id"], columns="defense", values="correct")
    f = b[~b.blocked].pivot_table(index=["model", "id"], columns="defense", values="faithfulness")
    rows, pa, pf = [], [], []
    for x in [d for d in ORDER if d in p]:
        g = b[b.defense == x]
        k = int(g.correct.sum()); n = len(g); lo, hi = wilson(k, n)
        r = {"defense": x, "n": n, "agreement": k / n, "lo": lo, "hi": hi,
             "agree_expert_correct": g[g.id.isin(expert_ok)].correct.mean(),
             "agree_region_independent": g[g.id.isin(region_no)].correct.mean(),
             "faithfulness": g[~g.blocked].faithfulness.mean(),
             "ODR": g.blocked.mean(), "IR": g.insufficient.sum() / n,
             "latency_mean_s": g.latency_api_s.mean(),
             "cost_per_1k_usd": 1000 * g.list_cost_usd.mean()}
        if x != "no_defense":
            t = mcnemar_exact(p.no_defense.astype(int), p[x].astype(int))
            j = f[["no_defense", x]].dropna()
            r.update({"b01": t["b01"], "b10": t["b10"], "p_agree": t["p_value"],
                      "p_faith": wilcoxon(j.no_defense, j[x]).pvalue})
            pa.append(t["p_value"]); pf.append(r["p_faith"])
        rows.append(r)
    s = pd.DataFrame(rows)
    m = s.defense != "no_defense"
    s.loc[m, "pH_agree"] = holm(pa); s.loc[m, "pH_faith"] = holm(pf)
    return s


def tradeoff(util, d_ind, s2):
    asr = s2[s2.set == "indirect"].groupby(["model", "defense"]).attack_success.agg(["mean", "size"])
    asr2 = d_ind.groupby(["model", "defense"]).attack_success.agg(["mean", "size"])
    asr = pd.concat([asr, asr2]).rename(columns={"mean": "asr_ind", "size": "n_ind"})
    t = util.merge(asr.reset_index(), on=["model", "defense"], how="left")
    t["pareto"] = False                      # Eq. (pareto): ASR, ODR, agreement
    for m, g in t.groupby("model"):
        g = g.dropna(subset=["asr_ind", "agreement"])
        for i, r in g.iterrows():
            dom = ((g.asr_ind <= r.asr_ind) & (g.ODR <= r.ODR) & (g.agreement >= r.agreement)
                   & ((g.asr_ind < r.asr_ind) | (g.ODR < r.ODR)
                      | (g.agreement > r.agreement))).any()
            t.loc[i, "pareto"] = not dom
    t["SUS"] = (1 - t.asr_ind) * (1 - t.ODR) * t.agreement       # Eq. (sus)
    return t


def plot(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"none": "#52514e", "query-side": "#2a78d6",
           "generation/output-side": "#eb6834", "evidence-side": "#1baf7a"}
    mk = {"no_defense": "o", "llm_input": "s", "promptguard_input": "v", "spotlight": "^",
          "llm_output": ">", "spotlight_output": "p", "doc_screen": "D", "layered": "h"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    ylo, yhi = t.agreement.min() - 0.02, t.agreement.max() + 0.02
    for ax, m in zip(axes, MODELS):
        g = t[t.model == m].set_index("defense")
        g = g.loc[[d for d in ORDER if d in g.index]].dropna(subset=["asr_ind"])
        ax.set_xlim(-0.025, max(0.32, t.asr_ind.max() + 0.03)); ax.set_ylim(ylo, yhi)
        seen = {}
        for dn, r in g.iterrows():
            key = (round(r.asr_ind, 4), round(r.agreement, 4))
            dx = 0.009 * seen.get(key, 0)          # dodge exactly coincident points
            seen[key] = seen.get(key, 0) + 1
            c = col[FAMILY[dn]]
            if r.pareto:          # Pareto-optimal: filled marker
                ax.scatter(r.asr_ind + dx, r.agreement, s=40, marker=mk[dn], color=c,
                           edgecolor="white", linewidth=0.9, zorder=3)
            else:                 # dominated: hollow marker
                ax.scatter(r.asr_ind + dx, r.agreement, s=34, marker=mk[dn], facecolor="white",
                           edgecolor=c, linewidth=1.3, zorder=3)
        ax.set_title(m.split("/")[-1], fontsize=8.5)
        ax.grid(color="#e5e4df", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Agreement with reference\n(higher is better)")
    fig.supxlabel("Indirect attack success on the 40 indirect attacks (lower is better)",
                  fontsize=8, y=0.20)
    hs = [plt.Line2D([], [], marker=mk[d], color=col[FAMILY[d]], ls="", ms=5.5,
                     label=SHORT[d] if d != "spotlight_output" else "spotlight+output")
          for d in ORDER]
    fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 0.06), handletextpad=0.2, columnspacing=0.9)
    fig.text(0.5, 0.01, "Colour = defense family (grey none, blue query-side, orange "
             "generation/output-side, green evidence-side).  Filled = Pareto-optimal, "
             "hollow = dominated.", ha="center", fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(path, dpi=300, facecolor="white")


def _p(x):
    if pd.isna(x):
        return "---"
    return "$<0.001$" if x < 0.001 else f"{x:.3f}"


def latex_tables(util, pool, path):
    """Emit tab:coverage and tab:cost straight from the result files."""
    tt = lambda d: "\\texttt{" + d.replace("_", "\\_") + "}"
    L = []
    L.append("% tab:coverage rows (generated by src/analyze_revision.py)")
    for _, r in pool.iterrows():
        dn = r.defense
        per = [util[(util.model == m) & (util.defense == dn)].agreement.iloc[0] for m in MODELS]
        L.append(f"{tt(dn):28s} & {r.ODR:.3f} & {r.IR:.3f} & {1 - r.ODR - r.IR:.3f} & "
                 + " & ".join(f"{x:.3f}" for x in per)
                 + f" & {r.agreement:.3f}\\,[{r.lo:.3f},{r.hi:.3f}] & {_p(r.get('pH_agree'))}"
                 + f" & {r.faithfulness:.3f} & {_p(r.get('pH_faith'))}\\\\")
    L.append("\n% tab:cost rows")
    for dn in pool.defense:
        g = util[util.defense == dn]
        L.append(f"{tt(dn):28s} & {g.calls_per_q.mean():.0f} & "
                 f"{g.latency_mean_s.min():.2f}--{g.latency_mean_s.max():.2f} & "
                 f"{g.latency_p95_s.max():.2f} & "
                 f"{g.cost_per_1k_usd.min():.3f}--{g.cost_per_1k_usd.max():.3f} & "
                 f"{g.guard_cost_share.min():.2f}--{g.guard_cost_share.max():.2f}\\\\")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    d = pd.read_csv(R / "raw_utility.csv")
    b = d[d.set == "benign"].copy()
    rv = pd.read_csv(R / "reference_validation.csv")
    expert_ok = set(rv[rv.rating.str.strip() == "Correct"].id)
    region_no = set(rv[rv.region_dependent.astype(str).str.strip() == "No"].id)

    util = per_model(b)
    util.to_csv(R / "revision_utility_table.csv", index=False)
    pool = pooled(b, expert_ok, region_no)
    pool.to_csv(R / "revision_pooled.csv", index=False)
    tr = tradeoff(util, d[d.set == "indirect"], pd.read_csv(R / "raw_defenses.csv"))
    tr.to_csv(R / "revision_tradeoff.csv", index=False)
    fig_dir = PATHS["root"] / "paper" / "assets"
    if not fig_dir.exists():
        fig_dir = PATHS["root"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot(tr, fig_dir / "fig3_pareto.png")
    latex_tables(util, pool, R / "revision_tables.tex")

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("=== per model x configuration ===")
    print(util.round(3).to_string(index=False))
    print("\n=== pooled over the three models (tests vs no_defense, Holm over 7) ===")
    print(pool.round(4).to_string(index=False))
    print(f"\nexpert-correct ids: {len(expert_ok)}, region-independent ids: {len(region_no)}")
    print("\n=== trade-off (indirect ASR on the same 40 attacks) ===")
    print(tr[["model", "defense", "asr_ind", "n_ind", "agreement", "faithfulness", "ODR",
              "coverage", "latency_mean_s", "cost_per_1k_usd", "SUS", "pareto"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
