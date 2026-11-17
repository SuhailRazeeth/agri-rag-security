# Guardrail Placement against Prompt Injection in RAG Agricultural Advisory Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22753855.svg)](https://doi.org/10.5281/zenodo.22753855)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible benchmark and harness for evaluating the **security** (direct and
indirect prompt injection), **answer quality**, **latency**, and **cost** of a
retrieval-augmented (RAG) agricultural-economics advisory assistant under eight
guardrail configurations on three language models.

This repository accompanies the paper *"Guardrail Placement against Prompt
Injection in Retrieval-Augmented Agricultural Advisory Systems"* by
M. S. Suhail Razeeth and M. M. Munzeef Ahamed (South Eastern University of Sri Lanka).

| Version | Contents | DOI |
|---|---|---|
| **v2.0.0** | Revised manuscript (Studies S1 to S8, three-model n = 100 replication, human checks) | this release |
| v1.1.0 | Context-side detection and correctness re-scoring | [10.5281/zenodo.22870807](https://doi.org/10.5281/zenodo.22870807) |
| v1.0.0 | Initial submission | [10.5281/zenodo.22753856](https://doi.org/10.5281/zenodo.22753856) |

The concept DOI [10.5281/zenodo.22753855](https://doi.org/10.5281/zenodo.22753855)
always resolves to the latest version.

## Key findings

- **Indirect injection is the dominant threat.** A poisoned retrieved document
  succeeds far more often than a malicious user query (0.225 vs. 0.025).
- **Query screening does not help.** At n = 100 attacks per model, neither an LLM
  input classifier nor Meta Llama Prompt Guard 2 gives a significant reduction on
  any of the three models (pooled attack success 0.27 and 0.30 vs. 0.30 undefended).
- **Screening the retrieved documents does.** The same LLM classifier applied to
  each retrieved passage blocks all 300 attacks (pooled 95% upper bound 0.013),
  with 4.8% false positives.
- **Spotlighting and output rails** reduce attack success on every model
  (p < 10^-4, Holm-adjusted), but spotlighting lowers agreement with reference
  answers (0.700 to 0.608).
- **Instruction-free poisoning** (a false claim with no instruction) evades
  screening more often (0.12). A layered defense reduces this to 0.036.
- **No defense is complete.** A three-round adaptive attacker reaches 0.10 to 0.15
  against every defense.

## Studies and result files

| Study | What it measures | Script | Raw results |
|---|---|---|---|
| S1 | Guardrail modes, benign and direct | `src/run_experiments.py` | `results/raw_runs.csv` |
| S2 | Defense comparison, 40 direct + 40 indirect, 3 models | `src/run_defenses.py` | `results/raw_defenses.csv` |
| S3 | Expanded scale, 100 indirect, 8 configurations, 3 models | `src/run_defenses.py --config-key defenses_revision` | `results/raw_defenses_expanded*.csv` |
| S4 | Document-side detection (TPR/FPR) | `src/run_context_detection.py` | `results/raw_context_detection.csv` |
| S5 | Adaptive attacker, 3 rounds | `src/run_adaptive.py` | `results/raw_adaptive.csv` |
| S6 | Unseen instruction-free attacks | `src/gen_stealth_attacks.py`, `src/run_unseen.py` | `results/raw_unseen.csv` |
| S7 | Utility, latency, and cost | `src/run_utility.py` | `results/raw_utility.csv` |
| S8 | Attack setting and retrieval configuration | `src/run_robustness.py` | `results/raw_robustness.csv` |
| Judges | RAGAS, second judge, human annotation | `src/rescore_faithfulness.py`, `src/second_judge.py`, `src/score_annotation.py`, `src/make_correctness_sheet.py` | `results/faithfulness_ragas.csv`, `inter_judge.csv`, `human_review*.csv`, `human_correctness_annotation.csv`, `reference_validation.csv` |

The S3 runs on the two Groq models were split across several output files
(`_partB`, `_partC`, `_partD`) to fit the free-tier daily token limit. Each file
holds a disjoint set of defense configurations, and `src/analyze_replication.py`
combines them. The `gpt-4o-mini` document-screening and layered rows of S3 come
from the rank-one condition of S8. Prompt Guard 2 threshold scores are in
`results/raw_pg_scores.csv` (`src/run_pg_sweep.py`). The blinded annotation keys
(`results/*annotation_key.csv`) map each annotated item back to its model and
configuration.

## Repository layout

```
src/                 evaluation harness and analysis scripts
tools/               shell wrappers that resume long free-tier runs
data/                hand-authored attack sets (CC BY 4.0) + provenance notes
results/             raw result CSVs and summary tables backing the paper
figures/             figures as they appear in the paper
docs/RESEARCH.md     research design notes
config.yaml          all experiment settings (models, defenses, prices, sample sizes)
requirements.txt     Python dependencies
.env.example         template for API keys (copy to .env; never commit .env)
```

## Setup

Requires Python 3.10+ and API keys for Groq and OpenAI.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
copy .env.example .env            # then paste your keys into .env
```

## Reproduce the tables from the released results (no API calls)

```bash
python -m src.stats                     # Wilson CIs + McNemar tests (S2)
python -m src.analyze                   # summary tables + figures (S1, S2)
python -m src.analyze_revision          # utility, latency, cost, trade-off (S7, Fig. 3)
python -m src.analyze_replication       # three-model n = 100 table (S3)
python -m src.run_unseen --summary      # S6
python -m src.run_robustness --summary  # S8
python -m src.run_utility summary       # S7
```

## Re-run the experiments

```bash
python -m src.data_prep             # download + build the public datasets (HF)
python -m src.build_kb              # embed the knowledge base (ChromaDB, local)
python -m src.smoke_test
python -m src.run_experiments       # S1
python -m src.run_defenses          # S2
python -m src.run_defenses --models groq/openai/gpt-oss-120b --sets indirect \
       --config-key defenses_revision --out raw_defenses_expanded_gpt-oss-120b.csv   # S3
python -m src.run_context_detection # S4
python -m src.run_adaptive          # S5
python -m src.gen_stealth_attacks
python -m src.run_unseen            # S6
python -m src.run_utility generate
python -m src.run_utility score     # S7
python -m src.run_robustness        # S8
```

Every runner is resumable. Re-running the same command skips completed items.
The Groq free tier caps tokens per day per model, so full runs span several days.
`tools/replicate_split.sh` wraps a run in a loop that retries every 30 minutes.

## Data and responsible use

The `data/` attack files exist **only to evaluate and harden defenses**, in line
with standard prompt-injection research. They target a research prototype, name no
real person or organization, and contain no working exploit code. See
[`data/README.md`](data/README.md) for provenance and the CC BY 4.0 terms.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Hand-authored attack data: CC BY 4.0.
