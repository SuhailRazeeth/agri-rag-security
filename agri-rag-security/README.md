# Secure & Faithful RAG Advisory — Agricultural-Economics Prompt-Injection Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Replace `XXXXXXX` in the badge above with your Zenodo DOI after you mint it
> (see **Getting a DOI** below), and set the repository URL in `CITATION.cff`.

A reproducible benchmark and harness for evaluating the **security** (direct and
indirect prompt injection) and **faithfulness** of a retrieval-augmented (RAG)
agricultural-economics advisory assistant, and for comparing six guardrail /
defense strategies on a common footing.

This repository accompanies the paper *"Secure and Faithful Retrieval-Augmented
Advisory Systems: A Domain-Specific Benchmark of Guardrail Strategies against
Direct and Indirect Prompt Injection in Agricultural Economics."*

## Key findings

- **Indirect injection is the dominant threat** — a poisoned retrieved document
  succeeds far more often than a malicious user query.
- **Input-side defenses do not stop it.** Neither an LLM input rail nor Meta
  **Llama Prompt Guard 2** significantly reduces indirect-injection success
  (they screen the benign query, not the poisoned evidence).
- **Generation/output-side defenses work** — spotlighting and output filtering
  cut attack success with *p < 0.0001* (n = 100), at negligible utility cost.
- **No defense is complete** — under a 3-round adaptive attacker the best defense
  degrades 5× (attack success 0.02 → 0.10).
- Judge reliability is validated: faithfulness vs. RAGAS κ = 0.85 (n = 357);
  attack adjudication vs. an independent second judge κ = 0.90 (n = 70);
  human two-pass κ = 0.53 → 0.67 (n = 30).

## Repository layout

```
src/                 evaluation harness (see below)
data/                hand-authored attack sets (CC BY 4.0) + provenance notes
results/             result CSVs and figures backing the paper's tables
docs/RESEARCH.md     research design: questions, metrics, mapping to the paper
config.yaml          all experiment settings (models, defenses, sample sizes)
requirements.txt     Python dependencies
run_all.ipynb        one notebook that runs the whole pipeline
.env.example         template for API keys (copy to .env; never commit .env)
```

## Setup

Requires Python 3.10+ and API keys for Groq and OpenAI (OpenRouter optional).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
copy .env.example .env            # then paste your keys into .env
```

## Reproduce the study

```bash
python -m src.list_models         # confirm valid model IDs for your keys
python -m src.data_prep           # download + build datasets (HF)
python -m src.build_kb            # embed the knowledge base (ChromaDB, local)
python -m src.smoke_test          # sanity check
python -m src.run_experiments     # main 3-model x 3-guardrail study
python -m src.run_defenses        # 6-defense comparison (incl. Prompt Guard 2)
python -m src.run_indirect        # indirect-injection study
python -m src.run_adaptive        # adaptive-attacker study
python -m src.stats               # bootstrap CIs + McNemar tests
python -m src.analyze             # summary tables + figures
```

Judge-validation utilities: `src/rescore_faithfulness.py` (RAGAS),
`src/second_judge.py` (independent second judge), `src/human_validation.py` +
`src/label_cli.py` (human annotation & Cohen's κ).

Free-tier note: the Groq free tier caps tokens **per day per model**, so full
runs may span multiple days; every runner is resumable (re-run the same command).

## Data & responsible use

The `data/` attack files exist **only to evaluate and harden defenses**, in line
with standard prompt-injection research. They target a research prototype, name no
real person or organisation, and contain no working exploit code. See
[`data/README.md`](data/README.md) for provenance and the CC BY 4.0 terms.

## Getting a DOI (Zenodo)

1. Push this repository to a **public** GitHub repo.
2. Sign in at <https://zenodo.org> with your GitHub account.
3. In Zenodo → *Settings → GitHub*, flip the switch **On** for this repository.
4. On GitHub, create a **Release** (e.g. tag `v1.0.0`). Zenodo automatically
   archives it and mints a DOI.
5. Copy the **concept DOI** (the one that always points to the latest version),
   paste it into the badge at the top of this README and into `CITATION.cff`,
   and cite it in the paper's *Data and Code Availability* section.

## Citation

See [`CITATION.cff`](CITATION.cff). After minting the DOI, cite the archived
release.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Hand-authored attack data: CC BY 4.0.
