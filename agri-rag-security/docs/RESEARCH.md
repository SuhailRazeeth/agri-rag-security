# Research design — how the code maps to your thesis

**Working title:** *Secure and Faithful: A Guardrail-and-Evaluation Framework for
Domain-Specific LLM Advisory Systems in Agricultural Economics.*

## Research questions

- **RQ1 (Security).** How vulnerable is a domain RAG advisory bot to prompt-injection /
  jailbreak / domain-specific misinformation attacks, and how much do guardrails reduce
  the **Attack Success Rate (ASR)**?
- **RQ2 (Faithfulness & over-defense).** Do guardrails hurt **faithfulness** /
  **answer relevancy**, and do they cause **over-defense** (refusing legitimate
  questions — the `benign_refusal_rate`)?
- **RQ3 (Trade-off).** Across models routed through the gateway, what is the Pareto
  frontier of **security × faithfulness × latency × cost**? Which configuration is best
  for a low-resource deployment?

## Independent / dependent variables

| Type | Variable | Where |
|------|----------|-------|
| Independent | Model (4) | `config.yaml → models` |
| Independent | Guardrail mode (off / input / input+output) | `config.yaml → guardrail_modes` |
| Dependent | Faithfulness, Answer relevancy | RAGAS (`eval_faithfulness.py`) |
| Dependent | Attack success rate, Block rate | DeepEval judge (`eval_security.py`) |
| Dependent | Benign refusal rate (over-defense) | `analyze.py` |
| Dependent | Latency, Cost | gateway telemetry (`gateway.py`) |

## How each output maps to a paper section

| Output file | Thesis section |
|-------------|----------------|
| `results/summary.csv` | Results tables (per model × guardrail) |
| `results/pareto.png` | RQ3 trade-off figure |
| `benign_refusal_rate` column | RQ2 over-defense analysis |
| `attack_success_rate` by `category` (group `raw_runs.csv`) | RQ1 threat analysis |
| `faith_backend` / `sec_backend` columns | Methodology / threats-to-validity |

## Datasets (secondary — cite these)

- **Agriculture QA** — KisanVaani/agriculture-qa-english-only (KB + benign eval).
- **Prompt injections** — deepset/prompt-injections (public attacks).
- **Domain attacks** — `data/domain_attacks.jsonl`, 15 hand-authored agri-economics
  attacks (your original contribution; expand toward ~40–50 for the final study).
- Optional upgrades to cite/add later: BIPIA (indirect injection into retrieved docs),
  JailbreakBench (JBB-Behaviors), KrishokChat (citation-grounded agri advisory).

## Methodology notes / threats to validity

- **LLM-as-judge.** RAGAS and DeepEval both use an LLM judge (`judge_model`). Report the
  judge model and, ideally, spot-check a sample against human labels for agreement.
- **Backend transparency.** If RAGAS/DeepEval can't initialise in your environment, the
  code falls back to an equivalent gateway LLM-judge and records this per row
  (`faith_backend`, `sec_backend`). Keep the backend constant across the whole study and
  state it in the paper.
- **Determinism.** `temperature = 0.0` and a fixed `random_seed` make runs reproducible.
- **Cost control.** Defaults are tiny (40 benign + 40 attacks). Scale up once stable.
- **Guardrail realism.** Rails here are LLM-classifier rails. Swapping in Guardrails AI /
  NeMo Guardrails behind the same interface is an easy, publishable ablation.

## Suggested experiment order

1. Baseline (`guardrail=off`) across all models → establishes vulnerability + quality.
2. Add `input` rail → measure ASR drop and any faithfulness/over-defense cost.
3. Add `input_output` rail → measure the additional security gain vs latency/cost.
4. Plot the Pareto frontier; identify the best low-resource configuration.
5. Break ASR down by attack `category` to show *which* threats survive.
