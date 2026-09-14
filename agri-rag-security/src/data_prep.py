"""Download and normalize the public datasets.

Outputs (in data/):
  kb_corpus.jsonl   - agriculture documents used to build the RAG knowledge base
  benign_qa.jsonl   - {question, ground_truth} eval set (faithfulness axis)
  attacks.jsonl     - {id, source, category, attack} eval set (security axis)

Run:  python -m src.data_prep
"""
from __future__ import annotations

import json
import random

from datasets import load_dataset

from src.config import CFG, PATHS

random.seed(CFG["random_seed"])
DATA = PATHS["data"]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5} rows -> {path.name}")


def _first_present(row: dict, *keys):
    """Return the first non-empty value among candidate column names."""
    for k in keys:
        if k in row and row[k]:
            return str(row[k]).strip()
    return ""


def prepare_agriculture():
    """Agriculture QA -> KB corpus + benign eval set."""
    name = CFG["datasets"]["agri_qa"]
    print(f"[agri] loading {name} …")
    ds = load_dataset(name, split="train")

    pairs = []
    for row in ds:
        q = _first_present(row, "question", "Question", "query", "instruction")
        a = _first_present(row, "answers", "answer", "Answer", "response", "output")
        if q and a and len(a) > 15:
            pairs.append({"question": q, "ground_truth": a})

    random.shuffle(pairs)

    # KB corpus = the answers (chunked later in build_kb). Use a broad slice so
    # retrieval has real material even for the held-out benign questions.
    kb_slice = pairs[: max(1500, CFG["benign_sample_size"] * 20)]
    kb = [
        {"id": f"agri-{i}", "text": f"Q: {p['question']}\nA: {p['ground_truth']}"}
        for i, p in enumerate(kb_slice)
    ]
    _write_jsonl(DATA / "kb_corpus.jsonl", kb)

    # Benign eval set = a held-out sample (kept out of the KB slice to be fair).
    held_out = pairs[len(kb_slice):] or pairs
    benign = random.sample(held_out, min(CFG["benign_sample_size"], len(held_out)))
    for i, r in enumerate(benign):
        r["id"] = f"benign-{i}"
    _write_jsonl(DATA / "benign_qa.jsonl", benign)


def prepare_attacks():
    """Public prompt-injections + your domain attacks -> unified attack set."""
    attacks = []

    # 1) Public prompt-injection dataset (label 1 = injection).
    name = CFG["datasets"]["prompt_injections"]
    print(f"[attacks] loading {name} …")
    try:
        ds = load_dataset(name, split="train")
        pub = []
        for row in ds:
            text = _first_present(row, "text", "prompt")
            label = row.get("label", 1)
            if text and int(label) == 1:
                pub.append(text)
        random.shuffle(pub)
        keep = pub[: max(1, CFG["attack_sample_size"] - 15)]
        for i, t in enumerate(keep):
            attacks.append(
                {"id": f"pub-{i}", "source": "deepset", "category": "prompt_injection", "attack": t}
            )
    except Exception as e:
        print(f"  [warn] could not load {name}: {e}. Using domain attacks only.")

    # 2) Your hand-written domain-specific attacks (always included).
    dom_path = PATHS["root"] / CFG["datasets"]["domain_attacks"]
    print(f"[attacks] loading domain attacks from {dom_path.name} …")
    with open(dom_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            attacks.append(
                {
                    "id": d["id"],
                    "source": "domain",
                    "category": d.get("category", "domain"),
                    "attack": d["attack"],
                }
            )

    # Trim to the configured sample size but keep ALL domain attacks.
    domain = [a for a in attacks if a["source"] == "domain"]
    public = [a for a in attacks if a["source"] != "domain"]
    n_public = max(0, CFG["attack_sample_size"] - len(domain))
    final = domain + public[:n_public]
    random.shuffle(final)
    _write_jsonl(DATA / "attacks.jsonl", final)


if __name__ == "__main__":
    print("=== Preparing datasets ===")
    prepare_agriculture()
    prepare_attacks()
    print("Done. Next:  python -m src.build_kb")
