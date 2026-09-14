"""Build the RAG knowledge base: embed the agriculture corpus into ChromaDB.

Embeddings run locally on CPU (sentence-transformers) — no API cost.

Run:  python -m src.build_kb
"""
from __future__ import annotations

import json

import chromadb
from chromadb.utils import embedding_functions

from src.config import CFG, PATHS

COLLECTION = "agri_kb"


def _load_corpus():
    path = PATHS["data"] / "kb_corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run  python -m src.data_prep  first.")
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def _chunk(text: str, max_chars: int) -> list[str]:
    """Simple char-based chunking; fine for short QA documents."""
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def build():
    docs = _load_corpus()
    print(f"[kb] {len(docs)} source documents")

    client = chromadb.PersistentClient(path=str(PATHS["chroma"]))
    # Fresh build each time for reproducibility.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=CFG["embedding_model"]
    )
    col = client.create_collection(name=COLLECTION, embedding_function=embed_fn)

    ids, texts, metas = [], [], []
    for d in docs:
        for j, chunk in enumerate(_chunk(d["text"], CFG["chunk_max_chars"])):
            ids.append(f"{d['id']}-{j}")
            texts.append(chunk)
            metas.append({"source_id": d["id"]})

    print(f"[kb] embedding {len(texts)} chunks (first run downloads the model)…")
    # Add in batches to keep memory reasonable.
    B = 256
    for i in range(0, len(texts), B):
        col.add(
            ids=ids[i : i + B],
            documents=texts[i : i + B],
            metadatas=metas[i : i + B],
        )
        print(f"    {min(i + B, len(texts))}/{len(texts)}")

    print(f"[kb] done. Collection '{COLLECTION}' has {col.count()} chunks.")
    print("Next:  python -m src.smoke_test")


if __name__ == "__main__":
    build()
