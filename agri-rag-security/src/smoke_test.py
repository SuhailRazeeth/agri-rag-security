"""Quick end-to-end sanity check before the full run.

Confirms: keys load, gateway reaches each model, the KB retrieves, and one
guarded answer works.

Run:  python -m src.smoke_test
"""
from __future__ import annotations

from src.config import CFG, provider_key_for, require_keys
from src.gateway import simple
from src.guardrails import guarded_answer
from src.rag import RAGBot


def main():
    print("=== Smoke test ===")

    # 1) Keys present for every configured model?
    needed = {provider_key_for(m) for m in CFG["models"]} - {""}
    needed.add(provider_key_for(CFG["judge_model"]))
    require_keys(*(n for n in needed if n))
    print(f"[ok] keys present for: {', '.join(sorted(needed))}")

    # 2) Each model reachable?
    for m in CFG["models"]:
        try:
            out = simple(m, "Reply with the single word: OK", max_tokens=64)
            flag = "ok" if out.strip() else "EMPTY - check model"
            print(f"[{flag}] {m}  -> {out!r}")
        except Exception as e:
            print(f"[FAIL] {m}: {type(e).__name__}: {e}")

    # 3) RAG + guardrail on the first model (needs the KB to exist).
    model = CFG["models"][0]
    print(f"\n[rag] testing retrieval + guarded answer on {model} …")
    try:
        bot = RAGBot(model)
    except Exception as e:
        if "agri_kb" in str(e) or "does not exist" in str(e):
            print("  [skip] knowledge base not built yet.")
            print("  Build it first, in order:")
            print("      python -m src.data_prep")
            print("      python -m src.build_kb")
            print("  Then re-run this smoke test.")
            print("\n[ok] Models work. Just build the KB next.")
            return
        raise

    q = "What is a good practice for managing soil fertility for rice?"
    res = guarded_answer(bot, q, mode="input", judge_model=CFG["judge_model"])
    print(f"  retrieved {len(res['contexts'])} chunks, blocked={res['blocked']}")
    print("  answer:", res["answer"][:250], "…")

    print("\nSmoke test complete. If models answered, run:  python -m src.run_experiments")


if __name__ == "__main__":
    main()
