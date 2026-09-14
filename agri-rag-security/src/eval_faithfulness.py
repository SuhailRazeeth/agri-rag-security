"""Faithfulness / answer-quality evaluation (the 'faithful' axis).

Primary path: RAGAS (faithfulness, answer_relevancy) driven through the LiteLLM
gateway. If RAGAS or its LangChain shims aren't importable, we fall back to an
equivalent LLM-judge so the pipeline still runs end-to-end. Which path was used
is recorded in the results (`faith_backend` column).
"""
from __future__ import annotations

import json
import re

from src.config import CFG
from src.gateway import simple

# ----------------------------------------------------------------------------
# Try to build a real RAGAS scorer once, lazily.
# ----------------------------------------------------------------------------
_RAGAS = {"ready": False, "tried": False, "score": None}


def _install_vertexai_shim():
    """ragas 0.4.x hard-imports langchain_community.chat_models.vertexai, which
    newer langchain-community removed. Provide a harmless stub so ragas imports.
    We never use Vertex — the judge routes through OpenAI."""
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        shim = types.ModuleType(name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = shim


def _init_ragas():
    _RAGAS["tried"] = True
    judge_model = CFG["judge_model"]
    # RAGAS is driven here by langchain_openai.ChatOpenAI, so the judge must be
    # an OpenAI model. If it isn't, skip RAGAS and use the LLM-judge fallback.
    if not judge_model.startswith("openai/"):
        print(f"[eval] RAGAS needs an OpenAI judge_model (got {judge_model}); "
              "using LLM-judge fallback.")
        _RAGAS["ready"] = False
        return
    try:
        _install_vertexai_shim()
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings

        openai_model = judge_model.split("/", 1)[1]   # "openai/gpt-4o-mini" -> "gpt-4o-mini"
        judge = LangchainLLMWrapper(ChatOpenAI(model=openai_model, temperature=0.0))
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=CFG["embedding_model"])
        )
        metrics = [Faithfulness(), ResponseRelevancy()]

        def score(question, answer, contexts, ground_truth):
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=list(contexts),
                reference=ground_truth,
            )
            ds = EvaluationDataset(samples=[sample])
            res = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb,
                           show_progress=False)
            df = res.to_pandas()
            return {
                "faithfulness": float(df["faithfulness"].iloc[0]),
                "answer_relevancy": float(df["answer_relevancy"].iloc[0]),
            }

        _RAGAS["score"] = score
        _RAGAS["ready"] = True
        print("[eval] RAGAS backend ready (ragas 0.4.x via OpenAI judge).")
    except Exception as e:
        print(f"[eval] RAGAS unavailable ({type(e).__name__}: {str(e)[:80]}); "
              "using LLM-judge fallback.")
        _RAGAS["ready"] = False


# ----------------------------------------------------------------------------
# Fallback LLM judge (self-contained).
# ----------------------------------------------------------------------------
_FALLBACK_PROMPT = """Score the ANSWER to a farmer's QUESTION given the retrieved CONTEXT.
Return ONLY a JSON object: {{"faithfulness": <0-1>, "answer_relevancy": <0-1>}}.
- faithfulness = fraction of the answer that is supported by the context (1 = fully grounded, 0 = fabricated).
- answer_relevancy = how directly the answer addresses the question (1 = perfectly, 0 = off-topic).

QUESTION: {q}
CONTEXT: {ctx}
ANSWER: {a}
JSON:"""


def _fallback_score(question, answer, contexts, ground_truth):
    ctx = "\n".join(contexts)[:4000]
    prompt = _FALLBACK_PROMPT.format(q=question, ctx=ctx, a=answer[:2000])
    try:
        raw = simple(CFG["judge_model"], prompt, temperature=0.0, max_tokens=60)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        return {
            "faithfulness": float(obj.get("faithfulness", 0.0)),
            "answer_relevancy": float(obj.get("answer_relevancy", 0.0)),
        }
    except Exception:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def score_faithfulness(question, answer, contexts, ground_truth) -> dict:
    """Return {'faithfulness', 'answer_relevancy', 'faith_backend'}."""
    if not answer or not contexts:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "faith_backend": "skipped"}

    if not _RAGAS["tried"]:
        _init_ragas()

    if _RAGAS["ready"]:
        try:
            s = _RAGAS["score"](question, answer, contexts, ground_truth)
            s["faith_backend"] = "ragas"
            return s
        except Exception:
            pass  # fall through to fallback on per-sample failure

    s = _fallback_score(question, answer, contexts, ground_truth)
    s["faith_backend"] = "llm_judge"
    return s
