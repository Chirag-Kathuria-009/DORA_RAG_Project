"""RAGAS evaluation of the DORA RAG pipeline.

Judged by Groq-hosted models, never by Gemini — the model that generated the
answer must not grade it.

Groq's free tier limits tokens per MINUTE per model (measured: 8,000 TPM on the
gpt-oss and qwen models). A faithfulness call carrying five context chunks runs
2-4k tokens, so one model sustains only two or three judge calls a minute. Two
consequences shape this file:

  1. Metrics are spread across models so each draws its own TPM bucket.
  2. Judge calls are paced. RunConfig(max_workers=1) serialises them but adds no
     delay, so RAGAS otherwise fires them back to back and exhausts the minute's
     budget in seconds — which is what kills a run partway through.

Judges are assigned by how many tokens each metric actually burns:

  faithfulness       heaviest  - extracts claims, NLI-verifies each vs context
  context_precision  heavy     - one verdict call per retrieved chunk
  context_recall     moderate  - splits the reference into statements
  answer_relevancy   lightest  - short generation, then local embeddings

groq/compound-mini advertises 70k TPM but is deliberately unused: it is an
agentic system that can invoke web search, so it may grade against the internet
rather than against the retrieved context.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.chain import build_generation_chain
from src.config import settings
from src.retriever import build_retriever

'''
Important points:
Question
   ↓
[RETRIEVAL] ────────── context_recall     "did we fetch what was needed?"
   ↓
[RANKING] ──────────── context_precision  "is what we fetched mostly useful?"
   ↓
[GENERATION] ───────── faithfulness       "did the model stay inside the context?"
   ↓
[RESPONSIVENESS] ───── answer_relevancy   "did it answer the question asked?"
'''

OUT_DIR = Path("data/eval")
GOLDEN_PATH = OUT_DIR / "golden_dataset.json"
CHECKPOINT_PATH = OUT_DIR / "eval_checkpoint.json"
METRIC_CACHE_PATH = OUT_DIR / "metric_cache.json"
CHUNKS_PATH = Path("data/chunks_cache.json")

# The exact string the system prompt tells the model to emit when the context
# does not answer the question. Detected so refusals report separately.
REFUSAL_MARKER = "not addressed in the provided DORA documentation sections"

# Ordered fallbacks per metric: if the first model is rate-limited or retired,
# the next is tried rather than losing the metric. Verified against the
# account's /models listing — llama-3.3-70b-versatile was decommissioned and 404s.
JUDGE_BY_METRIC = {
    "faithfulness":      ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "context_precision": ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
    "context_recall":    ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
    "answer_relevancy":  ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
}

# 8,000 TPM / ~3,000 tokens per judge call ~= 2.6 calls per minute.
JUDGE_REQUESTS_PER_SECOND = 0.045

METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


# --------------------------------------------------------------------------
# staleness detection
# --------------------------------------------------------------------------

def pipeline_fingerprint() -> str:
    """Identify the corpus and config a cached result was produced under.

    Cached predictions and scores are only meaningful for the pipeline that
    produced them. Re-ingesting the PDF or changing retrieval settings makes
    them describe something that no longer exists — and because
    collect_predictions skips questions already in the checkpoint, a stale
    checkpoint is silently reused for the entire run and nothing regenerates.
    That failure is invisible in the output, so it is detected here instead:
    if the fingerprint moved, cached state is discarded rather than trusted.
    """
    h = hashlib.sha256()
    if CHUNKS_PATH.exists():
        h.update(CHUNKS_PATH.read_bytes())
    for part in (
        settings.llm_model,
        settings.embedding_model_name,
        str(settings.retrieval_top_k),
        str(settings.rerank_top_n),
    ):
        h.update(part.encode())
    return h.hexdigest()[:16]


def _load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_stateful(path: Path, fingerprint: str, default):
    """Load cached state, discarding it if it came from a different pipeline."""
    blob = _load_json(path, None)
    if not isinstance(blob, dict) or "fingerprint" not in blob:
        if blob is not None:
            print(f"Discarding {path.name}: written before fingerprinting, "
                  f"cannot confirm it matches the current corpus")
        return default
    if blob["fingerprint"] != fingerprint:
        print(f"Discarding {path.name}: corpus or retrieval config changed "
              f"since it was written ({blob['fingerprint']} -> {fingerprint})")
        return default
    return blob.get("data", default)


def save_stateful(path: Path, fingerprint: str, data):
    _save_json(path, {"fingerprint": fingerprint, "data": data})


# --------------------------------------------------------------------------
# prediction collection
# --------------------------------------------------------------------------

def _is_rate_limit(exc: BaseException) -> bool:
    """Retry throttling and transient server errors; fail fast on everything else.

    The previous decorator retried every exception, so an invalid model or bad
    API key burned five exponential-backoff attempts before surfacing — which is
    why a configuration error presented as a hang.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        s in text
        for s in ("rate limit", "429", "quota", "timeout", "503", "502", "overloaded")
    )


@retry(
    retry=retry_if_exception(_is_rate_limit),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def safe_invoke(runnable, payload):
    return runnable.invoke(payload)


def sample_records(records: list, n: int | None) -> list:
    """Evenly spaced subset, so the sample spans the whole regulation."""
    if n is None or n >= len(records):
        return records
    step = len(records) // n
    return [records[i] for i in range(0, len(records), step)][:n]


def collect_predictions(golden_dataset: list, delay: float, fingerprint: str) -> list:
    """Run the pipeline over the questions, checkpointing after each one.

    Retrieval happens once per question and those same documents both generate
    the answer and get scored. The previous version invoked the chain and then
    the retriever separately, doubling the work and leaving the judged contexts
    unprovably different from the ones the answer came from.
    """
    retriever = build_retriever()
    generation_chain = build_generation_chain()

    records = load_stateful(CHECKPOINT_PATH, fingerprint, [])
    completed = {r["question"] for r in records}
    if completed:
        print(f"Resuming: {len(completed)} question(s) already collected")
    failures = []

    for i, entry in enumerate(golden_dataset, 1):
        question = entry.get("question")
        if question in completed:
            continue

        try:
            docs = safe_invoke(retriever, question)
            if not docs:
                print(f"[{i}] no chunks retrieved, skipping: {question[:60]}")
                failures.append(question)
                continue

            answer = safe_invoke(
                generation_chain, {"context": docs, "question": question}
            )

            records.append({
                "question": question,
                "answer": answer,
                "context": [d.page_content for d in docs],
                "articles_retrieved": [d.metadata.get("article") for d in docs],
                "ground_truth": entry.get("ground_truth"),
                "article_reference": entry.get("article_reference"),
                "refused": REFUSAL_MARKER in answer,
            })
            save_stateful(CHECKPOINT_PATH, fingerprint, records)
            print(f"[{i}/{len(golden_dataset)}] {question[:60]}")

        except Exception as e:
            # continue, not break: one bad question previously aborted the
            # entire collection loop and discarded every question after it.
            print(f"[{i}] FAILED {type(e).__name__}: {e}")
            failures.append(question)
            continue

        time.sleep(delay)

    if failures:
        print(f"\n{len(failures)} question(s) failed and were skipped")
    return records


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------

def build_judge(model_name: str) -> LangchainLLMWrapper:
    """A rate-limited Groq judge.

    The limiter is the point: it paces calls beneath the per-model
    tokens-per-minute ceiling. Without it RAGAS saturates the budget in seconds
    and the run dies partway through with a 429.
    """
    return LangchainLLMWrapper(
        ChatGroq(
            model=model_name,
            temperature=0,
            api_key=settings.groq_api_key,
            max_retries=2,          # RunConfig retries too; keep the layers shallow
            timeout=90,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=JUDGE_REQUESTS_PER_SECOND,
                check_every_n_seconds=0.5,
                max_bucket_size=1,
            ),
        )
    )


def evaluate_metric(dataset, metric, judge_llm, judge_embeddings):
    """Score one metric. Returns (score, per-question series, n_parsed, n_total)."""
    name = metric.name
    result = evaluate(
        dataset,
        llm=judge_llm,
        embeddings=judge_embeddings,
        metrics=[metric],
        raise_exceptions=False,
        run_config=RunConfig(
            max_workers=1,   # sequential; the rate limiter handles pacing
            timeout=180,
            max_retries=3,
            max_wait=60,
        ),
    )

    df = result.to_pandas()

    # Prefer the column named after the metric. The old positional cols[0]
    # silently picked the wrong column whenever RAGAS added an output field.
    inputs = {
        "user_input", "response", "retrieved_contexts", "reference",
        "question", "answer", "ground_truth", "contexts",
    }
    col = name if name in df.columns else next(
        (c for c in df.columns if c not in inputs), None
    )
    if col is None:
        return None, None, 0, len(df)

    series = pd.to_numeric(df[col], errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return None, series.rename(name), 0, len(series)

    return float(valid.mean()), series.rename(name), len(valid), len(series)


def summarise_by_refusal(per_question: pd.DataFrame, metric_names: list) -> None:
    """Report refused and answered questions separately.

    A large share of the golden set is refused by design — the system prompt
    instructs the model to decline when the context lacks the answer. RAGAS
    scores a noncommittal answer as ~0 relevancy, so averaging refusals together
    with real answers yields a number describing neither. Split apart, a low
    refusal score indicates retrieval missed, not that generation is weak.
    """
    if "refused" not in per_question.columns or not metric_names:
        return

    print("\n" + "-" * 64)
    print("SPLIT BY REFUSAL  (refusals score ~0 by design, not by failure)")
    print("-" * 64)
    print(f"{'group':<14}{'n':>4}" + "".join(f"{m[:15]:>17}" for m in metric_names))

    for label, mask in [
        ("answered", ~per_question["refused"]),
        ("refused", per_question["refused"]),
    ]:
        subset = per_question[mask]
        if subset.empty:
            continue
        cells = "".join(
            f"{subset[m].mean():>17.3f}"
            if m in subset.columns and subset[m].notna().any()
            else f"{'-':>17}"
            for m in metric_names
        )
        print(f"{label:<14}{len(subset):>4}{cells}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run_eval(sample_size: int | None, delay: float, fresh: bool):
    fingerprint = pipeline_fingerprint()
    print(f"Pipeline fingerprint: {fingerprint}")

    if fresh:
        for path in (METRIC_CACHE_PATH, CHECKPOINT_PATH):
            if path.exists():
                path.unlink()
                print(f"Cleared {path.name}")

    golden_dataset = _load_json(GOLDEN_PATH, [])

    # Sample BEFORE predicting. The old order generated all 56 answers and then
    # discarded 31, paying full cost for questions that were never scored.
    selected = sample_records(golden_dataset, sample_size)
    print(f"Evaluating {len(selected)} of {len(golden_dataset)} golden questions")

    predictions = collect_predictions(selected, delay, fingerprint)

    wanted = {e["question"] for e in selected}
    predictions = [r for r in predictions if r["question"] in wanted]
    if not predictions:
        print("No predictions collected — nothing to score.")
        return {}

    refused = sum(1 for r in predictions if r.get("refused"))
    print(f"\nCollected {len(predictions)} predictions ({refused} refusals)")

    dataset = Dataset.from_dict({
        "user_input":         [r["question"] for r in predictions],
        "response":           [r["answer"] for r in predictions],
        "retrieved_contexts": [r["context"] for r in predictions],
        "reference":          [r["ground_truth"] for r in predictions],
    })

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    per_question = pd.DataFrame({
        "question": [r["question"] for r in predictions],
        "refused": [bool(r.get("refused")) for r in predictions],
    })

    cache = load_stateful(METRIC_CACHE_PATH, fingerprint, {})
    scores, judges_used, coverage = {}, {}, {}

    for metric in METRICS:
        name = metric.name
        cached = cache.get(name, {})

        if cached.get("score") is not None:
            print(f"\n--- {name}: cached ({cached['score']:.4f}) ---")
            scores[name] = cached["score"]
            judges_used[name] = cached.get("judge")
            coverage[name] = cached.get("coverage", "unknown")
            continue

        for model_name in JUDGE_BY_METRIC.get(name, ["openai/gpt-oss-20b"]):
            print(f"\n--- {name} (judge: {model_name}) ---")
            try:
                score, series, n_ok, n_tot = evaluate_metric(
                    dataset, metric, build_judge(model_name), judge_embeddings
                )
            except Exception as e:
                print(f"  judge {model_name} failed: {type(e).__name__}: {e}")
                continue

            if score is None:
                print(f"  {model_name} parsed 0/{n_tot} — trying next judge")
                continue

            print(f"  {name}: {score:.4f}  ({n_ok}/{n_tot} parsed)")
            scores[name] = score
            judges_used[name] = model_name
            coverage[name] = f"{n_ok}/{n_tot}"
            if series is not None:
                per_question = pd.concat(
                    [per_question, series.reset_index(drop=True)], axis=1
                )
            break
        else:
            print(f"  all judges failed for {name}")
            scores[name] = None
            judges_used[name] = None
            coverage[name] = "0/0"

        # Cache after every metric so an interrupted run resumes cheaply.
        cache[name] = {
            "score": scores[name],
            "judge": judges_used[name],
            "coverage": coverage[name],
        }
        save_stateful(METRIC_CACHE_PATH, fingerprint, cache)
        time.sleep(10)

    print("\n" + "=" * 64)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 64)
    print(f"{'metric':<22}{'score':>10}{'parsed':>10}  judge")
    for name, score in scores.items():
        shown = f"{score:.4f}" if score is not None else "FAILED"
        print(f"{name:<22}{shown:>10}{str(coverage.get(name, '-')):>10}  "
              f"{judges_used.get(name) or '-'}")
    print("=" * 64)
    print("'parsed' is each metric's own denominator — metrics judged over "
          "different subsets\nare not directly comparable.")

    scored_metrics = [m for m in scores if m in per_question.columns]
    summarise_by_refusal(per_question, scored_metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(OUT_DIR / "evaluation_results.json", {
        "fingerprint": fingerprint,
        "generation_model": settings.llm_model,
        "judges_by_metric": judges_used,      # the judges actually used
        "coverage_by_metric": coverage,       # how many questions each parsed
        "questions_evaluated": len(predictions),
        "refusals": refused,
        "scores": scores,
    })
    per_question.to_csv(OUT_DIR / "per_question_results.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/")

    return scores


def main():
    p = argparse.ArgumentParser(description="RAGAS evaluation of the DORA pipeline")
    p.add_argument("--sample", type=int, default=25,
                   help="questions to evaluate (sampled before generation)")
    p.add_argument("--delay", type=float, default=20.0,
                   help="seconds between generation calls (Gemini pacing)")
    p.add_argument("--fresh", action="store_true",
                   help="delete cached predictions and scores, then re-run")
    args = p.parse_args()
    run_eval(args.sample, args.delay, args.fresh)


if __name__ == "__main__":
    main()


'''
Important conclusions to remember from results:

The diagnostic power comes from reading them as pairs, not individually:

Pattern	Failing stage	What's actually wrong
Low recall + low precision	Retrieval	Embeddings, chunking, or query mismatch
High recall + low precision	Ranking	Right chunks found, ranked poorly
High precision + low faithfulness	Generation	Model ignoring good context, prompt too weak
High faithfulness + low relevancy	Generation	Over-refusing, or hedging instead of answering
All four low	Corpus	The answers aren't in your documents

For tuning retrieval, prefer eval/retrieval_eval.py — it answers the same
retrieval question deterministically against article_reference in seconds, with
no judge and no rate limit. Use this RAGAS run for end-to-end validation once
retrieval is settled.
'''
