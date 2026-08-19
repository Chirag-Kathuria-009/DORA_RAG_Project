"""Deterministic retrieval evaluation — no LLM judge, no API calls.

RAGAS already measures retrieval through context_precision and context_recall,
so this is not a different question. It is the same question asked in a way you
can tune against: RAGAS needs ~15 minutes and Groq quota, varies run to run with
judge parsing, and only ever sees the final contexts. That last part is the real
limitation — it cannot tell you whether a missing chunk was never retrieved or
was retrieved and then ranked out by the cross-encoder, and those have opposite
fixes.

Scoring is deterministic because the golden dataset records an article_reference
per question and ingest tags every chunk with the Article it belongs to, so a
retrieved chunk is objectively right or wrong.

Pipeline construction is imported from src.retriever — this module only adds
what is genuinely new: parsing article references and scoring rank positions.

  Hit@5      correct Article in the top 5 — what the LLM actually receives
  MRR        1/rank of the first correct chunk — how highly it was ranked
  Recall@k   correct Article anywhere in the candidate set — the ceiling that
             reranking can possibly achieve

  low Recall@k             -> never retrieved; fix chunking/embeddings/weights
  high Recall@k, low Hit@5 -> retrieved then ranked out; fix the reranker
"""

import argparse
import json
import re
import time
from pathlib import Path

from src.config import settings
from src.retriever import build_stages

GOLDEN = Path("data/eval/golden_dataset.json")
OUT_DIR = Path("data/eval")

# "Article 8(1) & (4)" and "Article 5(2)(a)" each name one Article; a few
# entries name two ("Articles 5 and 6"), so collect every number present.
ARTICLE_NUM = re.compile(r"Article[s]?\s+(\d+)")


def expected_articles(reference: str) -> set[int]:
    return {int(n) for n in ARTICLE_NUM.findall(reference or "")}


def score_stage(retriever, dataset: list, cutoff: int) -> dict:
    """Run every question through one stage and score the Articles it returned."""
    hits = recalls = rr_total = 0.0
    scored = 0
    misses = []

    for entry in dataset:
        want = expected_articles(entry.get("article_reference", ""))
        if not want:
            continue  # no ground-truth Article to score against
        scored += 1

        docs = retriever.invoke(entry["question"])
        got = [d.metadata.get("article") for d in docs]

        if any(a in want for a in got[:cutoff]):
            hits += 1
        else:
            misses.append((entry["article_reference"], entry["question"]))

        if any(a in want for a in got):
            recalls += 1

        for rank, article in enumerate(got, 1):
            if article in want:
                rr_total += 1.0 / rank
                break

    return {
        "n": scored,
        "hit": hits / scored if scored else 0.0,
        "mrr": rr_total / scored if scored else 0.0,
        "recall": recalls / scored if scored else 0.0,
        "misses": misses,
    }


def _report(name: str, r: dict, elapsed: float, cutoff: int) -> None:
    print(
        f"{name:<22}{r['hit']:>10.3f}{r['mrr']:>9.3f}"
        f"{r['recall']:>11.3f}{elapsed:>8.1f}",
        flush=True,
    )


def run(dataset, top_k, rerank_top_n, bm25_weight, rerankers, cutoff, show_misses):
    """Score the retrieval stages, then each candidate reranker.

    bm25/vector/hybrid do not depend on the reranker, so they are scored once
    and reused across every reranker being compared — otherwise a 3-way
    comparison pays for the same hybrid retrieval three times.
    """
    print(f"\n{'=' * 78}")
    print(f"top_k={top_k}  rerank_top_n={rerank_top_n}  bm25_weight={bm25_weight}")
    print("=" * 78)
    print(
        f"{'stage':<22}{'hit@' + str(cutoff):>10}{'MRR':>9}{'recall@k':>11}{'sec':>8}",
        flush=True,
    )

    # Candidate stages: identical for every reranker, so build and score once.
    stages = build_stages(
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        bm25_weight=bm25_weight,
        reranker_model=None,
    )

    results = {}
    for name in ("bm25", "vector", "hybrid"):
        t0 = time.time()
        results[name] = score_stage(stages[name], dataset, cutoff)
        _report(name, results[name], time.time() - t0, cutoff)

    ceiling = results["hybrid"]["recall"]

    for model in rerankers:
        if not model:
            continue
        label = f"reranked:{model.split('/')[-1].replace('bge-reranker-', '')}"
        reranked = build_stages(
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            bm25_weight=bm25_weight,
            reranker_model=model,
        )["reranked"]
        t0 = time.time()
        results[label] = score_stage(reranked, dataset, cutoff)
        _report(label, results[label], time.time() - t0, cutoff)

    if ceiling:
        print(
            f"\nCandidate ceiling (hybrid recall@{top_k}): {ceiling:.3f} — "
            f"no reranker can exceed this without better retrieval",
            flush=True,
        )
        for label, r in results.items():
            if label.startswith("reranked"):
                print(
                    f"  {label:<22} captured {r['hit'] / ceiling * 100:5.1f}% of it",
                    flush=True,
                )

    if show_misses:
        worst = next(
            (r for k, r in results.items() if k.startswith("reranked")),
            results["hybrid"],
        )
        print(f"\nMissed ({len(worst['misses'])}):")
        for ref, question in worst["misses"]:
            print(f"  {ref:<22} {question[:64]}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    p.add_argument("--rerank-top-n", type=int, default=settings.rerank_top_n)
    p.add_argument("--bm25-weight", type=float, default=0.4)
    p.add_argument("--reranker", default="BAAI/bge-reranker-base",
                   help="cross-encoder model, or 'none' to skip reranking")
    p.add_argument("--compare-rerankers", action="store_true",
                   help="score none/base/large to see if the bigger model earns its cost")
    p.add_argument("--cutoff", type=int, default=5)
    p.add_argument("--show-misses", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Loaded {len(dataset)} golden questions")

    rerankers = (
        ["BAAI/bge-reranker-base", "BAAI/bge-reranker-large"]
        if args.compare_rerankers
        else ([] if args.reranker == "none" else [args.reranker])
    )

    all_results = run(
        dataset, args.top_k, args.rerank_top_n, args.bm25_weight,
        rerankers, args.cutoff, args.show_misses,
    )

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "retrieval_results.json"
        out.write_text(
            json.dumps(
                {
                    stage: {k: v for k, v in s.items() if k != "misses"}
                    for stage, s in all_results.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
