# Engineering Decisions

A record of the significant choices made on this project, what evidence drove
each one, and what was rejected along the way. Numbers here come from tests run
against the actual corpus and the 56-question golden dataset, not from
estimates. Where something was not measured, it says so.

---

## Contents

1. [PDF extraction library](#1-pdf-extraction-library)
2. [Text repair: root cause vs post-processing](#2-text-repair-root-cause-vs-post-processing)
3. [Building a deterministic retrieval eval](#3-building-a-deterministic-retrieval-eval)
4. [Reranker model: base vs large](#4-reranker-model-base-vs-large)
5. [BM25 fusion weight](#5-bm25-fusion-weight)
6. [rerank_top_n](#6-rerank_top_n)
7. [Judge model selection](#7-judge-model-selection)
8. [Multi-judge vs single judge](#8-multi-judge-vs-single-judge)
9. [Rate limiting placement](#9-rate-limiting-placement)
10. [Refusals as a separate bucket](#10-refusals-as-a-separate-bucket)
11. [Generation temperature](#11-generation-temperature)
12. [Article metadata tagging](#12-article-metadata-tagging)
13. [Cache invalidation by fingerprint](#13-cache-invalidation-by-fingerprint)
14. [Ingest reset semantics](#14-ingest-reset-semantics)
15. [Config: embedding model name](#15-config-embedding-model-name)
16. [Open questions](#open-questions)

---

## 1. PDF extraction library

**Decision:** replace `PyPDFLoader` (pypdf) with `PyMuPDFLoader`.

### How the problem surfaced

Not from a metric. While inspecting chunk text to work out how to tag Articles,
the extracted text visibly contained broken words:

```
propor tionality   unif orm   netw ork   secur ity   system s
```

### Why it happens

A PDF has no concept of a word. The content stream positions glyphs, and the
extractor reconstructs word boundaries by measuring the horizontal gap between
consecutive glyphs and calling anything over a threshold a space.

EUR-Lex publishes **justified** text. To make both margins flush the typesetter
widens letter-spacing *inside* words, not only the spaces between them. On a
stretched line the gap between two letters crosses pypdf's threshold and a space
is emitted mid-word.

This is a generic failure mode for justified PDFs (regulations, ISO standards,
filings, academic papers), not a DORA-specific quirk.

### Test

Counted intact vs broken occurrences of key regulatory terms across the whole
document. "Broken" means the same letters with one space inserted anywhere
inside.

Per-term, on the original pypdf output:

| term | intact | broken | example |
|---|---|---|---|
| security | **0** | **82** | `secur ity` |
| reporting | **0** | 49 | `repor ting` |
| network | 8 | 22 | `netw ork`, `netwo rk` |
| framework | 23 | 51 | `framew ork`, `framewo rk` |
| provider | 70 | 100 | `provi der`, `provid er` |
| management | 61 | 77 | `manag ement`, `manageme nt` |
| information | 175 | 1 | `informa tion` |
| financial | 525 | 0 | - |
| critical | 232 | 0 | - |
| resilience | 92 | 0 | - |

### Why this mattered so much

`BM25Retriever` tokenises on whitespace. A query containing "security" could not
match anything in the corpus, because the intact token occurred **zero** times.
Half of the hybrid retriever was structurally incapable of matching the single
most important term in a document about operational security.

Dense retrieval degrades too (`secur` and `ity` tokenise to something unrelated
to `security` in embedding space) but BM25 fails completely rather than
partially.

### Extractor comparison

Full document, identical scoring function, 12 terms:

| extractor | intact | broken | corrupt | time |
|---|---|---|---|---|
| pypdf (default) - original | 637 | 402 | **38.7%** | 1.9s |
| pypdf `extraction_mode="layout"` | 880 | 193 | 18.0% | 3.7s |
| **PyMuPDF** | **1112** | **0** | **0.0%** | **1.8s** |
| pdfplumber | 1112 | 0 | 0.0% | 11.5s |

Also tested pypdf's `space_width` parameter at 100 / 250 / 500 with **no effect
at any value**. The threshold is not the tunable part of the problem.

### Considered and rejected

- **pypdf `layout` mode** halves the corruption but does not eliminate it
  (`security` still 72 broken). Also produces 800k chars vs 310k, mostly
  whitespace padding.
- **pdfplumber** gives identical quality to PyMuPDF at 6x the runtime (11.5s vs
  1.8s). No reason to prefer it here.
- **Tuning `space_width`** was measured and had no effect.

### Outcome

PyMuPDF eliminates the corruption entirely, is the fastest option tested, and
recovers 475 keyword occurrences that were previously unmatchable. The change is
one import in `src/ingest.py`.

---

## 2. Text repair: root cause vs post-processing

**Decision:** fix the extraction, do not post-process the text.

### What was initially proposed

A dictionary-based repair pass: merge two adjacent fragments when the joined form
is a real English word and at least one fragment is not a standalone word, so
`secur|ity` merges while `in|to` and `a|ware` do not. Vocabulary would come from
the `bge-small-en-v1.5` tokenizer plus words appearing intact in the document.

### Why it was rejected

Raised during review: the vocabulary approach is tuned to this specific corpus,
and a different document would need the logic revisited.

That objection is correct, and the approach had worse problems on inspection:

- It treats a symptom. The corruption is introduced at extraction; repairing it
  downstream leaves bad text in the pipeline until then.
- It cannot fix what is not in the wordlist. `securities` appeared 22 times
  broken, and a general English vocabulary may or may not contain it.
- It carries permanent false-merge risk on every future document.

Switching extractor fixes the cause, needs no wordlist, and generalises to any
justified PDF. The measured result was 0% corruption, leaving nothing for a
repair pass to do.

### Principle

Prefer fixing where the defect is introduced over correcting it downstream. The
post-processing option looked attractive only because the extraction options had
not been measured yet.

---

## 3. Building a deterministic retrieval eval

**Decision:** add `eval/retrieval_eval.py`, scoring retrieval with no LLM judge.

### Why RAGAS alone was insufficient for tuning

RAGAS does measure retrieval. `context_precision` and `context_recall` are
retrieval metrics, not generation metrics. The problem is cost and resolution:

| | RAGAS `context_*` | deterministic |
|---|---|---|
| runtime | ~15 min | seconds |
| needs Groq quota | yes | no |
| reproducible | no (judge variance, parse failures) | exactly |
| stage attribution | final contexts only | bm25 / vector / hybrid / reranked |

The last row is the structural limitation. RAGAS only sees the contexts that
reached the LLM, so it cannot distinguish:

- the right chunk was **never retrieved**, which means fixing chunking,
  embeddings or weights
- the right chunk **was retrieved and then ranked out**, which means fixing the
  reranker

Those have opposite fixes, and you cannot tune against a signal that takes 15
minutes and moves when you rerun it unchanged.

### What made deterministic scoring possible

The golden dataset already recorded an `article_reference` per question
(`"Article 19(3)"`). Once ingest tags every chunk with its owning Article
(see section 12), a retrieved chunk is objectively correct or not. No judge
required.

### Metrics

- **Hit@5** - correct Article in the top 5, i.e. what the LLM actually receives
- **MRR** - 1/rank of the first correct chunk, i.e. how highly it ranked
- **Recall@k** - correct Article anywhere in the candidate set, i.e. the ceiling
  reranking could possibly achieve

### Validation against RAGAS

Before the extraction fix, a naive article-string match scored **0.589** while
RAGAS `context_recall` on the same pipeline scored **0.646**. Two independent
methods, one with an LLM judge and one without, landing close together gave
enough confidence to tune against the cheap one.

### Code reuse

The first draft built the retrieval stages inside the eval script. This was
rejected during review as duplicating `src/retriever.py`, which was the correct
call. Pipeline construction now lives only in `src/retriever.py`
(`build_stages()`), and the eval module contains only what is genuinely new:
article-reference parsing and rank scoring.

---

## 4. Reranker model: base vs large

**Decision:** keep `BAAI/bge-reranker-large`.

### Question being tested

`bge-reranker-large` was adopted while trying to raise metrics on the corrupted
corpus. Once extraction was fixed, was the larger model still needed, or was it
compensating for a problem that no longer existed?

### Test

56 questions, all stages scored, `top_k=20`, `rerank_top_n=5`:

| stage | hit@5 | MRR | recall@20 | questions hit |
|---|---|---|---|---|
| bm25 | 0.554 | 0.457 | 0.893 | 31/56 |
| vector | 0.821 | 0.663 | 0.982 | 46/56 |
| hybrid | 0.821 | 0.669 | **0.982** | 46/56 |
| reranked - **base** | 0.804 | 0.671 | - | 45/56 |
| reranked - **large** | **0.893** | **0.774** | - | **50/56** |

### Reading

- **base vs no reranking at all is 45 vs 46 questions.** A one-question
  difference on n=56 is noise. The base cross-encoder is contributing nothing.
- **large vs base is 50 vs 45 questions**, with MRR 0.774 vs 0.671. That gap is
  large enough to act on.

So the hypothesis was wrong: fixing ingestion did **not** make the base model
sufficient. Large earns its place; base does not justify being in the pipeline
at all.

### Cost caveat (unresolved)

The large model took ~1337s for 56 questions, roughly **24s per query** on CPU,
against 4.4s total (0.08s/query) for hybrid retrieval. The cross-encoder scores
every candidate pair, and the ensemble supplies up to 40.

`src/api.py` uses the same retriever, so `POST /query` carries that latency. See
[Open questions](#open-questions).

---

## 5. BM25 fusion weight

**Decision:** 0.3 is marginally best, but this knob is low-impact and not worth
further tuning.

### Test

Swept `bm25_weight` on the hybrid stage (pre-rerank), 56 questions:

| bm25_weight | hit@5 | MRR | recall@20 |
|---|---|---|---|
| 0.0 (vector only) | 0.821 | 0.663 | 0.982 |
| 0.2 | 0.821 | 0.683 | 0.982 |
| **0.3** | 0.821 | **0.690** | 0.982 |
| 0.4 (original) | 0.821 | 0.669 | 0.982 |
| 0.5 | 0.804 | 0.632 | 0.982 |

### Findings

1. **`recall@20` is 0.982 at every weight, including 0.0.** BM25 surfaces no
   candidate that dense retrieval missed. Its entire contribution is reordering
   candidates the vector search already found.
2. **`hit@5` is flat from 0.0 to 0.4**, then degrades at 0.5. Weighting BM25
   above roughly 0.4 actively hurts.
3. **MRR peaks at 0.3** (0.690 vs 0.669 at the original 0.4). Real but small.

### Interpretation

Hybrid retrieval is barely outperforming dense retrieval alone here: hit@5 is
identical and MRR gains 0.006 at the original weight. BM25 is worth keeping for
the small ranking improvement it feeds the reranker, but it is not carrying the
system.

Worth noting this measurement is only meaningful **after** the extraction fix.
On the corrupted corpus BM25 was crippled, so any weight tuning done then was
measuring the wrong thing.

---

## 6. rerank_top_n

**Decision:** keep `rerank_top_n=5`.

### Test

`top_k=20`, `bm25_weight=0.3`, large reranker. The cutoff moves with
`rerank_top_n`, so the hybrid column rises too because it is measuring more
positions.

| rerank_top_n | hybrid hit@n | reranked hit@n | MRR | reranker gain |
|---|---|---|---|---|
| **5** | 0.821 | **0.893** (50/56) | 0.774 | **+4 questions** |
| 8 | 0.893 | 0.893 (50/56) | 0.774 | 0 |
| 10 | 0.911 | 0.929 (52/56) | 0.778 | +1 question |

### Reading

- Going from 5 to 10 gains **2 questions** (50 to 52).
- At n=8 the reranker adds **literally nothing** over taking hybrid's top 8. The
  cross-encoder's entire value is concentrated at small n.

### Why 5 wins despite 10 scoring higher

`rerank_top_n` sets how much context reaches the LLM, and `context_precision`
makes **one judge call per retrieved chunk**. Going from 5 to 10 therefore
doubles the token burn on the same 8,000 TPM budget that was already breaking
evaluation runs (see section 9). Two questions out of 56 is not worth doubling
the constraint that was the actual blocker.

### Measurement caveat

Wall-clock times in this sweep (1337s / 1561s / 1232s for n=5/8/10) are **not
meaningful**. The cross-encoder scores all candidates regardless of `top_n`,
which only truncates the output, so reranking cost should be flat across this
sweep. The variance is machine noise and should not be read as a trend.

---

## 7. Judge model selection

**Decision:** replace `llama-3.3-70b-versatile`; assign judges by token cost.

### What was found

`llama-3.3-70b-versatile` was configured as the faithfulness judge and recorded
as `judge_model` in `evaluation_results.json`. Querying the account's model list
returned **404** for it, because Groq decommissioned the model. Every faithfulness
judge call was failing.

Available chat models on the account:

| model | requests/day | tokens/min |
|---|---|---|
| `openai/gpt-oss-120b` | 999 | 8,000 |
| `openai/gpt-oss-20b` | 999 | 8,000 |
| `qwen/qwen3.6-27b` | 999 | 8,000 |
| `groq/compound-mini` | 249 | **70,000** |
| `allam-2-7b` | 6,999 | 6,000 |

### Assignment, by how many tokens each metric burns

| metric | cost profile | judge |
|---|---|---|
| `faithfulness` | heaviest: extracts claims, NLI-verifies each against full context | `gpt-oss-120b` |
| `context_precision` | heavy: one verdict call **per chunk** | `gpt-oss-20b` |
| `context_recall` | moderate: splits reference into statements | `qwen/qwen3.6-27b` |
| `answer_relevancy` | lightest: short generation, then local embeddings | `gpt-oss-20b` |

The hardest metric gets the strongest model, and the rest are spread so each
draws on a separate TPM bucket.

### Considered and rejected

- **`groq/compound-mini`** despite its 70,000 TPM. It is an agentic system that
  can invoke web search and code execution. A judge that can reach the internet
  is not grading against the retrieved context, and its output structure is not
  predictable enough for RAGAS's JSON parsing.
- **`allam-2-7b`**, because 7B is too small to judge reliably and its TPM is the
  lowest available.
- **Gemini as judge**, excluded by design. The model that generated the answer
  must not grade it.

### Added: fallback chains

Each metric now has an ordered list of judges. If the first is rate-limited or
retired, the next is tried rather than losing the metric. The previous code had
one model per metric and `break` on failure, so a single dead model ended the
run.

---

## 8. Multi-judge vs single judge

**Decision:** keep the multi-judge design.

### Initial recommendation, and why it was wrong

A single judge across all four metrics was proposed first, on the grounds that
scores from different judges are not directly comparable. Different models have
different JSON reliability, failures become NaN, NaN rows get dropped, and each
metric ends up averaged over a different subset. Measured denominators from the
last run:

| metric | scored over |
|---|---|
| faithfulness | 23 / 25 |
| answer_relevancy | 24 / 25 |
| context_precision | 25 / 25 |
| context_recall | 24 / 25 |

Four averages over four different question subsets.

### The constraint that overrode it

The multi-judge design was a deliberate response to free-tier token limits: a
single judge exhausts its budget before finishing the run. Groq's limits are
**per model**, so spreading metrics across three models gives 24,000 TPM of
aggregate headroom instead of 8,000.

That is correct and it takes priority. The comparability problem is real but is a
**reporting** problem, not a reason to triple the chance of not finishing.

### Resolution

Multi-judge kept. The comparability issue is handled by making it visible rather
than by eliminating it:

- `coverage_by_metric` is recorded in `evaluation_results.json`
- the results table prints each metric's own denominator
- output states plainly that metrics judged over different subsets are not
  directly comparable

---

## 9. Rate limiting placement

**Decision:** rate-limit the judge calls, using `InMemoryRateLimiter`.

### The bug

`rate_limit_delay = 20` was applied inside `collect_predictions`, a loop that
calls **Gemini**. The token-limit breaches were happening during **Groq** judging,
which ran later inside `evaluate()` with `RunConfig(max_workers=1)`.

`max_workers=1` serialises calls but adds **no delay**. So RAGAS fired judge
calls back to back and exhausted an 8,000 TPM budget in seconds.

The throttle was protecting the wrong API.

### Fix

`ChatGroq` accepts a `rate_limiter`, and
`langchain_core.rate_limiters.InMemoryRateLimiter` is available in the installed
version (0.3.76). Set to 0.045 requests/second, derived from 8,000 TPM divided by
roughly 3,000 tokens per judge call, giving about 2.6 calls/minute.

### Also fixed: compounding retries

Three nested retry layers multiplied: tenacity (5) x `RunConfig.max_retries` (5)
x `ChatGroq.max_retries` (3), up to 75 attempts on a single call, which is why
failures presented as hangs. Reduced to 5 x 3 x 2.

### Also fixed: retry predicate

`@retry` was catching **every** exception, so an invalid model name or bad API
key consumed five exponential-backoff attempts before surfacing. It now retries
only on throttling and transient server errors (`429`, `rate limit`, `quota`,
`timeout`, `502`, `503`, `overloaded`), and configuration errors fail
immediately.

Verified:

```
429 rate limit exceeded    -> retry=True
model_not_found            -> retry=False
timeout                    -> retry=True
```

---

## 10. Refusals as a separate bucket

**Decision:** report refused and answered questions separately.

### What was found

The system prompt instructs the model to reply "This specific question is not
addressed in the provided DORA documentation sections" when the context does not
contain the answer. **22 of 56 golden questions (39%) triggered this.**

Of the 25 questions actually scored, 8 were refusals:

| group | n | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| answered | 17 | **0.870** | **0.819** | **0.775** | 0.688 |
| refused | 8 | 0.628 | 0.479 | 0.442 | 0.562 |

### Why this matters

RAGAS scores a noncommittal answer as roughly 0 relevancy **by design**.
Averaging refusals together with real answers produces a number describing
neither: the reported 0.720 answer_relevancy is a blend of "answered well" and
"correctly declined to answer".

Note that refusals also have low `context_precision` (0.442 vs 0.775). Those are
retrieval metrics, independent of the answer. So the model is refusing precisely
when retrieval failed, which means the prompt is working as intended and the low
scores are measuring a retrieval problem, not a generation problem.

Aggregated, that distinction is invisible.

### Implementation

Each checkpoint record carries a `refused` flag, and the results print an
answered/refused split table.

---

## 11. Generation temperature

**Decision:** pass `temperature=0.0` directly.

### The bug

```python
ChatGoogleGenerativeAI(
    model=...,
    generation_config=GenerationConfig(temperature=0.0),  # silently ignored
)
```

`generation_config` is not a recognised constructor parameter. LangChain moves it
into `model_kwargs` with a warning, and Gemini never receives it.

Verified directly:

```
current code      -> temperature = 0.7   model_kwargs = ['generation_config']
temperature=0.0   -> temperature = 0.0   model_kwargs = []
```

### Impact

The pipeline had been running at Gemini's default **0.7**, not 0.0. Two
consequences:

1. **Evaluation was non-deterministic.** Re-running produced different answers,
   so some observed metric variance was sampling noise rather than signal.
2. A regulatory compliance system was paraphrasing statutory text with creative
   sampling, which is the wrong behaviour for the use case regardless of metrics.

---

## 12. Article metadata tagging

**Decision:** tag each chunk with its owning Article number at ingest.

### Why

This is what makes deterministic retrieval scoring possible (section 3). The
golden dataset records `article_reference` per question; without a matching field
on each chunk there is nothing to compare against.

It also improves citation quality in the API response, which previously exposed
only page numbers.

### Implementation

Chunks are in document order, so a chunk with no heading of its own inherits the
last Article seen (carry-forward).

Distinguishing headings from cross-references matters: `Article 12` on its own
line is a heading, while the same string inside a sentence is a reference to
another Article. Only line-anchored matches count:

```python
ARTICLE_HEADING = re.compile(r"^[ \t]*Article\s+(\d+)[ \t]*$", re.MULTILINE)
```

Verified before implementing: 76 chunks contain a standalone `Article N` header
line, against 238 containing any mention of one. Using "any mention" would have
mis-tagged roughly two thirds of matches.

### Result

```
712 chunks from 79 pages
470/712 chunks tagged with an Article
63 distinct Articles, range 1-64
242 untagged (recitals, which precede Article 1 - correct)
```

---

## 13. Cache invalidation by fingerprint

**Decision:** fingerprint the corpus and retrieval config, and discard cached
state that does not match.

### The trap this closes

`collect_predictions` skips any question already present in
`eval_checkpoint.json`. That checkpoint held 56 predictions generated against the
**corrupted** corpus.

Running the eval after re-ingesting would have reused every one of them,
regenerated nothing, scored the old broken contexts, and reported results that
looked completely normal. A silent, invisible failure that would have made the
extraction fix appear to have done nothing.

The same applies to `metric_cache.json`, whose four scores describe a corpus that
no longer exists.

### Considered and rejected

Migrating the old cache format (`{name: score}`) into the new one so the four
scores would not be lost. Rejected on review, because the cached scores were
computed on the corrupted corpus at different retrieval settings, so preserving
them preserves misleading numbers. There was no benefit to keeping them.

### Implementation

`pipeline_fingerprint()` hashes `chunks_cache.json` plus `llm_model`,
`embedding_model_name`, `retrieval_top_k` and `rerank_top_n`. Cached state stores
its fingerprint, and on load a mismatch discards it with a message naming the
reason.

Verified against the existing stale files:

```
Discarding eval_checkpoint.json: written before fingerprinting, cannot confirm
  it matches the current corpus
Discarding metric_cache.json: written before fingerprinting, cannot confirm
  it matches the current corpus
checkpoint records reused: 0
cached metrics reused: 0
```

---

## 14. Ingest reset semantics

**Decision:** add `--reset` to `src/ingest.py`, and warn loudly when it is
omitted.

### The problem

Two storage paths with **opposite** semantics in the same script:

| path | operation | effect |
|---|---|---|
| `save_chunks_for_bm25()` | `open(path, "w")` | **overwrites** |
| `vectorstore.add_documents()` | insert with fresh UUIDs, no dedup | **appends** |

Re-running ingest without clearing would produce:

| retriever | reads | after re-ingest |
|---|---|---|
| BM25 | `chunks_cache.json` | 712 chunks, clean only |
| pgvector | `langchain_pg_embedding` | **1435**: 723 corrupted + 712 clean |

`EnsembleRetriever` fuses two rankings on the assumption they are views of one
corpus. They would not have been.

Downstream effects: every passage exists twice as near-identical vectors so both
can occupy slots in the same top-k, halving effective candidate diversity; the
reranker can return the same passage twice, dropping `context_precision` for
reasons unrelated to retrieval quality; and the resulting numbers could easily
have suggested the extraction fix did not help.

### Implementation

`--reset` calls `delete_collection()` then `create_collection()` (both confirmed
present on `langchain_postgres` 0.0.13) before indexing, making pgvector
overwrite-only like the BM25 cache. Omitting the flag prints a warning naming the
desync risk, since silent divergence is the whole problem.

---

## 15. Config: embedding model name

**Decision:** repurpose the dead `embedding_model: int = 384` field into
`embedding_model_name: str`.

### Reasoning

`"BAAI/bge-small-en-v1.5"` was hardcoded in three files: `src/ingest.py`,
`src/retriever.py` and `eval/run_eval.py`. If those ever drift apart, query
vectors get embedded by a different model than the index, and retrieval returns
quiet garbage with **no error**.

`settings.embedding_model` was verified to be defined and read nowhere, making it
a write-only field. Its value `384` is the output *dimensionality* of the model,
so the field was misnamed from the start; because nothing read it, the mismatch
never surfaced.

The type changed because the data changed: a dead field holding a dimension was
replaced by a live field holding a model identifier.

### Considered

Keeping both fields, or leaving config untouched and reverting ingest to the
literal. Both were viable. The repurpose was chosen because it leaves no dead
code and creates a single source of truth.

---

## Open questions

Things measured but not resolved.

### Reranker latency in the API

`bge-reranker-large` costs roughly 24s per query on CPU (1337s / 56 questions),
against 0.08s for hybrid retrieval. `src/api.py` uses the same retriever, so
`POST /query` carries that latency before Gemini is called.

The obvious lever is `top_k`: `recall@20` is 0.982, so 20 candidates per
retriever may be more than needed, and the cross-encoder cost scales with
candidate count. A `top_k` sweep at 8 and 12 was started but not completed.

Other options not evaluated: GPU inference, or a faster reranker that still beats
base.

### RAGAS scores post-fix

The four end-to-end scores have **not** been re-measured since the extraction
fix, the temperature fix and the re-ingest. The last recorded values
(faithfulness 0.786, answer_relevancy 0.720, context_precision 0.668,
context_recall 0.646) describe the corrupted corpus at `temperature=0.7` and
should not be cited.

Run `python -m eval.run_eval --fresh` to regenerate. `--fresh` is required
because the cached state predates the re-ingest, though the fingerprint check
would discard it regardless.

### Chunking strategy

Chunk size is 512 characters with Article-aware separators. Long Articles are
still split awkwardly. Not tested: larger chunks, or a header-aware splitter that
keeps an Article intact. Now cheap to evaluate with the deterministic harness.

### Golden dataset refusal rate

39% of questions produce refusals. Whether that reflects genuine gaps in the
retrieved context or an over-strict system prompt has not been separated. The
answered/refused split table makes it observable, but it has not been acted on.

### Before/after retrieval comparison

The improvement from the extraction fix is directionally clear but not cleanly
quantified. The pre-fix figure of 0.589 was measured by matching article strings
inside chunk text, while the post-fix 0.893 reads chunk metadata. Different
methods, so the pair is not a controlled A/B and should not be quoted as one.
The corruption measurement (38.7% to 0%) is the defensible number, since both
sides used the same document and the same scoring function.
