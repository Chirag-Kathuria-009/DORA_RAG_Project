# DORA RAG Application

A question-answering API over the EU Digital Operational Resilience Act
(Regulation EU 2022/2554). You ask a compliance question in plain English, it
retrieves the relevant passages from the regulation and answers with the Article
numbers it used.

This started as a learning project to build a RAG pipeline end to end without
skipping the boring parts: hybrid retrieval, reranking, tracing, and an actual
evaluation harness with numbers attached. The regulation was a good fit because
it is long, dense, heavily cross-referenced, and the answers are verifiable
against the source text.

## What it does

The single source document is the DORA regulation PDF (~723 chunks after
splitting). On a query:

```
question
   |
   +-- BM25 (top 35)          keyword match, catches exact Article refs
   +-- pgvector (top 35)      semantic match via bge-small-en-v1.5
   |
   v
EnsembleRetriever  (0.4 BM25 / 0.6 vector)
   |
   v
CrossEncoder rerank  (bge-reranker-base, keeps top 5)
   |
   v
Gemini + system prompt that forces Article citations
   |
   v
JSON: answer + the 5 source chunks + latency
```

The system prompt is deliberately strict. The model is told to answer only from
the retrieved text, cite the Article and paragraph for every claim, quote exact
figures for thresholds and deadlines, and say "not addressed in the provided
DORA documentation sections" rather than guess. Refusing is better than
inventing a compliance requirement.

## Stack

| Piece | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | 384-dim, runs on CPU, no API cost per chunk |
| Vector store | Postgres + pgvector | already know Postgres, and the metadata filtering is free |
| Keyword search | BM25 (`rank-bm25`) | regulation text is full of exact terms the embeddings blur |
| Reranker | `BAAI/bge-reranker-base` cross-encoder | started with Cohere rerank, hit the free-tier call limit, swapped to local |
| Generation | Gemini via `langchain-google-genai` | temperature 0 |
| Tracing | Langfuse | to see what context actually reached the model |
| Evaluation | RAGAS 0.2.6, judged by Groq-hosted models | judging with the same model that generates is a bad idea |

The Cohere reranker code is still in [src/retriever.py](src/retriever.py#L51-L55),
commented out, in case you have quota.

## Getting it running

### 1. Prerequisites

- Python 3.12
- Docker (for the Postgres/pgvector container)
- A Google AI Studio API key (generation) and a Groq API key (only needed if you
  run the evaluation)

### 2. Install

```bash
git clone <repo-url>
cd DORA_RAG_Application

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

First run downloads the embedding and reranker models from HuggingFace (a few
hundred MB), so give it a minute.

### 3. Environment

Create a `.env` in the project root:

```env
google_api_key=your_google_key
groq_api_key=your_groq_key
cohere_api_key=unused_but_required
database_url=postgresql+psycopg://dora:dorapass@localhost:5434/doradb

langfuse_public_key=pk-...
langfuse_secret_key=sk-...
langfuse_base_url=https://cloud.langfuse.com
```

All of these are required by the settings class in [src/config.py](src/config.py),
including `cohere_api_key` even though the Cohere reranker is currently off. Put
any non-empty string there if you don't have one. Same for the Langfuse keys if
you don't care about tracing.

### 4. Start the database

```bash
docker compose up -d db
```

That maps the container's 5432 to **5434** on the host, which is what the
`database_url` above expects. If you already have Postgres on 5434, change both.

### 5. Ingest the regulation

Drop the DORA PDF at `data/raw/DORA_regulation_EU_2022_2554.pdf`
(the `data/` folder is gitignored, so it isn't in the repo. Grab the PDF from
EUR-Lex), then:

```bash
python -m src.ingest
```

This loads the PDF, splits it into ~723 chunks, writes them into pgvector, and
saves a copy to `data/chunks_cache.json`. The cache exists because BM25 needs the
documents in memory at startup and re-parsing the PDF every boot is wasteful.
Run this once.

### 6. Start the API

```bash
uvicorn src.api:app --reload --port 8000
```

Swagger UI at http://localhost:8000/docs.

## Using it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Within what timeframe must a major ICT-related incident be reported?"}'
```

Response shape (content trimmed):

```json
{
  "answer": "Per Article 19(1), financial entities shall report major ICT-related incidents to the relevant competent authority ...",
  "sources": [
    {
      "source": "data/raw/DORA_regulation_EU_2022_2554.pdf",
      "page": "42",
      "content": "Article 19 Reporting of major ICT-related incidents ...",
      "relevance_rank": 1
    }
  ],
  "latency_ms": 3412.8,
  "model_used": "models/gemini-3.5-flash-lite"
}
```

`GET /health` returns a static status object, no dependency checks.

Set `"trace_id": false` in the request body to skip the Langfuse callback for
that call.

## Evaluation

There is a 56-question golden dataset in `data/eval/golden_dataset.json`, each
entry with a question, a ground-truth answer, and the Article it comes from.

```bash
python -m eval.run_eval
```

It runs the pipeline over the questions, then scores four RAGAS metrics. Each
metric gets its own judge model on Groq (see `JUDGE_BY_METRIC` in
[eval/run_eval.py](eval/run_eval.py#L42-L47)) partly to spread out rate limits
and partly so one weak judge doesn't skew everything.

**Be aware this run is slow on purpose.** There's a 20-second sleep between
questions and judge calls are sequential (`max_workers=1`), because free-tier
Groq will otherwise rate-limit you halfway through. Budget 15+ minutes.

Two things make it resumable:

- `data/eval/eval_checkpoint.json` stores predictions per question, so an
  interrupted run picks up where it stopped
- `data/eval/metric_cache.json` stores scores per metric, so a metric that
  already succeeded isn't re-judged

Delete either file to force that stage to run again.

### Current numbers

25 questions, generation by Gemini, judged by Llama 3.3 70B / GPT-OSS / Qwen.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.786 | did the answer stay inside the retrieved context |
| Answer relevancy | 0.720 | did it actually answer the question asked |
| Context precision | 0.668 | was what we retrieved mostly useful |
| Context recall | 0.646 | did we retrieve what was needed |

Reading these as pairs is more useful than reading them individually. Recall and
precision are both sitting in the mid-60s, which points at the retrieval stage
rather than generation: the chunks needed to answer often aren't making it into
the top 5 at all. Faithfulness holding up at 0.79 says the model is mostly
behaving itself with whatever context it does get.

Per-question scores land in `data/eval/per_question_results.csv`, which is where
the interesting failures show up. Several questions score 0.0 on faithfulness
and relevancy while scoring 1.0 on recall, and those are usually the strict
system prompt refusing to answer from context that technically contained the
answer.

Next things to try on retrieval: larger chunks with a header-aware splitter
(512 tokens cuts Articles in half), and query rewriting before retrieval.

## Layout

```
src/
  config.py       pydantic-settings, reads .env
  ingest.py       PDF -> chunks -> pgvector + BM25 cache
  retriever.py    hybrid retrieval + cross-encoder rerank
  chain.py        prompt + Gemini + LCEL wiring
  api.py          FastAPI endpoints
eval/
  run_eval.py     RAGAS harness with checkpointing
data/            all gitignored, see note below
  raw/            the source PDF
  eval/           golden dataset, results, caches
docker-compose.yml
```

## Known rough edges

Listing these because they're real, not because I plan to fix all of them.

- **No Dockerfile.** `docker-compose.yml` declares an `app` service with
  `build: .` but there's no Dockerfile to build. `docker compose up -d db` works;
  `docker compose up` does not. Run the API locally for now.
- **Retrieval runs twice per request.** [src/api.py](src/api.py#L45) calls the
  retriever a second time just to populate the `sources` field, after the chain
  already retrieved internally. It roughly doubles latency. The fix is to
  restructure the chain with `RunnableParallel` so both come out of one pass.
- **`cohere_api_key` is still mandatory** in the settings class despite being
  unused.
- **No test suite.** `test_models.py` is a scratch script for checking that the
  retriever returns chunks, not pytest.
- **`main.py` is the uv-generated placeholder** and does nothing.
- Chunking is character-based with Article-aware separators, which is better than
  nothing but still splits long Articles awkwardly.
- **The whole `data/` directory is gitignored**, which keeps the PDF and the
  caches out of the repo but also means a fresh clone has no
  `golden_dataset.json` to evaluate against. Worth un-ignoring
  `data/eval/golden_dataset.json` specifically.

## Notes

This is a personal learning project, not compliance advice. Answers come from a
language model reading a PDF, and the evaluation numbers above should tell you
how much to trust it.
