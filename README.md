# RAG Proposal — Vector RAG vs Graph RAG

Implementation companion to the thesis proposal *"Comparative Analysis on
Vector-Based and Graph-Based Retrieval-Augmented Generation (RAG) Approaches
in Large Language Model Response"* (Mamata Jirel, Purbanchal University, 2026).

The project builds two parallel RAG pipelines — **Vector RAG** and **Graph
RAG** — over the same document corpus and LLM backbone, then evaluates both
with a shared metrics suite so any performance difference is attributable to
the retrieval mechanism, not the model or the data.

## Objectives

- Design and implement vector-based and graph-based retrieval on a common LLM
  backbone.
- Evaluate retrieval performance with standard information-retrieval metrics
  (retrieval accuracy, context precision, context recall).
- Assess generated-response quality with RAGAS, LLM-as-a-judge scoring, and
  LangChain callbacks (token usage, cost, latency).

## Architecture

```
Document Corpus & Query Set
            │
    Shared LLM Backbone (Ollama / OpenAI / Gemini — see below)
            │
   ┌────────┴────────┐
   │                 │
Vector RAG        Graph RAG
(embed, cosine    (entity extraction,
 similarity, ANN)  knowledge graph, BFS traversal)
   │                 │
Generated Output  Generated Output
   └────────┬────────┘
            │
        Evaluation
 (RAGAS, LLM-as-a-judge, LangChain callbacks)
            │
     Comparative Analysis → Report
```

### Pipeline stages

| Stage | Vector RAG | Graph RAG |
|---|---|---|
| 1. Ingestion | Chunk + embed, stored in FAISS index | LLM extracts entity-relationship triplets per chunk → NetworkX `MultiDiGraph`, persisted as JSON |
| 2. Retrieval | Top-3 chunks via cosine similarity / ANN | LLM extracts query entities → fuzzy-matched to graph nodes → BFS traversal (2 hops, up to 20 triplets) |
| 3. Generation | Same LLM, temperature 0, identical context-only prompt | Same LLM, temperature 0, identical context-only prompt |
| 4. Evaluation | RAGAS + LLM-as-a-judge + LangChain callbacks | RAGAS + LLM-as-a-judge + LangChain callbacks |

Both pipelines share the same corpus, query set, LLM backbone, and evaluation
framework so results are directly comparable.

## Corpus & query set

The evaluation corpus is domain-specific legal/regulatory text — e.g. official
Nepali statutory law, procurement regulations, or customs tariff schedules —
processed chapter-wise to preserve contextual hierarchy during indexing and
retrieval. A single corpus and a standardized query set are shared by both
pipelines so retrieval differences can't be explained by data differences.

The query set should include a labeled "relevant items" mapping per query
(ground truth), since Retrieval Accuracy, Context Precision, and Context
Recall are all defined relative to it. This mapping doesn't exist yet in the
repo — it needs to be authored alongside the corpus during the "Dataset
preparation" roadmap phase.

## Evaluation metrics

- **Efficiency**: token usage, cost (USD), end-to-end latency.
- **Quality**: answer quality, relevance, completeness (LLM-as-a-judge).
- **Retrieval**: retrieval accuracy, context precision, context recall.
- **Reliability**: hallucination rate, faithfulness (RAGAS).

## Tech stack

| Component | Tool |
|---|---|
| LLM framework | LangChain |
| LLM | Local Ollama model, GPT-4o mini (OpenAI), or Gemini — see [LLM backend selection](#llm-backend-selection) |
| Embeddings | Hugging Face sentence embeddings |
| Vector store | FAISS |
| Vector retrieval | Cosine similarity / Approximate Nearest Neighbor (ANN) |
| Graph store | NetworkX, persisted as JSON triplets (`data/graph_index/<doc_id>/graph.json`) |
| Graph retrieval | Breadth-First Search (BFS) traversal |
| Evaluation | RAGAS, LLM-as-a-judge, LangChain callbacks |
| Language | Python 3.11 |

## LLM backend selection

The LLM backend is chosen at runtime from `.env`, by provider priority:

1. **Ollama** — used when `OLLAMA_MODEL` is set (local model, no API key).
2. **OpenAI GPT** — used when `OPENAI_API_KEY` is set (and Ollama isn't).
3. **Gemini** — used when `GEMINI_API_KEY` is set (and neither of the above is).

Only the first configured provider is used; set just the credentials for the
one you want. This lets the pipelines run entirely offline against a local
model during development, and switch to GPT-4o mini (or Gemini) for the
experiments reported in the thesis, without touching code.

Copy `.env.example` to `.env` and fill in the values you need:

```bash
cp .env.example .env
```

See [config.py](config.py) for the selection logic (`get_llm()`).

## Project structure

```
rag-proposal/
├── main.py                  # CLI entry point (prints selected LLM backend)
├── app.py                   # FastAPI backend — upload + chat endpoints, serves static/
├── config.py                 # get_llm() / get_embeddings() — Ollama/OpenAI/Gemini backend selection
├── .env.example               # provider credentials
├── ingestion/
│   ├── vector_index.py          # Stage 1: PDF → chunks → embeddings → FAISS index (persisted)
│   └── graph_index.py            # Stage 1: PDF → LLM entity/relationship extraction → NetworkX graph (persisted as JSON)
├── retrieval/
│   ├── vector_rag.py             # Stage 2+3: top-k retrieval (cosine/ANN) → context-only generation
│   └── graph_rag.py               # Stage 2+3: query entity extraction → BFS traversal (2 hops) → context-only generation
├── static/
│   └── index.html                  # chat UI — upload a PDF, pick Vector/Graph/Both, see cited evidence
├── evaluation/                      # (planned) RAGAS + LLM-as-a-judge + LangChain callbacks as a batch harness
└── data/
    ├── uploads/                      # ingested PDFs (gitignored)
    ├── faiss_index/<doc_id>/          # one FAISS index per uploaded PDF, keyed by content hash (gitignored)
    └── graph_index/<doc_id>/graph.json # one knowledge graph per uploaded PDF, same content hash (gitignored)
```

The batch evaluation harness (RAGAS / LLM-as-a-judge scoring across a labeled
query set) is not implemented yet — see the roadmap.

## Vector RAG pipeline

Implements thesis §3.3.1–3.3.3 for the vector-based half of the comparison:

1. **Ingestion** (`ingestion/vector_index.py`) — `PyPDFLoader` extracts text
   page-by-page, `RecursiveCharacterTextSplitter` chunks it (1000 chars,
   150 overlap), each chunk is embedded and stored in a FAISS index persisted
   to `data/faiss_index/<doc_id>/`, where `doc_id` is a SHA-256 hash of the
   PDF bytes — re-uploading the same file reuses its existing index.
2. **Retrieval** (`retrieval/vector_rag.py`) — cosine similarity / ANN search
   over the FAISS index returns the top-3 chunks for a query.
3. **Generation** — the same LLM backbone (`get_llm()`), temperature 0, with
   a system prompt that forces context-only answers (no parametric memory).
4. **Instrumentation** — every answer reports latency, prompt/completion/total
   tokens, and cost in USD (via LangChain's OpenAI callback — token/cost
   figures are populated for the OpenAI backend; local/Gemini backends report
   latency only, since the callback only instruments OpenAI-compatible calls).

## Graph RAG pipeline

Implements thesis §3.3.1–3.3.3 for the graph-based half of the comparison:

1. **Ingestion** (`ingestion/graph_index.py`) — `PyPDFLoader` extracts text
   page-by-page, chunked (2000 chars, 200 overlap — larger than the vector
   chunks since entity extraction benefits from more surrounding context per
   LLM call). Each chunk is sent to the LLM backbone with a prompt that
   returns `{subject, relation, object}` triplets as JSON. Triplets are
   assembled into a NetworkX `MultiDiGraph` and persisted as
   `data/graph_index/<doc_id>/graph.json`, keyed by the same content-hash
   `doc_id` as the vector index.
2. **Retrieval** (`retrieval/graph_rag.py`) — the LLM extracts entities from
   the query, each is fuzzy-matched against graph node labels (exact match,
   substring, then `SequenceMatcher` similarity ≥ 0.6), and a breadth-first
   search from the matched nodes — traversing edges in both directions, up to
   2 hops — collects up to 20 relationship triplets.
3. **Generation** — identical to Vector RAG: same LLM backbone, temperature
   0, context-only system prompt. The retrieved triplets are serialized as
   `subject —[relation]→ object` lines and passed as context.
4. **Instrumentation** — same latency/token/cost reporting as Vector RAG, via
   LangChain's OpenAI callback.

## Web chat UI

`static/index.html`, served by `app.py`, is a single-page chat interface:

- Drag-and-drop or click to upload a PDF — it's chunked/embedded into a FAISS
  index *and* run through entity extraction into a knowledge graph on upload,
  so both pipelines are ready immediately.
- A **retrieval mode** dropdown selects **Vector RAG**, **Graph RAG**, or
  **Both (compare)** before asking a question.
- Ask questions in the composer:
  - Vector/Graph mode shows one answer with its evidence (source chunks with
    similarity scores, or matched entities + relationship triplets) and
    latency/token/cost metrics.
  - Both mode runs both pipelines against the same question and renders them
    side by side for direct comparison.

### Endpoints (`app.py`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload a PDF, build/reuse its FAISS + graph indexes |
| `POST` | `/api/chat` | `{doc_id, question, mode}` → answer(s) + evidence + metrics (`mode`: `vector` \| `graph` \| `both`) |
| `GET` | `/api/docs` | List indexed `doc_id`s |
| `GET` | `/` | Chat UI |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- One of: a local [Ollama](https://ollama.com) server + pulled chat model
  (embeddings default to `embeddinggemma:latest`, also pulled locally), an
  OpenAI API key (embeddings default to `text-embedding-3-small`), or a
  Gemini API key — see [LLM backend selection](#llm-backend-selection)

## Getting started

```bash
uv sync
cp .env.example .env   # set OLLAMA_MODEL, OPENAI_API_KEY, or GEMINI_API_KEY
uv run uvicorn app:app --reload
```

Then open http://127.0.0.1:8000, upload a PDF, and start asking questions.

## Roadmap

Mirrors the thesis Gantt chart (Apr–Oct 2026):

1. **Thesis planning & proposal submission** — done.
2. **System design & framework development** — repo scaffold, LLM backend
   selection, corpus/query set definition — done.
3. **Dataset preparation & implementation**:
   - Vector RAG (FAISS ingestion + retrieval) + chat UI — done.
   - Graph RAG (entity extraction, graph construction, BFS traversal) —
     done. Chat UI has a Vector/Graph/Both mode selector.
4. **Experiment execution & data collection** — run both pipelines over a
   shared, labeled query set (ground-truth relevant items per query, needed
   for Retrieval Accuracy / Context Precision / Context Recall) — not started.
5. **Evaluation & result analysis** — RAGAS + LLM-as-a-judge batch scoring,
   cross-pipeline comparison — not started.
6. **Thesis writing, review, revision & final submission**.
