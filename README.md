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
| 1. Ingestion | Chunk + embed, stored in FAISS index | Entity/relationship extraction via LLM → graph (NetworkX/Neo4j), serialized as JSON triplets |
| 2. Retrieval | Top-3 chunks via cosine similarity / ANN | Extract query entities → match graph nodes → BFS traversal (2 hops, up to 20 triplets) |
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
| Graph store | NetworkX (dev) / Neo4j |
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
├── main.py            # entry point
├── config.py           # get_llm() — Ollama/OpenAI/Gemini backend selection
├── .env.example         # provider credentials, corpus path, etc.
├── retrieval/           # (planned) vector_rag.py, graph_rag.py
├── ingestion/           # (planned) chunking/embedding, entity/graph extraction
├── evaluation/          # (planned) RAGAS + LLM-as-a-judge + LangChain callbacks
└── data/                # (planned) corpus + query set + ground-truth labels
```

The `retrieval/`, `ingestion/`, `evaluation/`, and `data/` directories don't
exist yet — they're the planned home for the work in the roadmap below.

## Project status

Only the LLM backend selection (`config.py`) is implemented so far. The
retrieval pipelines, corpus ingestion, and evaluation harness described above
are upcoming work — see the roadmap.

## Roadmap

Mirrors the thesis Gantt chart (Apr–Oct 2026):

1. **Thesis planning & proposal submission** — done.
2. **System design & framework development** — repo scaffold, LLM backend
   selection, corpus/query set definition.
3. **Dataset preparation & implementation** — Vector RAG (FAISS ingestion +
   retrieval) and Graph RAG (entity extraction, graph construction, BFS
   traversal) pipelines.
4. **Experiment execution & data collection** — run both pipelines over the
   shared query set, log tokens/cost/latency via LangChain callbacks.
5. **Evaluation & result analysis** — RAGAS + LLM-as-a-judge scoring,
   cross-pipeline comparison.
6. **Thesis writing, review, revision & final submission**.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- One of: a local [Ollama](https://ollama.com) server + pulled model, an
  OpenAI API key, or a Gemini API key (see [LLM backend selection](#llm-backend-selection))

## Getting started

```bash
uv sync
cp .env.example .env   # set OLLAMA_MODEL, OPENAI_API_KEY, or GEMINI_API_KEY
uv run main.py
```
