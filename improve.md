# Vector RAG / Graph RAG — Diagnosis & Improvement Report

Date: 2026-07-26

This report covers two reported problems — **(1) slow indexing** and **(2)
both Vector RAG and Graph RAG returning poor/incorrect answers** — with root
causes traced to specific files/lines, and concrete fixes ordered by
effort/impact.

Local setup found in `.env`: `OLLAMA_MODEL=gemma4:latest` (a 9.6 GB local
model, per `ollama list`), `OLLAMA_EMBED_MODEL=embeddinggemma:latest`. Several
issues below are specific to running a large local model through Ollama with
default settings.

---

## 1. Why indexing is slow

### 1.1 Graph indexing is one full LLM call per chunk, run sequentially
[`ingestion/graph_index.py:41-44`](ingestion/graph_index.py#L41-L44) sets
`EXTRACTION_WORKERS = 1` whenever `OLLAMA_MODEL` is set, and each chunk
(6000 chars, [`graph_index.py:31`](ingestion/graph_index.py#L31)) triggers a
full LLM completion in [`extract_chunk`](ingestion/graph_index.py#L137-L159).
With a single worker, a 30-page document (~10-15 chunks at that size) means
10-15 **sequential** round-trips through a 9.6 GB model on local hardware —
each call can easily take 10-60+ seconds depending on your GPU/CPU, so
ingestion time scales linearly and adds up fast. This is the dominant cost of
"indexing is slow" for the graph pipeline.

### 1.2 Embedding cost scales with chunk count and cold-start overhead
[`ingestion/vector_index.py:47`](ingestion/vector_index.py#L47) calls
`FAISS.from_documents(chunks, get_embeddings())`. `OllamaEmbeddings.embed_documents`
does send all chunk texts in a single batched request to Ollama's `/api/embed`
endpoint (not one HTTP call per chunk — verified against the installed
`langchain_ollama` source), so this isn't a network-overhead problem. The real
cost here is Ollama running inference over every chunk within that batch
sequentially on a single model instance, plus a cold-start model load if
`embeddinggemma:latest` isn't already resident in memory (idle beyond
`OLLAMA_KEEP_ALIVE`, default 5 minutes) — for a large PDF that's still
meaningful time, just not from per-chunk HTTP round-trips.

### 1.3 Vector index and graph index are built strictly one after another
In [`app.py:116-135`](app.py#L116-L135), `/api/upload` calls `build_index(...)`
and then `build_graph_index(...)` in sequence. These two pipelines don't
depend on each other's output — they both start from `pdf_path` — but today
total upload latency is `vector_time + graph_time` instead of
`max(vector_time, graph_time)`.

### 1.4 The PDF is parsed twice
`build_index` ([`vector_index.py:33`](ingestion/vector_index.py#L33)) and
`build_graph_index` ([`graph_index.py:119`](ingestion/graph_index.py#L119))
each call `PyPDFLoader(...).load()` independently on the same file. Minor
compared to 1.1-1.3, but free to fix.

### 1.5 FastAPI endpoints block the event loop
`/api/upload` and `/api/chat` in `app.py` are declared `async def`
([`app.py:101`](app.py#L101), [`app.py:159`](app.py#L159)) but everything
inside them — PDF parsing, embedding, LLM calls — is fully synchronous
blocking code with no `await`. FastAPI only gets its "run blocking work off
the event loop" behavior automatically for **plain `def`** endpoints (it
dispatches those to a worker thread pool); for `async def` endpoints, any
blocking call inside runs directly on the single asyncio event loop thread.
Practically this means: while one upload or chat request is running, the
**entire server is frozen** — even a cheap `GET /api/docs` from another tab
won't respond until the current request finishes. This makes the app feel
far slower than the actual work justifies, especially if you have the UI
open in two tabs or are testing while an upload is in flight.

---

## 2. Why responses aren't working properly

### 2.1 Graph RAG: fragile entity matching, no fallback
[`retrieval/graph_rag.py:99-126`](retrieval/graph_rag.py#L99-L126) extracts
entities from the *question* via the LLM, then matches them to graph node
strings using substring containment or `difflib.SequenceMatcher` (pure
character overlap) with `NODE_MATCH_THRESHOLD = 0.6`
([`graph_rag.py:29`](retrieval/graph_rag.py#L29)). This breaks down constantly:
- Paraphrased or partial entity names ("the developer's skills" vs. node
  `"Aashish Rana Magar"`) score low and never match.
- If **zero** query entities match a graph node, `matched_nodes` is empty,
  `_bfs_collect_triplets` returns `[]`, and the prompt context becomes
  `"(no relevant relationships found in the knowledge graph)"`
  ([`graph_rag.py:174-175`](retrieval/graph_rag.py#L174-L175)) — even when the
  graph clearly has relevant triplets. There's no fallback (e.g., keyword
  search across all triplet text, or embedding similarity over node labels).

### 2.2 Graph RAG: no entity resolution across chunks → fragmented graph
Each chunk's extraction call in `build_graph_index`
([`graph_index.py:137-159`](ingestion/graph_index.py#L137-L159)) runs
independently, so the same real-world entity can surface as different
strings in different chunks (case, abbreviation, or wording differences) —
the prompt asks the model to "keep entity names short and consistent" but
there is nothing enforcing consistency *across* independent LLM calls. The
result is a graph with more nodes than real entities (confirmed in the
existing sample: `data/graph_index/0f3049f1d6f83922/graph.json` has 41
triplets across only 11 subjects but 43 total unique nodes — for one single
resume-length page, most of that being one person's attributes, that's a
sign object nodes aren't being reused/normalized). A fragmented graph means
BFS traversal from a matched seed node misses facts that got attached to a
"duplicate" node instead.

### 2.3 Silent extraction failures
Both `extract_chunk` ([`graph_index.py:149-158`](ingestion/graph_index.py#L149-L158))
and `_extract_query_entities` ([`graph_rag.py:92-96`](retrieval/graph_rag.py#L92-L96))
swallow LLM/JSON-parsing failures and just return `[]`, logged only via
`logger.warning`. Local models (especially general chat models like
`gemma4:latest`, not instruction-tuned specifically for strict JSON output)
are more prone to malformed JSON than GPT-4o-mini. A chunk that fails to
parse silently contributes zero triplets — ingestion reports "success" with
a triplet count, but coverage may be quietly low with no visible signal.

### 2.4 Vector RAG: fixed top-3 chunks, uncalibrated relevance score
[`retrieval/vector_rag.py:20`](retrieval/vector_rag.py#L20) hardcodes
`TOP_K = 3`. For anything longer than a couple of pages, 3 chunks of 1000
characters each may not carry enough context to answer a question fully,
producing incomplete or "not enough information" answers even though the
document has the answer. Separately,
`similarity_search_with_relevance_scores` ([`vector_rag.py:65`](retrieval/vector_rag.py#L65))
relies on FAISS's default relevance-score function, which LangChain only
calibrates correctly when the index is L2-normalized — `FAISS.from_documents`
here doesn't set `normalize_L2=True` ([`vector_index.py:47`](ingestion/vector_index.py#L47)),
so the `score` field returned to the UI/report is not a reliable confidence
number (this matters more once you get to the thesis's RAGAS/LLM-as-judge
evaluation stage, since a broken relevance score undermines the
context-precision metric).

---

## 3. Recommended fixes, ordered by effort → impact

### Quick wins (config/parameter changes, no architecture change)
| Fix | Where | Effect |
|---|---|---|
| Bump `GRAPH_EXTRACTION_WORKERS` if your Ollama server can serve >1 request at once (`OLLAMA_NUM_PARALLEL`), otherwise switch to a smaller/faster local model for extraction (e.g. `qwen2.5-coder:7b` you already have pulled is much smaller than `gemma4:latest`'s 9.6 GB) | `.env` | Directly cuts graph indexing time |
| Raise `TOP_K` to 5-6, or make it a request parameter | `retrieval/vector_rag.py:20` | More complete context → fewer "not enough info" answers |
| Set `normalize_L2=True` when building the FAISS index (and re-embed with a normalized distance strategy) | `ingestion/vector_index.py:47` | Makes the relevance score meaningful |
| Lower `NODE_MATCH_THRESHOLD` cautiously or switch entity matching to embedding cosine similarity instead of `SequenceMatcher` | `retrieval/graph_rag.py:29,99-126` | Better recall on paraphrased entity mentions |

### Structural fixes (moderate effort, biggest correctness impact)
1. **Add a fallback path in Graph RAG when zero entities match.** If
   `matched_nodes` is empty, fall back to a keyword/substring search over all
   triplet subjects+objects+relations for terms from the question (or reuse
   the vector index to find nodes mentioned in the top vector-retrieved
   chunks). Right now a zero-match miss silently returns "no relevant
   relationships found" — a fallback would recover most of these cases.
2. **Add an entity-canonicalization pass after extraction, before persisting
   the graph.** After collecting `all_triplets` in `build_graph_index`
   ([`graph_index.py:194`](ingestion/graph_index.py#L194)), run a merge step:
   normalize case/whitespace, then cluster near-duplicate node labels (e.g.,
   via string similarity or a single LLM call listing all unique node labels
   and asking it to group aliases) and rewrite triplets to use one canonical
   label per cluster. This directly fixes the fragmentation in §2.2 and
   makes BFS traversal actually find connected facts.
3. **Run vector and graph indexing concurrently in `/api/upload`.** Since
   `build_index` and `build_graph_index` don't depend on each other, wrap both
   in a thread pool / `asyncio.gather(asyncio.to_thread(...), asyncio.to_thread(...))`
   in `app.py:116-135` so upload latency becomes `max()` instead of `sum()`.
4. **Stop blocking the event loop.** Either change `/api/upload` and
   `/api/chat` to plain `def` (FastAPI will run them in its worker thread
   pool automatically), or explicitly offload the blocking calls with
   `await asyncio.to_thread(build_index, ...)` etc. This alone should make the
   app *feel* dramatically more responsive under any concurrent use, even
   before touching indexing speed itself.
5. **Keep the embedding model warm.** Set `OLLAMA_KEEP_ALIVE` to a longer
   duration (or `-1` to keep it resident indefinitely) so `embeddinggemma:latest`
   doesn't need a cold model load on every upload after 5 minutes of
   inactivity — the embedding call is already batched in one request
   (§1.2), so cold-start avoidance is the main lever left here.

### Nice-to-have
- Share a single `PyPDFLoader(...).load()` result between `build_index` and
  `build_graph_index` (pass `pages` in instead of reloading) to remove the
  duplicate PDF parse in §1.4.
- Surface graph-extraction parse failures back through the `/api/upload`
  response (e.g. `failed_chunks: int`) instead of only logging them, so low
  coverage is visible instead of silent (§2.3).
- Since this is explicitly a **comparative thesis** between Vector RAG and
  Graph RAG, consider capturing per-query "matched_entities empty" and
  "chunks failed to parse" rates as part of the evaluation metrics already
  planned (RAGAS / LLM-as-judge) — these are exactly the failure modes that
  explain *why* one pipeline underperforms the other, which is valuable data
  for the thesis itself, not just a bug to silently fix.

---

## Summary

- **Indexing is slow** primarily because graph extraction runs one LLM call
  per chunk, sequentially, against a large (9.6 GB) local model, and because
  vector/graph indexing run one after another instead of concurrently — made
  worse by FastAPI endpoints blocking the whole server on every request.
- **Responses are unreliable** primarily because Graph RAG's entity matching
  is brittle character-level fuzzy matching with no fallback when it fails,
  compounded by a fragmented graph from missing entity canonicalization
  across chunks; Vector RAG's answers suffer from a small fixed `top_k` and
  an uncalibrated relevance score.
- Recommended order of attack: fix the event-loop blocking (#4) and add the
  graph-RAG fallback (#1) first — both are relatively small changes with an
  outsized effect on perceived speed and answer quality — then tackle entity
  canonicalization (#2) and indexing concurrency (#3).
