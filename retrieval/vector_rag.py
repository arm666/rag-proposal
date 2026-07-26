"""Vector RAG: Stage 2 (retrieval) + Stage 3 (generation) of the pipeline
described in the thesis proposal §3.3.2-3.3.3.

Retrieval: cosine similarity / ANN search over the FAISS index, top-k chunks.
Generation: same LLM backbone, temperature 0, context-only prompt — the model
must answer strictly from retrieved context, never from parametric memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from ingestion.vector_index import load_index
from retrieval.prompts import SYSTEM_PROMPT

TOP_K = 6


@dataclass
class Source:
    chunk_id: int
    page: int | None
    text: str
    score: float


@dataclass
class VectorRAGResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _build_prompt(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(
        f"[Chunk {s.chunk_id}{f', p.{s.page + 1}' if s.page is not None else ''}]\n{s.text}"
        for s in sources
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer_question(doc_id: str, question: str, top_k: int = TOP_K) -> VectorRAGResult:
    start = time.perf_counter()

    store = load_index(doc_id)
    results = store.similarity_search_with_relevance_scores(question, k=top_k)

    sources = [
        Source(
            chunk_id=doc.metadata.get("chunk_id", i),
            page=doc.metadata.get("page"),
            text=doc.page_content,
            score=score,
        )
        for i, (doc, score) in enumerate(results)
    ]

    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(question, sources)),
    ]

    prompt_tokens = completion_tokens = total_tokens = 0
    cost_usd = 0.0
    try:
        with get_openai_callback() as cb:
            response = llm.invoke(messages)
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            total_tokens = cb.total_tokens
            cost_usd = cb.total_cost
    except Exception:
        # LangChain's OpenAI callback only instruments OpenAI-compatible
        # calls; local/Gemini backends fall through without token/cost stats.
        response = llm.invoke(messages)

    latency = time.perf_counter() - start

    return VectorRAGResult(
        answer=response.content,
        sources=sources,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
