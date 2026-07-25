"""Graph RAG: Stage 2 (retrieval) + Stage 3 (generation) of the pipeline
described in the thesis proposal §3.3.2-3.3.3.

Retrieval: extract query entities via the LLM, fuzzy-match them to graph
nodes, then run BFS traversal up to 2 hops to collect up to 20 relationship
triplets. Generation: same LLM backbone, temperature 0, context-only prompt —
identical constraint to the Vector RAG pipeline so output quality depends on
retrieval, not model knowledge.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import networkx as nx
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from ingestion.graph_index import load_graph

MAX_HOPS = 2
MAX_TRIPLETS = 20
NODE_MATCH_THRESHOLD = 0.6

ENTITY_EXTRACTION_PROMPT = (
    "Extract the key entities (people, places, organizations, concepts, "
    "terms) mentioned in this question. Return ONLY a JSON array of strings, "
    "no prose, no markdown fences.\n\nQuestion: {question}"
)

SYSTEM_PROMPT = (
    "You're a helpful assistant. Write naturally, like you're explaining "
    "something to someone directly — no robotic phrasing, no unnecessary "
    "hedging or filler, and don't preface your answer with phrases like "
    "\"based on the context\" or \"according to the provided information\" — "
    "just answer the question. Stick to what's actually in the context and "
    "don't bring in outside knowledge. If the context doesn't have enough "
    "to answer the question, just say that plainly instead of guessing or "
    "padding the answer."
)


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str
    page: int | None


@dataclass
class GraphRAGResult:
    answer: str
    triplets: list[Triplet] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _parse_string_list(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return [str(x).strip() for x in data if isinstance(data, list) and str(x).strip()]


def _extract_query_entities(llm, question: str) -> list[str]:
    messages = [
        SystemMessage(
            content="You are a precise information-extraction system that outputs only JSON."
        ),
        HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(question=question)),
    ]
    try:
        response = llm.invoke(messages)
    except Exception:
        return []
    return _parse_string_list(response.content)


def _match_nodes(graph: nx.MultiDiGraph, entities: list[str]) -> list[str]:
    """Fuzzy-match extracted entity strings to actual graph node labels."""
    nodes = list(graph.nodes)
    matched = []
    for entity in entities:
        best_node, best_score = None, 0.0
        entity_lower = entity.lower()
        for node in nodes:
            node_lower = node.lower()
            if entity_lower == node_lower:
                best_node, best_score = node, 1.0
                break
            if entity_lower in node_lower or node_lower in entity_lower:
                score = 0.85
            else:
                score = SequenceMatcher(None, entity_lower, node_lower).ratio()
            if score > best_score:
                best_node, best_score = node, score
        if best_node is not None and best_score >= NODE_MATCH_THRESHOLD:
            matched.append(best_node)
    # de-dupe, preserve order
    seen = set()
    unique = []
    for node in matched:
        if node not in seen:
            seen.add(node)
            unique.append(node)
    return unique


def _bfs_collect_triplets(
    graph: nx.MultiDiGraph, seed_nodes: list[str], max_hops: int, max_triplets: int
) -> list[Triplet]:
    """Breadth-first traversal from seed nodes (both edge directions), up to
    max_hops away, collecting relationship triplets until max_triplets."""
    visited_edges: set[tuple] = set()
    triplets: list[Triplet] = []
    visited_nodes = set(seed_nodes)
    queue = deque((node, 0) for node in seed_nodes)

    while queue and len(triplets) < max_triplets:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue

        neighbors = []
        for _, target, data in graph.out_edges(node, data=True):
            neighbors.append((node, target, data, "out"))
        for source, _, data in graph.in_edges(node, data=True):
            neighbors.append((source, node, data, "in"))

        for subj, obj, data, _direction in neighbors:
            edge_key = (subj, data.get("relation"), obj)
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                triplets.append(
                    Triplet(
                        subject=subj,
                        relation=data.get("relation", "related_to"),
                        object=obj,
                        page=data.get("page"),
                    )
                )
                if len(triplets) >= max_triplets:
                    break

            next_node = obj if subj == node else subj
            if next_node not in visited_nodes:
                visited_nodes.add(next_node)
                queue.append((next_node, depth + 1))

    return triplets[:max_triplets]


def _build_prompt(question: str, triplets: list[Triplet]) -> str:
    if not triplets:
        context = "(no relevant relationships found in the knowledge graph)"
    else:
        context = "\n".join(
            f"- {t.subject} —[{t.relation}]→ {t.object}"
            + (f" (p.{t.page + 1})" if t.page is not None else "")
            for t in triplets
        )
    return f"Context (knowledge graph relationships):\n{context}\n\nQuestion: {question}"


def answer_question(
    doc_id: str,
    question: str,
    max_hops: int = MAX_HOPS,
    max_triplets: int = MAX_TRIPLETS,
) -> GraphRAGResult:
    start = time.perf_counter()

    graph = load_graph(doc_id)
    llm = get_llm(temperature=0)

    entities = _extract_query_entities(llm, question)
    matched_nodes = _match_nodes(graph, entities)
    triplets = _bfs_collect_triplets(graph, matched_nodes, max_hops, max_triplets)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(question, triplets)),
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
        response = llm.invoke(messages)

    latency = time.perf_counter() - start

    return GraphRAGResult(
        answer=response.content,
        triplets=triplets,
        matched_entities=matched_nodes,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
