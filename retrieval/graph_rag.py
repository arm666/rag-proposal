"""Graph RAG: Stage 2 (retrieval) + Stage 3 (generation) of the pipeline
described in the thesis proposal §3.3.2-3.3.3.

Retrieval: extract query entities via the LLM, fuzzy-match them to graph
nodes, then traverse up to 2 hops to collect up to 20 relationship triplets.
The graph itself lives in Neo4j when NEO4J_URI is configured (traversal runs
as a Cypher query), otherwise in an in-memory NetworkX graph loaded from
data/graph_index/<doc_id>/graph.json (traversal is a Python BFS) — see
config.neo4j_enabled(). Generation: same LLM backbone, temperature 0,
context-only prompt — identical constraint to the Vector RAG pipeline so
output quality depends on retrieval, not model knowledge.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import networkx as nx
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_embeddings, get_llm, get_neo4j_driver, neo4j_database, neo4j_enabled
from ingestion.graph_index import EXTRACTION_SYSTEM_PROMPT, load_graph
from retrieval.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_HOPS = 2
MAX_TRIPLETS = 20
NODE_MATCH_THRESHOLD = 0.6
# Below this, embedding cosine similarity is treated as "no real match" —
# lower than NODE_MATCH_THRESHOLD because embedding similarity between
# unrelated short strings tends to sit higher than character-overlap
# similarity does, so the same cutoff would over-match.
EMBEDDING_MATCH_THRESHOLD = 0.55
# Question words shorter than this are too generic (stopword-ish) to use as
# a keyword-search fallback signal.
MIN_FALLBACK_KEYWORD_LEN = 4

ENTITY_EXTRACTION_PROMPT = (
    "Extract the key entities (people, places, organizations, concepts, "
    "terms) mentioned in this question. Return ONLY a JSON array of strings, "
    "no prose, no markdown fences.\n\nQuestion: {question}"
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
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=ENTITY_EXTRACTION_PROMPT.format(question=question)),
    ]
    try:
        response = llm.invoke(messages)
    except Exception:
        return []
    return _parse_string_list(response.content)


def _char_match_score(entity_lower: str, node_lower: str) -> float:
    if entity_lower == node_lower:
        return 1.0
    if entity_lower in node_lower or node_lower in entity_lower:
        return 0.85
    return SequenceMatcher(None, entity_lower, node_lower).ratio()


def _normalized_embeddings(texts: list[str]) -> np.ndarray | None:
    try:
        vectors = np.array(get_embeddings().embed_documents(texts), dtype=np.float32)
    except Exception:
        logger.warning("graph_rag: embedding-based node matching unavailable", exc_info=True)
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# doc_id -> (node labels as of last embed, their normalized vectors). Node
# labels only change when the graph is rebuilt, so this avoids re-embedding
# the entire node vocabulary on every single query — previously the biggest
# hidden latency/cost source in Graph RAG retrieval, scaling with graph size.
_node_embedding_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}


def _cached_node_embeddings(doc_id: str, nodes: list[str]) -> np.ndarray | None:
    key = tuple(nodes)
    cached = _node_embedding_cache.get(doc_id)
    if cached is not None and cached[0] == key:
        return cached[1]
    vectors = _normalized_embeddings(nodes)
    if vectors is not None:
        _node_embedding_cache[doc_id] = (key, vectors)
    return vectors


def _embedding_similarity_matrix(doc_id: str, entities: list[str], nodes: list[str]) -> np.ndarray | None:
    """Cosine similarity between each entity and each node label, via the
    configured embeddings backend. Returns None (falling back to
    character-only matching) if embedding fails for any reason."""
    node_vecs = _cached_node_embeddings(doc_id, nodes)
    if node_vecs is None:
        return None
    entity_vecs = _normalized_embeddings(entities)
    if entity_vecs is None:
        return None
    return entity_vecs @ node_vecs.T


def _match_nodes(doc_id: str, nodes: list[str], entities: list[str]) -> list[str]:
    """Fuzzy-match extracted entity strings to actual graph node labels.

    Combines character-level overlap (good at catching casing/typo
    variants) with embedding cosine similarity (good at catching
    paraphrases and partial names that share no characters in common, e.g.
    "the developer" vs. a node like "Aashish Rana Magar") — pure
    character matching alone missed most non-verbatim mentions.

    Takes a plain node-label list rather than a graph object so both the
    NetworkX and Neo4j backends can share this exact matching logic.
    """
    if not nodes or not entities:
        return []

    embed_sim = _embedding_similarity_matrix(doc_id, entities, nodes)

    matched = []
    for i, entity in enumerate(entities):
        entity_lower = entity.lower()
        best_node, best_score = None, -1.0
        for j, node in enumerate(nodes):
            char_score = _char_match_score(entity_lower, node.lower())
            embed_score = float(embed_sim[i, j]) if embed_sim is not None else 0.0
            # Only candidates that clear one of the two thresholds are
            # considered at all — previously the single highest-`combined`
            # node across *all* candidates was picked first and only then
            # checked against the threshold, so a node that narrowly failed
            # both thresholds could shadow a different, lower-scoring node
            # that actually qualified, silently dropping a valid match.
            if char_score < NODE_MATCH_THRESHOLD and embed_score < EMBEDDING_MATCH_THRESHOLD:
                continue
            combined = max(char_score, embed_score)
            if combined > best_score:
                best_node, best_score = node, combined
        if best_node is not None:
            matched.append(best_node)
    # de-dupe, preserve order
    seen = set()
    unique = []
    for node in matched:
        if node not in seen:
            seen.add(node)
            unique.append(node)
    return unique


def _keyword_fallback_triplets(
    graph: nx.MultiDiGraph, question: str, max_triplets: int
) -> list[Triplet]:
    """When entity-based matching finds nothing to traverse from, fall back
    to a plain keyword search over every triplet's subject/relation/object
    text. Recovers cases where the LLM's extracted query entities don't map
    onto any graph node at all — e.g. a question that doesn't name an
    entity explicitly — instead of silently returning an empty context even
    though the graph has relevant relationships.
    """
    keywords = {
        w for w in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(w) >= MIN_FALLBACK_KEYWORD_LEN
    }
    if not keywords:
        return []

    triplets: list[Triplet] = []
    seen_edges: set[tuple] = set()
    for subj, obj, data in graph.edges(data=True):
        relation = data.get("relation", "related_to")
        haystack = f"{subj} {relation} {obj}".lower()
        if not any(kw in haystack for kw in keywords):
            continue
        edge_key = (subj, relation, obj)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        triplets.append(Triplet(subject=subj, relation=relation, object=obj, page=data.get("page")))
        if len(triplets) >= max_triplets:
            break
    return triplets


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


# --- Neo4j-backed retrieval -------------------------------------------------
# When NEO4J_URI is configured (see config.neo4j_enabled), the graph lives in
# Neo4j instead of an in-memory NetworkX graph loaded from JSON. Entity
# matching (_match_nodes above) is identical either way — only where the
# candidate node list comes from, and how traversal/keyword search run,
# differ. Traversal and keyword search are pushed down into Cypher instead
# of walking edges in a Python loop, which is the actual point of using a
# real graph database instead of NetworkX+JSON.


def _neo4j_node_names(doc_id: str) -> list[str]:
    driver = get_neo4j_driver()
    with driver.session(database=neo4j_database()) as session:
        result = session.run("MATCH (n:Entity {docId: $doc_id}) RETURN n.name AS name", doc_id=doc_id)
        return [record["name"] for record in result]


def _neo4j_bfs_collect_triplets(
    doc_id: str, seed_nodes: list[str], max_hops: int, max_triplets: int
) -> list[Triplet]:
    """Cypher equivalent of _bfs_collect_triplets: variable-length path
    traversal (both directions) from the seed nodes, up to max_hops away.

    Neo4j doesn't allow parameterizing a variable-length relationship
    pattern's hop bound (`*1..N`) — it must be a literal in the query text.
    max_hops is always one of this module's own int constants/defaults
    (never user input), so interpolating it is safe; doc_id and seed_nodes
    stay proper query parameters.
    """
    max_hops = int(max_hops)
    driver = get_neo4j_driver()
    # ORDER BY the shortest path each relationship appears on before LIMIT —
    # without it, Neo4j's path-enumeration order (not hop-distance) decides
    # which edges survive truncation, unlike the NetworkX BFS below, which
    # is a real breadth-first walk and always fills the triplet budget with
    # the closest relationships first. Keeps the two backends' retrieval
    # behavior comparable, which matters for a thesis comparing them.
    query = f"""
        UNWIND $seeds AS seed
        MATCH (s:Entity {{docId: $doc_id, name: seed}})
        MATCH path = (s)-[:RELATION*1..{max_hops}]-(:Entity {{docId: $doc_id}})
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS a, rel, endNode(rel) AS b, min(length(path)) AS hop_distance
        RETURN DISTINCT a.name AS subject, rel.type AS relation, b.name AS object, rel.page AS page, hop_distance
        ORDER BY hop_distance
        LIMIT $max_triplets
    """
    with driver.session(database=neo4j_database()) as session:
        result = session.run(query, doc_id=doc_id, seeds=seed_nodes, max_triplets=max_triplets)
        return [
            Triplet(subject=r["subject"], relation=r["relation"] or "related_to", object=r["object"], page=r["page"])
            for r in result
        ]


def _neo4j_keyword_fallback_triplets(doc_id: str, question: str, max_triplets: int) -> list[Triplet]:
    """Cypher equivalent of _keyword_fallback_triplets."""
    keywords = [
        w for w in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(w) >= MIN_FALLBACK_KEYWORD_LEN
    ]
    if not keywords:
        return []

    driver = get_neo4j_driver()
    query = """
        MATCH (a:Entity {docId: $doc_id})-[r:RELATION]->(b:Entity {docId: $doc_id})
        WHERE any(kw IN $keywords WHERE
            toLower(a.name) CONTAINS kw OR toLower(b.name) CONTAINS kw OR toLower(r.type) CONTAINS kw
        )
        RETURN DISTINCT a.name AS subject, r.type AS relation, b.name AS object, r.page AS page
        LIMIT $max_triplets
    """
    with driver.session(database=neo4j_database()) as session:
        result = session.run(query, doc_id=doc_id, keywords=keywords, max_triplets=max_triplets)
        return [
            Triplet(subject=r["subject"], relation=r["relation"] or "related_to", object=r["object"], page=r["page"])
            for r in result
        ]


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

    llm = get_llm(temperature=0)
    entities = _extract_query_entities(llm, question)

    if neo4j_enabled():
        nodes = _neo4j_node_names(doc_id)
        matched_nodes = _match_nodes(doc_id, nodes, entities)
        triplets = (
            _neo4j_bfs_collect_triplets(doc_id, matched_nodes, max_hops, max_triplets)
            if matched_nodes
            else []
        )
        if not triplets:
            triplets = _neo4j_keyword_fallback_triplets(doc_id, question, max_triplets)
            if triplets and not matched_nodes:
                matched_nodes = sorted({t.subject for t in triplets} | {t.object for t in triplets})
    else:
        graph = load_graph(doc_id)
        matched_nodes = _match_nodes(doc_id, list(graph.nodes), entities)
        triplets = _bfs_collect_triplets(graph, matched_nodes, max_hops, max_triplets) if matched_nodes else []
        if not triplets:
            triplets = _keyword_fallback_triplets(graph, question, max_triplets)
            if triplets and not matched_nodes:
                matched_nodes = sorted({t.subject for t in triplets} | {t.object for t in triplets})

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(question, triplets)),
    ]
    response = llm.invoke(messages)

    latency = time.perf_counter() - start

    return GraphRAGResult(
        answer=response.content,
        triplets=triplets,
        matched_entities=matched_nodes,
        latency_seconds=latency,
    )
