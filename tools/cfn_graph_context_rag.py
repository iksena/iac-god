# tools/cfn_graph_context_rag.py
"""
GraphRAG context tool — LogSage-inspired multi-route retrieval.

Retrieval architecture (mirrors LogSage Stage 2):
  Route A — Symbolic / exact:  regex extraction of AWS::X::Y + /PropName
             from error strings → direct graph node lookup  (fast, zero-cost)
  Route B — Sparse lexical:    BM25 over property-spec text corpus
             (handles paraphrased deploy errors like "encryption at rest")
  Route C — Dense semantic:    FAISS cosine-KNN over sentence embeddings
             (handles novel error phrasings not seen in Route A/B)

Results from all three routes are merged, deduplicated, and reranked by a
simple reciprocal-rank-fusion (RRF) score before being rendered into a
Markdown context block for the remediator prompt.

Offline prerequisite (run once via scripts/build_cfn_rag_index.py):
    data/cfn_rag_corpus.jsonl   — one doc per property/resource node
    data/cfn_rag_faiss.index    — FAISS FlatIP index
    data/cfn_rag_bm25.pkl       — BM25Okapi index + id list
    data/cfn_graph.pkl          — existing networkx DiGraph
"""
from __future__ import annotations

import json
import pickle
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np

# ── Optional heavy deps — degrade gracefully ─────────────────────────────────
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_OK = True
except ImportError:
    _ST_OK = False

# ── Paths ────────────────────────────────────────────────────────────────────
_DATA = Path(__file__).resolve().parents[1] / "data"
_GRAPH_PATH  = _DATA / "cfn_graph.pkl"
_CORPUS_PATH = _DATA / "cfn_rag_corpus.jsonl"
_FAISS_PATH  = _DATA / "cfn_rag_faiss.index"
_BM25_PATH   = _DATA / "cfn_rag_bm25.pkl"

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 22 MB, very fast

# ── Regex (same as cfn_graph_context.py — kept local for independence) ───────
_CFN_LINT_PROP_RE = re.compile(
    r"(AWS::[A-Za-z0-9]+::[A-Za-z0-9]+)/([A-Za-z0-9]+)"
)
_RESOURCE_TYPE_RE = re.compile(r"AWS::[A-Za-z0-9]+::[A-Za-z0-9]+(?!\.[A-Za-z])")

# RRF constant (standard value from literature)
_RRF_K = 60

_SYNTACTIC_STAGES = {"yaml", "json", "comments"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class CorpusDoc(NamedTuple):
    """One entry in cfn_rag_corpus.jsonl."""
    doc_id: str           # "AWS::S3::Bucket"  or  "AWS::S3::Bucket/BucketEncryption"
    resource_type: str    # always "AWS::S3::Bucket"
    property_name: str    # "" for resource-level docs
    text: str             # natural-language description used for embedding


class RetrievedNode(NamedTuple):
    resource_type: str
    property_name: str   # "" → resource-level hit
    rrf_score: float
    sources: frozenset[str]   # which routes contributed: {"symbolic","bm25","faiss"}
    pinned: bool              # True if directly named in an error


# ─────────────────────────────────────────────────────────────────────────────
# Lazy loaders — one-time per process
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph | None:
    if not _GRAPH_PATH.exists():
        return None
    with _GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj


@lru_cache(maxsize=1)
def _load_corpus() -> list[CorpusDoc]:
    if not _CORPUS_PATH.exists():
        return []
    docs = []
    with _CORPUS_PATH.open() as fh:
        for line in fh:
            d = json.loads(line)
            docs.append(CorpusDoc(**d))
    return docs


@lru_cache(maxsize=1)
def _load_faiss():
    """Returns (index, id_list) or (None, [])."""
    if not _FAISS_OK or not _FAISS_PATH.exists():
        return None, []
    index = faiss.read_index(str(_FAISS_PATH))
    corpus = _load_corpus()
    return index, [d.doc_id for d in corpus]


@lru_cache(maxsize=1)
def _load_bm25():
    """Returns (BM25Okapi, id_list) or (None, [])."""
    if not _BM25_OK or not _BM25_PATH.exists():
        return None, []
    with _BM25_PATH.open("rb") as fh:
        bm25_obj, id_list = pickle.load(fh)
    return bm25_obj, id_list


@lru_cache(maxsize=1)
def _load_embedder():
    if not _ST_OK:
        return None
    return SentenceTransformer(_EMBED_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Query construction  (LogSage: "hybrid query = root cause + log snippet")
# ─────────────────────────────────────────────────────────────────────────────

def _build_queries(
        validation_results: list[dict],
        deploy_validation_result: dict | None,
        *,
        exclude_stages: set[str] | None = None,
    ) -> list[str]:
    """
    Produce one query string per distinct error.
    Each query combines the raw error text with any structured fields
    (rule code, resource type) found — mirroring LogSage's hybrid query
    construction from RCA report + critical log block.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            queries.append(s)

    for result in validation_results:
        stage = result.get("stage", "")
        if stage in exclude_stages:
            continue 
        for error in result.get("errors", []):
            error_str = str(error)
            # Prefix the stage so BM25/FAISS can weight it
            _add(f"[{stage}] {error_str}")

    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("target") != "skipped":
            if deploy_validation_result.get("error_message"):
                _add(f"[deploy] {deploy_validation_result['error_message']}")
            for fr in deploy_validation_result.get("failed_resources", []):
                for val in fr.values():
                    if val:
                        _add(f"[deploy] {val}")

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Route A — Symbolic (regex, zero-cost)
# ─────────────────────────────────────────────────────────────────────────────

def _symbolic_retrieve(
    queries: list[str],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Returns:
        ranked:         {doc_id: rank_position}  (1-based, lower = better)
        property_hints: {resource_type: [prop_name, ...]}
    """
    ranked: dict[str, int] = {}
    property_hints: dict[str, list[str]] = {}
    position = 1

    for q in queries:
        # Typed paths first (most specific)
        for rtype, prop in _CFN_LINT_PROP_RE.findall(q):
            doc_id = f"{rtype}/{prop}"
            if doc_id not in ranked:
                ranked[doc_id] = position
                position += 1
            hints = property_hints.setdefault(rtype, [])
            if prop not in hints:
                hints.append(prop)
            # Also include the resource-level doc
            if rtype not in ranked:
                ranked[rtype] = position
                position += 1

        # Bare resource types
        for rtype in _RESOURCE_TYPE_RE.findall(q):
            if rtype not in ranked:
                ranked[rtype] = position
                position += 1

    return ranked, property_hints


# ─────────────────────────────────────────────────────────────────────────────
# Route B — BM25 (sparse lexical)
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    """
    Tokenise each query, score against corpus, return combined ranking.
    Multiple queries → scores are max-pooled per doc before ranking.
    """
    bm25, id_list = _load_bm25()
    if bm25 is None or not id_list:
        return {}

    # Accumulate best BM25 score per doc across all queries
    best_scores: dict[str, float] = {}
    for q in queries:
        tokens = q.lower().split()
        scores = bm25.get_scores(tokens)
        for idx, score in enumerate(scores):
            doc_id = id_list[idx]
            if score > best_scores.get(doc_id, 0.0):
                best_scores[doc_id] = score

    # Rank by score descending, keep top_k
    sorted_docs = sorted(best_scores, key=lambda d: best_scores[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}


# ─────────────────────────────────────────────────────────────────────────────
# Route C — FAISS dense semantic (KNN)
# ─────────────────────────────────────────────────────────────────────────────

def _faiss_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    """
    Embed each query, search FAISS, max-pool similarities per doc, rank.
    """
    index, id_list = _load_faiss()
    embedder = _load_embedder()
    if index is None or embedder is None or not id_list:
        return {}

    # Embed all queries in one batch
    q_vecs = embedder.encode(queries, normalize_embeddings=True).astype("float32")

    best_scores: dict[str, float] = {}
    for q_vec in q_vecs:
        distances, indices = index.search(q_vec[None, :], top_k * 2)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = id_list[idx]
            if float(dist) > best_scores.get(doc_id, -1.0):
                best_scores[doc_id] = float(dist)

    sorted_docs = sorted(best_scores, key=lambda d: best_scores[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(
    *ranked_lists: dict[str, int],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """
    Merge multiple {doc_id: rank} dicts via RRF.
    Returns [(doc_id, rrf_score)] sorted best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Context renderer (same as before — graph lookup drives the Markdown)
# ─────────────────────────────────────────────────────────────────────────────

def _props_for_resource(G: nx.DiGraph, rtype: str) -> tuple[list[dict], list[dict]]:
    props, ptypes = [], []
    for _, nbr in G.out_edges(rtype):
        nd = G.nodes[nbr]
        if nd.get("ntype") == "Property":
            props.append(nd)
        elif nd.get("ntype") == "PropertyType":
            ptypes.append(nd)
    return props, ptypes


def _render_block(
    G: nx.DiGraph,
    rtype: str,
    pinned_props: list[str],
    sources: frozenset[str],
    rrf_score: float,
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    if rtype not in G:
        return (
            f"### {rtype}\n"
            f"*(not found in CFN spec — may be an invalid resource type)*\n"
            f"*Retrieved via: {', '.join(sorted(sources))} | RRF score: {rrf_score:.4f}*"
        )
    
    node_data = G.nodes[rtype]
    if node_data.get("ntype") not in ("Resource", None):
        return ""

    props, ptypes = _props_for_resource(G, rtype)
    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]
    pinned_set = set(pinned_props)
    lines = [f"### {rtype}"]
    lines.append(
        f"*Retrieved via: {', '.join(sorted(sources))} | RRF score: {rrf_score:.4f}*"
    )

    # Pinned properties (from errors / Route A)
    if pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in pinned_props:
            match = next((p for p in props if p.get("name") == name), None)
            if match:
                prim = match.get("primitive_type") or match.get("type") or "Any"
                req  = "**required**" if match.get("required") else "optional"
                upd  = match.get("update_type", "")
                lines.append(
                    f"  - `{name}` ({prim}, {req}"
                    + (f" — UpdateType: {upd}" if upd else "") + ")"
                )
            else:
                lines.append(f"  - `{name}` *(not in spec — invalid property)*")

    # Required remainder
    req_rest = [p for p in required if p.get("name") not in pinned_set]
    if req_rest:
        req_parts = [
            f"`{p.get('name','?')}` "
            f"({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)"
            for p in req_rest
        ]
        lines.append("**Other required properties:** " + ", ".join(req_parts))

    # Optional sample
    opt_rest = [p for p in optional if p.get("name") not in pinned_set]
    if opt_rest:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_rest[:max_optional]:
            name = p.get("name", "?")
            prim = p.get("primitive_type") or p.get("type") or "Any"
            upd  = p.get("update_type", "")
            lines.append(
                f"  - `{name}` ({prim}"
                + (f" — UpdateType: {upd}" if upd else "") + ")"
            )
        leftover = len(opt_rest) - max_optional
        if leftover > 0:
            lines.append(f"  - … and {leftover} more")

    # Nested types
    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested property types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def get_cfn_graph_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None = None,
    *,
    top_k_bm25: int = 8,
    top_k_faiss: int = 8,
) -> str:
    """
    LogSage-inspired multi-route GraphRAG context for the remediator.

    Route A (symbolic) always runs — zero cost, highest precision.
    Routes B (BM25) and C (FAISS) fill gaps when errors are paraphrased
    or don't contain explicit AWS::X::Y patterns (e.g. deploy timeouts,
    cyclic dependency errors, ambiguous property-value errors).

    All three route rankings are merged via Reciprocal Rank Fusion and
    reranked before rendering — so the most relevant resource types rise
    to the top regardless of which route found them.

    Args:
        validation_results:       state["validation_results"]
        deploy_validation_result: state.get("deploy_validation_result")
        top_k_bm25:               candidates to keep from BM25 route
        top_k_faiss:              candidates to keep from FAISS route

    Returns Markdown string ready for remediator prompt injection.
    """
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    # 1. Build queries (LogSage: hybrid query = error context + log snippet)
    queries = _build_queries(validation_results, deploy_validation_result, exclude_stages=_SYNTACTIC_STAGES)
    if not queries:
        return "No actionable errors found in validation results."

    # 2. Multi-route retrieval
    _MIN_RRF_SCORE = 0.03
    sym_ranked, property_hints = _symbolic_retrieve(queries)
    bm25_ranked  = _bm25_retrieve(queries, top_k=top_k_bm25)
    faiss_ranked = _faiss_retrieve(queries, top_k=top_k_faiss)

    # 3. RRF merge
    fused = _rrf_merge(sym_ranked, bm25_ranked, faiss_ranked)
    filtered_fused = [
        (doc_id, score) for doc_id, score in fused
        if doc_id in sym_ranked or score >= _MIN_RRF_SCORE
    ]

    if not filtered_fused:
        return (
            "No CFN resource schema context applicable to current errors.\n"
            "(No AWS resource types identified in non-syntactic errors.)"
        )

    # 4. Resolve to resource-level (strip "/PropName" suffixes for graph lookup)
    #    but track which properties were pinned via Route A
    seen_resources: dict[str, RetrievedNode] = {}
    for doc_id, rrf_score in fused:
        # Determine resource_type and optional property_name
        if "/" in doc_id:
            rtype, prop = doc_id.split("/", 1)
        else:
            rtype, prop = doc_id, ""

        # Determine which routes contributed to this doc_id
        sources: set[str] = set()
        if doc_id in sym_ranked or rtype in sym_ranked:
            sources.add("symbolic")
        if doc_id in bm25_ranked:
            sources.add("bm25")
        if doc_id in faiss_ranked:
            sources.add("faiss")

        pinned = rtype in {rt for rt in property_hints}

        if rtype not in seen_resources:
            seen_resources[rtype] = RetrievedNode(
                resource_type=rtype,
                property_name=prop,
                rrf_score=rrf_score,
                sources=frozenset(sources),
                pinned=pinned,
            )
        else:
            # Merge sources for same resource found by multiple routes
            existing = seen_resources[rtype]
            seen_resources[rtype] = existing._replace(
                sources=existing.sources | frozenset(sources),
                rrf_score=max(existing.rrf_score, rrf_score),
            )

    if not seen_resources:
        return "No AWS resource types identified in errors."

    # 5. Render — pinned (symbolic) resources first, then by RRF score
    ordered = sorted(
        seen_resources.values(),
        key=lambda n: (not n.pinned, -n.rrf_score),
    )

    blocks: list[str] = []
    for node in ordered:
        block = _render_block(
            G,
            node.resource_type,
            pinned_props=property_hints.get(node.resource_type, []),
            sources=node.sources,
            rrf_score=node.rrf_score,
        )
        blocks.append(block)

    routes_active = []
    routes_active.append("Route A: symbolic regex")
    if bm25_ranked:
        routes_active.append("Route B: BM25 lexical")
    if faiss_ranked:
        routes_active.append("Route C: FAISS dense semantic")

    header = textwrap.dedent(f"""\
        Schema context from AWS CloudFormation Resource Specification v243.
        Multi-route retrieval: {' | '.join(routes_active)}.
        Rankings merged via Reciprocal Rank Fusion (k={_RRF_K}).
        Only resources retrieved from current errors are included.

    """)
    return header + "\n\n".join(blocks)