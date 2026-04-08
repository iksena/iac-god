"""
GraphRAG context tool — retrieves CFN schema context via BM25 + FAISS + template lookup.

Public interface:
    get_cfn_schema_context(queries, template_yaml, top_k) -> str

Offline prerequisite (run once via scripts/build_cfn_rag_index.py):
    data/cfn_rag_corpus.jsonl   — one doc per property/resource node
    data/cfn_rag_faiss.index    — FAISS FlatIP index
    data/cfn_rag_bm25.pkl       — BM25Okapi index + id list
    data/cfn_graph.pkl          — networkx DiGraph of CFN spec
"""
from __future__ import annotations

import json
import pickle
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import networkx as nx

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from rank_bm25 import BM25Okapi  # noqa: F401
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

_DATA        = Path(__file__).resolve().parents[1] / "data"
_GRAPH_PATH  = _DATA / "cfn_graph.pkl"
_CORPUS_PATH = _DATA / "cfn_rag_corpus.jsonl"
_FAISS_PATH  = _DATA / "cfn_rag_faiss.index"
_BM25_PATH   = _DATA / "cfn_rag_bm25.pkl"

_EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
_RRF_K          = 60
_MIN_RRF_SCORE  = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class CorpusDoc(NamedTuple):
    doc_id: str
    resource_type: str
    property_name: str
    text: str


class RetrievedNode(NamedTuple):
    resource_type: str
    rrf_score: float
    sources: frozenset[str]
    pinned: bool        # True = came from template lookup
    pinned_props: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Lazy loaders
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
    with _CORPUS_PATH.open() as fh:
        return [CorpusDoc(**json.loads(line)) for line in fh]


@lru_cache(maxsize=1)
def _load_faiss():
    if not _FAISS_OK or not _FAISS_PATH.exists():
        return None, []
    index = faiss.read_index(str(_FAISS_PATH))
    return index, [d.doc_id for d in _load_corpus()]


@lru_cache(maxsize=1)
def _load_bm25():
    if not _BM25_OK or not _BM25_PATH.exists():
        return None, []
    with _BM25_PATH.open("rb") as fh:
        bm25_obj, id_list = pickle.load(fh)
    return bm25_obj, id_list


@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(_EMBED_MODEL) if _ST_OK else None


# ─────────────────────────────────────────────────────────────────────────────
# Template lookup — the one structural operation that cannot be replaced by RAG
# Maps logical resource names (only knowable at runtime) to AWS::X::Y types
# ─────────────────────────────────────────────────────────────────────────────

def _parse_template_resource_map(template_yaml: str | None) -> dict[str, str]:
    """Returns {LogicalName: "AWS::X::Y"} from CloudFormation YAML."""
    if not template_yaml or not _YAML_OK:
        return {}
    try:
        tpl = _yaml.safe_load(template_yaml)
        if not isinstance(tpl, dict):
            return {}
        return {
            name: body["Type"]
            for name, body in tpl.get("Resources", {}).items()
            if isinstance(body, dict) and "Type" in body
        }
    except Exception:
        return {}


def _template_retrieve(
    queries: list[str],
    logical_to_type: dict[str, str],
) -> dict[str, int]:
    """
    Scan query strings for logical resource names that appear in the template map.
    Token-based: split each query on whitespace and punctuation, check each token.
    Returns {AWS::X::Y: rank}.
    """
    ranked: dict[str, int] = {}
    position = 1

    for query in queries:
        # Tokenise on any non-alphanumeric boundary — covers "S3Bucket:", "[EC2Instance]"
        tokens = [t.strip("[]().,:'\"") for t in query.split()]
        for token in tokens:
            rtype = logical_to_type.get(token)
            if rtype and rtype not in ranked:
                ranked[rtype] = position
                position += 1

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    bm25, id_list = _load_bm25()
    if bm25 is None or not id_list:
        return {}

    best: dict[str, float] = {}
    for q in queries:
        scores = bm25.get_scores(q.lower().split())
        for idx, score in enumerate(scores):
            doc_id = id_list[idx]
            if score > best.get(doc_id, 0.0):
                best[doc_id] = score

    sorted_docs = sorted(best, key=lambda d: best[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}


# ─────────────────────────────────────────────────────────────────────────────
# FAISS retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _faiss_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    index, id_list = _load_faiss()
    embedder = _load_embedder()
    if index is None or embedder is None or not id_list:
        return {}

    q_vecs = embedder.encode(queries, normalize_embeddings=True).astype("float32")
    best: dict[str, float] = {}
    for q_vec in q_vecs:
        distances, indices = index.search(q_vec[None, :], top_k * 2)
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                doc_id = id_list[idx]
                if float(dist) > best.get(doc_id, -1.0):
                    best[doc_id] = float(dist)

    sorted_docs = sorted(best, key=lambda d: best[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}


# ─────────────────────────────────────────────────────────────────────────────
# RRF merge
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(*ranked_lists: dict[str, int], k: int = _RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
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
    node: RetrievedNode,
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    rtype = node.resource_type

    if not rtype.startswith("AWS::"):
        return ""
    
    sources_str = ", ".join(sorted(node.sources))

    if rtype not in G:
        return (
            f"### {rtype}\n"
            f"*(not in CFN spec)*\n"
            f"*via: {sources_str} | RRF: {node.rrf_score:.4f}*"
        )

    if G.nodes[rtype].get("ntype") not in ("Resource", None):
        return ""

    props, ptypes = _props_for_resource(G, rtype)
    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]
    pinned_set = set(node.pinned_props)

    lines = [
        f"### {rtype}",
        f"*via: {sources_str} | RRF: {node.rrf_score:.4f}*",
    ]

    if node.pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in node.pinned_props:
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
                lines.append(f"  - `{name}` *(invalid property)*")

    req_rest = [p for p in required if p.get("name") not in pinned_set]
    if req_rest:
        parts = [
            f"`{p.get('name','?')}` ({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)"
            for p in req_rest
        ]
        lines.append("**Required properties:** " + ", ".join(parts))

    opt_rest = [p for p in optional if p.get("name") not in pinned_set]
    if opt_rest:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_rest[:max_optional]:
            name = p.get("name", "?")
            prim = p.get("primitive_type") or p.get("type") or "Any"
            upd  = p.get("update_type", "")
            lines.append(
                f"  - `{name}` ({prim}" + (f" — UpdateType: {upd}" if upd else "") + ")"
            )
        if len(opt_rest) > max_optional:
            lines.append(f"  - … and {len(opt_rest) - max_optional} more")

    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — accepts pre-extracted query strings from the caller
# ─────────────────────────────────────────────────────────────────────────────
def _build_prop_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """Map lowercase property name → [AWS::X::Y, ...] that own it."""
    index: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("ntype") == "Property":
            name = (data.get("name") or "").lower()
            # Walk in-edges to find parent resource
            for parent, _ in G.in_edges(node_id):
                if G.nodes[parent].get("ntype") in ("Resource", None):
                    index.setdefault(name, []).append(parent)
    return index

def get_cfn_schema_context(
    queries: list[str],
    template_yaml: str | None = None,
    *,
    top_k: int = 8,
) -> str:
    """
    Given a list of human-readable error/query strings, retrieve relevant
    CFN schema context via template lookup + BM25 + FAISS, merged with RRF.

    Args:
        queries:       Pre-extracted error strings from the caller.
                       These are the actual error messages — no parsing done here.
        template_yaml: Raw CloudFormation YAML, used only to resolve logical
                       resource names (e.g. "S3Bucket") to AWS::X::Y types.
        top_k:         Candidates per BM25/FAISS route.
    """
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    if not queries:
        return "No error queries provided."

    logical_to_type = _parse_template_resource_map(template_yaml)

    template_ranked = _template_retrieve(queries, logical_to_type)
    bm25_ranked     = _bm25_retrieve(queries, top_k=top_k)
    faiss_ranked    = _faiss_retrieve(queries, top_k=top_k)

    prop_index = _build_prop_index(G)
    prop_ranked: dict[str, int] = {}
    position = 1
    for query in queries:
        # Check each whitespace token — "VpcId", "SecurityGroupEgress" will match
        for token in query.split():
            token_clean = token.strip("'\"[]().,:")
            for rtype in prop_index.get(token_clean.lower(), []):
                if rtype not in prop_ranked:
                    prop_ranked[rtype] = position
                    position += 1

    fused = _rrf_merge(template_ranked, prop_ranked, bm25_ranked, faiss_ranked)

    # Template hits always pass; soft routes need minimum RRF score
    filtered = [
        (doc_id, score) for doc_id, score in fused
        if doc_id in template_ranked or score >= _MIN_RRF_SCORE
    ]

    if not filtered:
        return (
            "No CFN resource schema context applicable to current errors.\n"
            "(No resource types identified — errors may be syntactic or value-only.)"
        )

    # Resolve to resource-level; collect property-level hits as pinned props
    seen: dict[str, RetrievedNode] = {}
    pinned_props: dict[str, list[str]] = {}

    for doc_id, rrf_score in filtered:
        if "/" in doc_id:
            rtype, prop = doc_id.split("/", 1)
            if prop:
                pinned_props.setdefault(rtype, [])
                if prop not in pinned_props[rtype]:
                    pinned_props[rtype].append(prop)
        else:
            rtype = doc_id

        sources: set[str] = set()
        if doc_id in template_ranked or rtype in template_ranked:
            sources.add("template")
        if doc_id in bm25_ranked:
            sources.add("bm25")
        if doc_id in faiss_ranked:
            sources.add("faiss")

        pinned = rtype in template_ranked

        if rtype not in seen:
            seen[rtype] = RetrievedNode(
                resource_type=rtype,
                rrf_score=rrf_score,
                sources=frozenset(sources),
                pinned=pinned,
                pinned_props=[],
            )
        else:
            existing = seen[rtype]
            seen[rtype] = existing._replace(
                sources=existing.sources | frozenset(sources),
                rrf_score=max(existing.rrf_score, rrf_score),
                pinned=existing.pinned or pinned,
            )

    # Attach pinned props after all merging
    final_nodes = [
        node._replace(pinned_props=pinned_props.get(node.resource_type, []))
        for node in seen.values()
    ]

    ordered = sorted(
        final_nodes,
        key=lambda n: (not n.pinned, -n.rrf_score),
    )

    blocks = [b for n in ordered if (b := _render_block(G, n))]
    if not blocks:
        return "No renderable schema context found."

    routes = ["template lookup"]
    if bm25_ranked:
        routes.append("BM25")
    if faiss_ranked:
        routes.append("FAISS")

    header = textwrap.dedent(f"""\
        CFN Resource Specification v243 schema context for current errors.
        Retrieved via: {', '.join(routes)} and merged with RRF (k={_RRF_K}).

    """)
    return header + "\n\n".join(blocks)