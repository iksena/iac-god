"""
GraphRAG context tool — retrieves CFN schema context via BM25 + FAISS + template anchoring.
Dynamically bounds semantic search to prevent hallucinated resource types.
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
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

# Paths configuration
_DATA        = Path(__file__).resolve().parents[1] / "data"
_GRAPH_PATH  = _DATA / "cfn_graph.pkl"
_CORPUS_PATH = _DATA / "cfn_rag_corpus.jsonl"
_FAISS_PATH  = _DATA / "cfn_rag_faiss.index"
_BM25_PATH   = _DATA / "cfn_rag_bm25.pkl"

_EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
_RRF_K          = 60
_MIN_RRF_SCORE  = 0.01

class CorpusDoc(NamedTuple):
    doc_id: str
    resource_type: str
    property_name: str
    text: str

class RetrievedNode(NamedTuple):
    resource_type: str
    rrf_score: float
    sources: frozenset[str]
    pinned: bool
    pinned_props: list[str]

# =============================================================================
# Loaders & Caching
# =============================================================================

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph | None:
    if not _GRAPH_PATH.exists(): return None
    with _GRAPH_PATH.open("rb") as fh:
        obj = pickle.load(fh)
    return obj[0] if isinstance(obj, tuple) else obj

@lru_cache(maxsize=1)
def _load_corpus() -> list[CorpusDoc]:
    if not _CORPUS_PATH.exists(): return []
    with _CORPUS_PATH.open() as fh:
        return [CorpusDoc(**json.loads(line)) for line in fh]

@lru_cache(maxsize=1)
def _load_faiss():
    if not _FAISS_OK or not _FAISS_PATH.exists(): return None, []
    index = faiss.read_index(str(_FAISS_PATH))
    return index, [d.doc_id for d in _load_corpus()]

@lru_cache(maxsize=1)
def _load_bm25():
    if not _BM25_OK or not _BM25_PATH.exists(): return None, []
    with _BM25_PATH.open("rb") as fh:
        bm25_obj, id_list = pickle.load(fh)
    return bm25_obj, id_list

@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(_EMBED_MODEL) if _ST_OK else None

# =============================================================================
# Retrieval Mechanisms
# =============================================================================

def _parse_template_resource_map(template_yaml: str | None) -> dict[str, str]:
    """Extracts a map of {LogicalId: AWS::Service::Type} from the current template."""
    if not template_yaml or not _YAML_OK: return {}
    try:
        tpl = _yaml.safe_load(template_yaml)
        if not isinstance(tpl, dict): return {}
        return {
            name: body["Type"]
            for name, body in tpl.get("Resources", {}).items()
            if isinstance(body, dict) and "Type" in body
        }
    except Exception:
        return {}

def _template_retrieve(queries: list[str], logical_to_type: dict[str, str]) -> dict[str, int]:
    """Retrieves resources by exact Logical ID or AWS::Type match in the queries."""
    ranked: dict[str, int] = {}
    position = 1
    logical_to_type_lower = {k.lower(): v for k, v in logical_to_type.items()}
    template_types_lower = {v.lower(): v for v in logical_to_type.values()}

    for query in queries:
        tokens = [t.strip("[]().,:'\"").lower() for t in query.split()]
        for token in tokens:
            rtype = logical_to_type_lower.get(token) or template_types_lower.get(token)
            if rtype and rtype not in ranked:
                ranked[rtype] = position
                position += 1
    return ranked

def _bm25_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    bm25, id_list = _load_bm25()
    if bm25 is None or not id_list: return {}
    best: dict[str, float] = {}
    for q in queries:
        scores = bm25.get_scores(q.lower().split())
        for idx, score in enumerate(scores):
            doc_id = id_list[idx]
            if score > best.get(doc_id, 0.0):
                best[doc_id] = score
    sorted_docs = sorted(best, key=lambda d: best[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}

def _faiss_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    index, id_list = _load_faiss()
    embedder = _load_embedder()
    if index is None or embedder is None or not id_list: return {}
    
    # Batch encode queries
    q_vecs = embedder.encode(queries, normalize_embeddings=True).astype("float32")
    if len(q_vecs.shape) == 1:
        q_vecs = q_vecs.reshape(1, -1)
        
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

def _rrf_merge(*ranked_lists: dict[str, int], k: int = _RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# =============================================================================
# Graph Context & Markdown Rendering
# =============================================================================

def _build_prop_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """Maps property names (e.g. 'ServerSideEncryptionConfiguration') to parent resources."""
    index: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("ntype") == "Property":
            name = (data.get("name") or "").lower()
            for parent, _ in G.in_edges(node_id):
                # FIX: Handle "ResourceType" from builder script
                if G.nodes[parent].get("ntype") in ("ResourceType", "Resource", None):
                    index.setdefault(name, []).append(parent)
    return index

def _props_for_resource(G: nx.DiGraph, rtype: str) -> tuple[list[dict], list[dict]]:
    props, ptypes = [], []
    for _, nbr in G.out_edges(rtype):
        nd = G.nodes[nbr]
        if nd.get("ntype") == "Property": props.append(nd)
        elif nd.get("ntype") == "PropertyType": ptypes.append(nd)
    return props, ptypes

def _render_block(G: nx.DiGraph, node: RetrievedNode, *, max_optional: int = 12, max_nested: int = 8) -> str:
    rtype = node.resource_type
    if not rtype.startswith("AWS::"): return ""
    sources_str = ", ".join(sorted(node.sources))
    if rtype not in G:
        return f"### {rtype}\n*(not in CFN spec)*\n*via: {sources_str} | RRF: {node.rrf_score:.4f}*"
    
    # FIX: Must match graph schema's 'ResourceType'
    if G.nodes[rtype].get("ntype") not in ("ResourceType", "Resource", None):
        return ""

    props, ptypes = _props_for_resource(G, rtype)
    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]
    pinned_set = set(node.pinned_props)

    lines = [f"### {rtype}", f"*via: {sources_str} | RRF: {node.rrf_score:.4f}*"]

    if node.pinned_props:
        lines.append("**Properties flagged in errors:**")
        for name in node.pinned_props:
            match = next((p for p in props if p.get("name") == name), None)
            if match:
                prim = match.get("primitive_type") or match.get("type") or "Any"
                req  = "**required**" if match.get("required") else "optional"
                lines.append(f"  - `{name}` ({prim}, {req})")
            else:
                lines.append(f"  - `{name}` *(invalid property for this resource)*")

    req_rest = [p for p in required if p.get("name") not in pinned_set]
    if req_rest:
        parts = [f"`{p.get('name','?')}` ({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)" for p in req_rest]
        lines.append("**Required properties:** " + ", ".join(parts))

    opt_rest = [p for p in optional if p.get("name") not in pinned_set]
    if opt_rest:
        lines.append(f"**Optional properties (first {max_optional}):**")
        for p in opt_rest[:max_optional]:
            name = p.get("name", "?")
            prim = p.get("primitive_type") or p.get("type") or "Any"
            lines.append(f"  - `{name}` ({prim})")
        if len(opt_rest) > max_optional:
            lines.append(f"  - … and {len(opt_rest) - max_optional} more")

    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)

# =============================================================================
# Core Orchestration (The Fixed RAG Pipeline)
# =============================================================================

def get_cfn_schema_context(
    queries: list[str] | str,
    template_yaml: str | None = None,
    *,
    top_k: int = 8,
) -> str:
    """Retrieves context by explicitly anchoring search to the current YAML state."""
    
    # 1. Standardize Inputs
    if isinstance(queries, str):
        queries = [queries]
        
    G = _load_graph()
    if G is None: return "CFN schema graph not available."
    if not queries: return "No error queries provided."

    # 2. Extract Template Ground Truth
    logical_to_type = _parse_template_resource_map(template_yaml)
    template_types = set(logical_to_type.values())

    # 3. Anchor Semantic Queries to Template State
    # This prevents FAISS from hallucinating resources based on vague error strings.
    augmented_queries = queries.copy()
    if logical_to_type:
        augmented_queries.append(" ".join(logical_to_type.keys()))  # Add Logical IDs
        augmented_queries.append(" ".join(template_types))          # Add AWS::Types

    template_ranked = _template_retrieve(augmented_queries, logical_to_type)
    bm25_ranked     = _bm25_retrieve(augmented_queries, top_k=top_k)
    faiss_ranked    = _faiss_retrieve(augmented_queries, top_k=top_k)

    # 4. Strict Property Filtering
    # Map raw error words to properties, but strictly reject properties
    # belonging to resources that are not currently in the user's template.
    prop_index = _build_prop_index(G)
    prop_ranked: dict[str, int] = {}
    position = 1
    
    for query in queries:
        for token in query.split():
            token_clean = token.strip("'\"[]().,:")
            for rtype in prop_index.get(token_clean.lower(), []):
                # STRICT FILTER: Prevents generic words like "Name" from retrieving 500 schemas
                if template_types and rtype not in template_types: 
                    continue
                if rtype not in prop_ranked:
                    prop_ranked[rtype] = position
                    position += 1

    # 5. Fusion & Resolution
    fused = _rrf_merge(template_ranked, prop_ranked, bm25_ranked, faiss_ranked)

    filtered = []
    for doc_id, score in fused:
        base_rtype = doc_id.split("/")[0]
        # Always retain exact template hits, regardless of RRF score
        if doc_id in template_ranked or base_rtype in template_types:
            filtered.append((doc_id, score))
        elif score >= _MIN_RRF_SCORE:
            filtered.append((doc_id, score))

    if not filtered:
        return "No CFN resource schema context applicable to current errors."

    seen: dict[str, RetrievedNode] = {}
    pinned_props: dict[str, list[str]] = {}

    for doc_id, rrf_score in filtered:
        if "/" in doc_id:
            rtype, prop = doc_id.split("/", 1)
            if prop:
                pinned_props.setdefault(rtype, []).append(prop)
        else:
            rtype = doc_id

        sources = set()
        if doc_id in template_ranked or rtype in template_ranked: sources.add("template_anchor")
        if doc_id in prop_ranked or rtype in prop_ranked: sources.add("property_match")
        if doc_id in bm25_ranked: sources.add("bm25")
        if doc_id in faiss_ranked: sources.add("faiss")

        pinned = rtype in template_ranked
        if rtype not in seen:
            seen[rtype] = RetrievedNode(rtype, rrf_score, frozenset(sources), pinned, [])
        else:
            existing = seen[rtype]
            seen[rtype] = existing._replace(
                sources=existing.sources | frozenset(sources),
                rrf_score=max(existing.rrf_score, rrf_score),
                pinned=existing.pinned or pinned,
            )

    # 6. Render
    final_nodes = [n._replace(pinned_props=pinned_props.get(n.resource_type, [])) for n in seen.values()]
    ordered = sorted(final_nodes, key=lambda n: (not n.pinned, -n.rrf_score))
    blocks = [b for n in ordered if (b := _render_block(G, n))]
    
    if not blocks: 
        return "No renderable schema context found."

    return f"CFN Resource Specification v243 Schema Context.\nRetrieved & Anchored via Template State, merged with FAISS/BM25 (k={_RRF_K}).\n\n" + "\n\n".join(blocks)

# =============================================================================
# Adapter Interface for Remediator Agent
# =============================================================================

def get_cfn_graph_context_for_state(
    validation_results: list[dict], 
    deploy_validation_result: dict | None, 
    template_yaml: str | None
) -> str:
    """Parses LangGraph state dictionary into clean query arrays for the RAG pipeline."""
    queries = []
    
    # Add static validation errors
    for result in validation_results:
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip(): 
                    queries.append(str(err))
                    
    # Add dynamic deploy validation errors
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("error_message"): 
            queries.append(deploy_validation_result["error_message"])
            
        for fr in deploy_validation_result.get("failed_resources", []):
            name = fr.get("logical_name") or fr.get("resource") or ""
            reason = fr.get("status_reason") or fr.get("reason") or ""
            if name or reason: 
                queries.append(f"{name} {reason}")
                
    return get_cfn_schema_context(queries, template_yaml=template_yaml)