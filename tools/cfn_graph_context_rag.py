"""
GraphRAG context tool — RAG-only multi-route retrieval (no regex).

Retrieval architecture:
  Route A — Template lookup: parse CloudFormation YAML → resolve logical
             resource names from failed_resources/deploy logs to AWS::X::Y
             types. Zero ambiguity, zero false positives.
  Route B — Sparse lexical (BM25): scores error text against corpus docs.
             Handles cfn-lint rule descriptions, AWS API error codes,
             property names mentioned in prose.
  Route C — Dense semantic (FAISS): cosine-KNN over sentence embeddings.
             Handles paraphrased errors, novel phrasings, deploy messages
             that share no tokens with spec text.

Results merged via Reciprocal Rank Fusion (RRF). Rendered into Markdown
schema blocks for injection into the remediator prompt.

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
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore  # noqa: F401
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_OK = True
except ImportError:
    _ST_OK = False

# ── Paths ────────────────────────────────────────────────────────────────────
_DATA        = Path(__file__).resolve().parents[1] / "data"
_GRAPH_PATH  = _DATA / "cfn_graph.pkl"
_CORPUS_PATH = _DATA / "cfn_rag_corpus.jsonl"
_FAISS_PATH  = _DATA / "cfn_rag_faiss.index"
_BM25_PATH   = _DATA / "cfn_rag_bm25.pkl"

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RRF constant (standard from literature)
_RRF_K = 60

# Minimum fused RRF score to include a result when template lookup found nothing
_MIN_RRF_SCORE = 0.03

# Stages that carry no resource schema information — skip entirely
_SYNTACTIC_STAGES = {"yaml", "json", "comments"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class CorpusDoc(NamedTuple):
    doc_id: str        # "AWS::S3::Bucket" or "AWS::S3::Bucket/BucketEncryption"
    resource_type: str
    property_name: str  # "" for resource-level docs
    text: str


class RetrievedNode(NamedTuple):
    resource_type: str
    property_name: str
    rrf_score: float
    sources: frozenset[str]
    pinned: bool   # True = came from template lookup (highest confidence)

# ─────────────────────────────────────────────────────────────────────────────
# Pure-string utility helpers (no regex)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bracketed_names(text: str) -> list[str]:
    """
    Extract comma-separated names from the first '[...]' in text.
    Used to parse cfn-lint E3004 circular dependency lists and
    deploy ValidationError parameter lists.
    e.g. "Parameters: [KeyName, SubnetId] must have values"
         "Circular dependency with [S3Bucket]"
    """
    start = text.find("[")
    end   = text.find("]", start)
    if start == -1 or end == -1:
        return []
    return [n.strip() for n in text[start + 1:end].split(",") if n.strip()]


def _extract_quoted_names(text: str) -> list[str]:
    """
    Extract single-quoted tokens from cfn-lint error messages.
    cfn-lint always wraps property names, values, and rule targets
    in single quotes: "'AccessControl' is a legacy property"
    Returns every odd-indexed token from split("'") that is non-trivial.
    """
    parts = text.split("'")
    return [
        parts[i].strip()
        for i in range(1, len(parts), 2)
        if len(parts[i].strip()) >= 2
    ]

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
    docs = []
    with _CORPUS_PATH.open() as fh:
        for line in fh:
            d = json.loads(line)
            docs.append(CorpusDoc(**d))
    return docs


@lru_cache(maxsize=1)
def _load_faiss():
    if not _FAISS_OK or not _FAISS_PATH.exists():
        return None, []
    index = faiss.read_index(str(_FAISS_PATH))
    corpus = _load_corpus()
    return index, [d.doc_id for d in corpus]


@lru_cache(maxsize=1)
def _load_bm25():
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

@lru_cache(maxsize=1)
def _build_prop_to_resource_index(G: nx.DiGraph) -> dict[str, set[str]]:
    """
    Reverse index: lowercase property name → set of resource types that own it.
    Built once from graph edges. Used by Route A to resolve cfn-lint errors
    that name a property ('AccessControl') but not its resource type.

    e.g. {"accesscontrol": {"AWS::S3::Bucket"}, "imageid": {"AWS::EC2::Instance"}}
    """
    index: dict[str, set[str]] = {}
    for node_id, data in G.nodes(data=True):
        if data.get("ntype") != "Property":
            continue
        prop_name = data.get("name", "")
        if not prop_name:
            continue
        for parent, _ in G.in_edges(node_id):
            parent_data = G.nodes[parent]
            if parent_data.get("ntype") in ("Resource", None):
                index.setdefault(prop_name.lower(), set()).add(parent)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Route A — Template lookup (replaces regex)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_template_resource_map(template_yaml: str | None) -> dict[str, str]:
    """
    Parse CloudFormation YAML and return {LogicalName: "AWS::X::Y"}.
    Returns empty dict on any failure — callers degrade gracefully.
    """
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

def _template_lookup_retrieve(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    logical_to_type: dict[str, str],
) -> tuple[dict[str, int], set[str]]:
    ranked: dict[str, int] = {}
    failed_types: set[str] = set()
    position = 1

    def _add_type(rtype: str, failed: bool = False) -> None:
        nonlocal position
        if rtype not in ranked:
            ranked[rtype] = position
            position += 1
        if failed:
            failed_types.add(rtype)

    if not deploy_validation_result or deploy_validation_result.get("passed"):
        return ranked, failed_types
    if deploy_validation_result.get("target") == "skipped":
        return ranked, failed_types

    # ── Primary: failed_resources (structured, post-Fix 1) ───────────────────
    for fr in deploy_validation_result.get("failed_resources", []):
        logical = fr.get("logical_name") or fr.get("resource") or ""
        if logical and logical in logical_to_type:
            _add_type(logical_to_type[logical], failed=True)

    # ── Fallback: deployment_logs parsed by string split, no regex ───────────
    # Log entries are formatted as "LogicalName: STATUS - reason"
    # Split on ": " to get the logical name prefix — zero regex involved.
    for line in deploy_validation_result.get("deployment_logs", []):
        line_str = str(line)
        if ": " not in line_str:
            continue
        # Take only the part before the first colon-space
        candidate = line_str.split(": ", 1)[0].strip()
        if candidate in logical_to_type:
            is_failed = "FAILED" in line_str
            _add_type(logical_to_type[candidate], failed=is_failed)

    # ── cfn-lint structured errors ────────────────────────────────────────────
    for result in validation_results:
        if result.get("stage") in _SYNTACTIC_STAGES:
            continue
        for error in result.get("errors", []):
            if isinstance(error, dict):
                logical = error.get("resource") or error.get("logical_id") or ""
                if logical and logical in logical_to_type:
                    _add_type(logical_to_type[logical])
    
    # ── Pass 3: cfn-lint string errors — logical name from "for resource X" ──
    # Handles E3004 circular dependency: "for resource S3Bucket"
    # and "Circular dependency with [S3Bucket]" bracket list.
    _LOGICAL_MARKERS = ("for resource ", "with resource ", "resource ID ")
    for result in validation_results:
        if result.get("stage") in _SYNTACTIC_STAGES:
            continue
        for error in result.get("errors", []):
            if isinstance(error, dict):
                continue  # already handled in Pass 2 above
            error_str = str(error)

            # Marker-based: "for resource S3Bucket"
            for marker in _LOGICAL_MARKERS:
                if marker in error_str:
                    tail = error_str.split(marker, 1)[1]
                    logical = tail.split()[0].rstrip(".,]")
                    rtype = logical_to_type.get(logical, "")
                    if rtype:
                        _add_type(rtype)

            # Bracket list: "[S3Bucket, OtherBucket]"
            for name in _extract_bracketed_names(error_str):
                rtype = logical_to_type.get(name, "")
                if rtype:
                    _add_type(rtype)

    # ── Pass 4: property-name extraction via single-quote tokens ─────────────
    # Handles E3045/W3045: "'AccessControl' is a legacy property"
    # Resolves property name → resource type via reverse graph index.
    G = _load_graph()
    if G is not None:
        prop_index = _build_prop_to_resource_index(G)
        for result in validation_results:
            if result.get("stage") in _SYNTACTIC_STAGES:
                continue
            for error in result.get("errors", []):
                for candidate in _extract_quoted_names(str(error)):
                    for rtype in prop_index.get(candidate.lower(), set()):
                        _add_type(rtype)

    # ── Pass 5: stack-level deploy error → all template resource types ────────
    # Handles "Parameters: [KeyName] must have values" (CreateStack rejected
    # before any resource runs — no logical name available).
    _STACK_LEVEL_SIGNALS = (
        "must have values",
        "does not exist in the template",
        "Unresolved resource dependencies",
        "No export named",
    )
    if deploy_validation_result and not deploy_validation_result.get("passed"):
        msg = deploy_validation_result.get("error_message", "") or ""
        is_stack_level = any(sig in msg for sig in _STACK_LEVEL_SIGNALS)

        # Also stack-level if only "stack" appears in failed_resources
        failed = deploy_validation_result.get("failed_resources", [])
        has_only_stack = bool(failed) and all(
            (fr.get("logical_name") or fr.get("resource", "")) in ("stack", "")
            for fr in failed
        )

        if is_stack_level or has_only_stack:
            for rtype in logical_to_type.values():
                _add_type(rtype)

    return ranked, failed_types


# ─────────────────────────────────────────────────────────────────────────────
# Query construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_queries(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            queries.append(s)

    for result in validation_results:
        stage = result.get("stage", "")
        if stage in _SYNTACTIC_STAGES:
            continue
        for error in result.get("errors", []):
            if isinstance(error, dict):
                msg  = error.get("message") or error.get("rule", "")
                rule = error.get("rule", "")
                _add(f"[{stage}] {rule} {msg}".strip())
            else:
                error_str = str(error)
                # Strip cfn-lint location dict: everything before last "}: "
                # "[E3045] {'ColumnNumber': 5, 'LineNumber': 258}: <human message>"
                # Split on "}: " and keep only the trailing human message.
                if "}: " in error_str:
                    human = error_str.split("}: ")[-1].strip()
                else:
                    human = error_str
                _add(f"[{stage}] {human}")

    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("target") != "skipped":
            if deploy_validation_result.get("error_message"):
                _add(f"[deploy] {deploy_validation_result['error_message']}")
            for fr in deploy_validation_result.get("failed_resources", []):
                reason = fr.get("status_reason") or fr.get("reason") or ""
                if reason:
                    _add(f"[deploy] {reason}")

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Route B — BM25 (sparse lexical)
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    bm25, id_list = _load_bm25()
    if bm25 is None or not id_list:
        return {}

    best_scores: dict[str, float] = {}
    for q in queries:
        tokens = q.lower().split()
        scores = bm25.get_scores(tokens)
        for idx, score in enumerate(scores):
            doc_id = id_list[idx]
            if score > best_scores.get(doc_id, 0.0):
                best_scores[doc_id] = score

    sorted_docs = sorted(best_scores, key=lambda d: best_scores[d], reverse=True)
    return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_docs[:top_k])}


# ─────────────────────────────────────────────────────────────────────────────
# Route C — FAISS dense semantic
# ─────────────────────────────────────────────────────────────────────────────

def _faiss_retrieve(queries: list[str], top_k: int = 10) -> dict[str, int]:
    index, id_list = _load_faiss()
    embedder = _load_embedder()
    if index is None or embedder is None or not id_list:
        return {}

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
# RRF merge
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(*ranked_lists: dict[str, int], k: int = _RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Context renderer
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
    sources: frozenset[str],
    rrf_score: float,
    pinned_prop_names: list[str],
    *,
    max_optional: int = 12,
    max_nested: int = 8,
) -> str:
    if rtype not in G:
        return (
            f"### {rtype}\n"
            f"*(not found in CFN spec — resource type may be invalid)*\n"
            f"*Retrieved via: {', '.join(sorted(sources))} | RRF: {rrf_score:.4f}*"
        )

    node_data = G.nodes[rtype]
    if node_data.get("ntype") not in ("Resource", None):
        return ""  # filter PropertyType ghost nodes

    props, ptypes = _props_for_resource(G, rtype)
    required = [p for p in props if p.get("required")]
    optional = [p for p in props if not p.get("required")]
    pinned_set = set(pinned_prop_names)

    lines = [
        f"### {rtype}",
        f"*Retrieved via: {', '.join(sorted(sources))} | RRF: {rrf_score:.4f}*",
    ]

    # Pinned properties — BM25/FAISS returned a property-level doc for this resource
    if pinned_prop_names:
        lines.append("**Properties relevant to errors:**")
        for name in pinned_prop_names:
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
                lines.append(f"  - `{name}` *(not in spec — invalid property name)*")

    # Required remainder
    req_rest = [p for p in required if p.get("name") not in pinned_set]
    if req_rest:
        parts = [
            f"`{p.get('name','?')}` ({p.get('primitive_type') or p.get('type') or 'Any'}, **required**)"
            for p in req_rest
        ]
        lines.append("**Required properties:** " + ", ".join(parts))
    elif not pinned_prop_names:
        lines.append("**Required properties:** *(none)*")

    # Optional sample
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
        leftover = len(opt_rest) - max_optional
        if leftover > 0:
            lines.append(f"  - … and {leftover} more")

    if ptypes:
        nested = [nd.get("name", "?").rsplit(".", 1)[-1] for nd in ptypes[:max_nested]]
        lines.append("**Nested types:** " + ", ".join(f"`{n}`" for n in nested))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def get_cfn_graph_context_for_state(
    validation_results: list[dict],
    deploy_validation_result: dict | None = None,
    template_yaml: str | None = None,
    *,
    top_k_bm25: int = 8,
    top_k_faiss: int = 8,
) -> str:
    """
    RAG-only GraphRAG context — no regex on error strings.

    Route A: template YAML parse → logical name → resource type (exact)
    Route B: BM25 lexical match on error text vs corpus
    Route C: FAISS dense semantic match on error text vs corpus

    Merged via RRF. Template-matched resources are always included;
    BM25/FAISS results require a minimum RRF score when Route A is empty.

    Args:
        validation_results:       state["validation_results"]
        deploy_validation_result: state.get("deploy_validation_result")
        template_yaml:            state.get("cloudformation_template")
    """
    G = _load_graph()
    if G is None:
        return "CFN schema graph not available (data/cfn_graph.pkl missing)."

    # Route A — template lookup (no regex, no ambiguity)
    logical_to_type = _parse_template_resource_map(template_yaml)
    template_ranked, failed_types = _template_lookup_retrieve(
        validation_results, deploy_validation_result, logical_to_type
    )

    if G is not None:
        prop_index = _build_prop_to_resource_index(G)
        _early_pinned: dict[str, list[str]] = {}
        for result in validation_results:
            if result.get("stage") in _SYNTACTIC_STAGES:
                continue
            for error in result.get("errors", []):
                for candidate in _extract_quoted_names(str(error)):
                    for rtype in prop_index.get(candidate.lower(), set()):
                        _early_pinned.setdefault(rtype, [])
                        if candidate not in _early_pinned[rtype]:
                            _early_pinned[rtype].append(candidate)

    # Build error queries for BM25/FAISS
    queries = _build_queries(validation_results, deploy_validation_result)

    if not queries and not template_ranked:
        return (
            "No CFN resource schema context applicable to current errors.\n"
            "(Errors appear to be syntactic — YAML formatting, indentation, etc.)"
        )

    # Routes B and C
    bm25_ranked  = _bm25_retrieve(queries, top_k=top_k_bm25)  if queries else {}
    faiss_ranked = _faiss_retrieve(queries, top_k=top_k_faiss) if queries else {}

    # RRF merge across all three routes
    fused = _rrf_merge(template_ranked, bm25_ranked, faiss_ranked)

    # Gate: template hits always pass; soft routes need minimum score
    filtered_fused = [
        (doc_id, score) for doc_id, score in fused
        if doc_id in template_ranked
        or (score >= _MIN_RRF_SCORE)
    ]

    if not filtered_fused:
        return (
            "No CFN resource schema context applicable to current errors.\n"
            "(No resource types identified, errors may be syntactic or value-only.)"
        )

    # Resolve to resource-level, collect property-level hits as pinned props
    seen_resources: dict[str, RetrievedNode] = {}
    pinned_props_map: dict[str, list[str]] = dict(_early_pinned)

    for doc_id, rrf_score in filtered_fused:
        if "/" in doc_id:
            rtype, prop = doc_id.split("/", 1)
            # Property-level doc hit → promote as pinned prop for this resource
            if prop:
                pinned_props_map.setdefault(rtype, [])
                if prop not in pinned_props_map[rtype]:
                    pinned_props_map[rtype].append(prop)
        else:
            rtype, prop = doc_id, ""

        sources: set[str] = set()
        if doc_id in template_ranked or rtype in template_ranked:
            sources.add("template")
        if doc_id in bm25_ranked:
            sources.add("bm25")
        if doc_id in faiss_ranked:
            sources.add("faiss")

        pinned = rtype in template_ranked

        if rtype not in seen_resources:
            seen_resources[rtype] = RetrievedNode(
                resource_type=rtype,
                property_name=prop,
                rrf_score=rrf_score,
                sources=frozenset(sources),
                pinned=pinned,
            )
        else:
            existing = seen_resources[rtype]
            seen_resources[rtype] = existing._replace(
                sources=existing.sources | frozenset(sources),
                rrf_score=max(existing.rrf_score, rrf_score),
                pinned=existing.pinned or pinned,
            )

    if not seen_resources:
        return "No AWS resource types identified in errors."

    # Render — failed resources first, then template-matched, then soft-route by RRF
    ordered = sorted(
        seen_resources.values(),
        key=lambda n: (
            n.resource_type not in failed_types,  # failed first
            not n.pinned,                          # template-matched second
            -n.rrf_score,                          # highest RRF last tiebreak
        ),
    )

    blocks: list[str] = []
    for node in ordered:
        block = _render_block(
            G,
            node.resource_type,
            sources=node.sources,
            rrf_score=node.rrf_score,
            pinned_prop_names=pinned_props_map.get(node.resource_type, []),
        )
        if block:
            blocks.append(block)

    if not blocks:
        return "No renderable schema context found."

    routes_active = ["Route A: template lookup"]
    if bm25_ranked:
        routes_active.append("Route B: BM25 lexical")
    if faiss_ranked:
        routes_active.append("Route C: FAISS dense semantic")

    header = textwrap.dedent(f"""\
        Schema context from AWS CloudFormation Resource Specification v243.
        Multi-route retrieval: {' | '.join(routes_active)}.
        Rankings merged via Reciprocal Rank Fusion (k={_RRF_K}).
        Only resources from current errors are included.

    """)
    return header + "\n\n".join(blocks)