"""06_test_rag_queries.py

Smoke-test the Terraform GraphRAG pipeline (ChromaDB + Neo4j) against the
kinds of queries that arrive in real remediation scenarios:

  1. Raw TFLint error messages
  2. Raw terraform-apply / LocalStack deploy error messages
  3. Retriever Agent-style structured queries (resource + property names)

For each query the script runs BOTH retrieval paths and prints a side-by-side
comparison so you can judge whether the returned context would actually help
the Generator LLM fix the error.

Usage
-----
    # From the scripts/graphrag/terraform/ directory, with the ChromaDB and
    # Neo4j instances already populated by scripts 01–05:
    python 06_test_rag_queries.py

Environment variables (all optional, fall back to local defaults):
    CHROMADB_PATH   path to ChromaDB persist dir  (default: ./chroma_db)
    EMBED_MODEL     sentence-transformers model    (default: all-MiniLM-L6-v2)
    NEO4J_URI       bolt URI                       (default: bolt://localhost:7687)
    NEO4J_USER                                     (default: neo4j)
    NEO4J_PASSWORD                                 (default: password)
    TOP_K           number of ChromaDB results     (default: 5)
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "all-MiniLM-L6-v2")
NEO4J_URI     = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
NEO4J_USER    = os.getenv("NEO4J_USER",    "neo4j")
NEO4J_PASSWORD= os.getenv("NEO4J_PASSWORD","password")
TOP_K         = int(os.getenv("TOP_K", "5"))
COLLECTION    = "tf_schema_properties"

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    label: str          # short human-readable name shown in the report
    category: str       # "tflint" | "deploy" | "retriever"
    query: str          # the raw string sent to ChromaDB semantic search
    # Optional: resource names to anchor the Neo4j lookup.
    # If empty the script tries to infer them from ChromaDB top-1 result.
    resource_hints: list[str] = field(default_factory=list)


TEST_CASES: list[TestCase] = [

    # ------------------------------------------------------------------
    # CATEGORY 1: TFLint errors
    # These are the literal error strings produced by `tflint --format=json`
    # or the default text output. The Retriever Agent would forward these
    # (or a compressed version) to the RAG tool.
    # ------------------------------------------------------------------

    TestCase(
        label="tflint: missing required attribute bucket in aws_s3_bucket",
        category="tflint",
        query=(
            "tflint error: Missing required argument. "
            "The argument 'bucket' is required, but no definition was found. "
            "Resource: aws_s3_bucket"
        ),
        resource_hints=["aws_s3_bucket"],
    ),

    TestCase(
        label="tflint: invalid instance_type for aws_instance",
        category="tflint",
        query=(
            "tflint aws_instance_invalid_type: "
            "\"t4g.nano\" is an invalid value as instance_type. "
            "Resource: aws_instance"
        ),
        resource_hints=["aws_instance"],
    ),

    TestCase(
        label="tflint: deprecated attribute lifecycle in aws_security_group",
        category="tflint",
        query=(
            "tflint aws_resource_missing_tags_rule: "
            "aws_security_group is missing the required tag 'Environment'. "
            "Rule: aws-security-group-missing-tag"
        ),
        resource_hints=["aws_security_group"],
    ),

    TestCase(
        label="tflint: aws_db_instance missing engine_version",
        category="tflint",
        query=(
            "tflint error: Missing required argument 'engine_version' "
            "for resource aws_db_instance."
        ),
        resource_hints=["aws_db_instance"],
    ),

    # ------------------------------------------------------------------
    # CATEGORY 2: terraform apply / LocalStack deploy errors
    # These come from stderr when `terraform apply` fails, either against
    # real AWS or the LocalStack emulator.
    # ------------------------------------------------------------------

    TestCase(
        label="deploy: InvalidParameterValue vpc_id on aws_subnet",
        category="deploy",
        query=(
            "Error: creating EC2 Subnet: InvalidParameterValue: "
            "The destination for cidrBlock '10.0.1.0/24' conflicts with an existing subnet. "
            "on main.tf line 12, in resource aws_subnet my_subnet: "
            "vpc_id = aws_vpc.main.id"
        ),
        resource_hints=["aws_subnet", "aws_vpc"],
    ),

    TestCase(
        label="deploy: BucketAlreadyOwnedByYou on aws_s3_bucket",
        category="deploy",
        query=(
            "Error: creating Amazon S3 Bucket (my-bucket): "
            "BucketAlreadyOwnedByYou: Your previous request to create the named bucket "
            "succeeded and you already own it. "
            "Resource: aws_s3_bucket"
        ),
        resource_hints=["aws_s3_bucket"],
    ),

    TestCase(
        label="deploy: InvalidClientTokenId — missing provider region",
        category="deploy",
        query=(
            "Error: error configuring Terraform AWS Provider: "
            "error validating provider credentials: "
            "InvalidClientTokenId: The security token included in the request is invalid. "
            "provider aws region us-east-1"
        ),
        resource_hints=[],  # provider-level, not resource-level
    ),

    TestCase(
        label="deploy: aws_iam_role missing assume_role_policy",
        category="deploy",
        query=(
            "Error: creating IAM Role: MalformedPolicyDocument: "
            "The policy document is invalid. Specifically, the following required field "
            "is missing: assume_role_policy. "
            "Resource: aws_iam_role"
        ),
        resource_hints=["aws_iam_role"],
    ),

    TestCase(
        label="deploy: LocalStack — unsupported attribute tags_all on aws_lambda_function",
        category="deploy",
        query=(
            "Error: Provider produced inconsistent result after apply. "
            "on main.tf line 8, in resource aws_lambda_function my_func: "
            "Attribute tags_all: unexpected new value. "
            "LocalStack may not support this attribute."
        ),
        resource_hints=["aws_lambda_function"],
    ),

    # ------------------------------------------------------------------
    # CATEGORY 3: Retriever Agent structured queries
    # These are what the Retriever LLM produces after reading the error —
    # concise schema-lookup strings mixing resource names, attribute names,
    # and functional intent. This is the most direct input format for the RAG.
    # ------------------------------------------------------------------

    TestCase(
        label="retriever: aws_s3_bucket required arguments and encryption config",
        category="retriever",
        query="aws_s3_bucket required arguments bucket server_side_encryption_configuration",
        resource_hints=["aws_s3_bucket"],
    ),

    TestCase(
        label="retriever: aws_instance valid instance_type values and ami",
        category="retriever",
        query="aws_instance valid instance_type ami required attributes",
        resource_hints=["aws_instance"],
    ),

    TestCase(
        label="retriever: aws_security_group ingress egress block schema",
        category="retriever",
        query="aws_security_group ingress egress block required attributes from_port to_port protocol",
        resource_hints=["aws_security_group"],
    ),

    TestCase(
        label="retriever: aws_iam_role assume_role_policy JSON structure",
        category="retriever",
        query="aws_iam_role assume_role_policy required format JSON policy document",
        resource_hints=["aws_iam_role"],
    ),

    TestCase(
        label="retriever: aws_lambda_function runtime handler role required",
        category="retriever",
        query="aws_lambda_function required arguments runtime handler role filename",
        resource_hints=["aws_lambda_function"],
    ),

    TestCase(
        label="retriever: aws_db_instance engine engine_version required blocks",
        category="retriever",
        query="aws_db_instance engine engine_version allocated_storage required arguments",
        resource_hints=["aws_db_instance"],
    ),

    TestCase(
        label="retriever: aws_subnet vpc_id cidr_block availability_zone",
        category="retriever",
        query="aws_subnet required attributes vpc_id cidr_block availability_zone",
        resource_hints=["aws_subnet"],
    ),
]

# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------

def chroma_search(collection, query: str, n_results: int = TOP_K) -> list[dict]:
    """Return top-N results from the tf_schema_properties collection."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "meta": meta, "distance": round(dist, 4)})
    return hits

# ---------------------------------------------------------------------------
# Neo4j graph retrieval
# ---------------------------------------------------------------------------

_GRAPH_QUERY = """
MATCH (r:TFResource {name: $resource_name})
OPTIONAL MATCH (r)-[:HAS_ATTRIBUTE]->(a:TFAttribute)
OPTIONAL MATCH (r)-[:HAS_BLOCK]->(b:TFBlock)
OPTIONAL MATCH (b)-[:HAS_ATTRIBUTE]->(ba:TFAttribute)
OPTIONAL MATCH (r)-[:HAS_EXAMPLE]->(ex:TFExample)
RETURN
    r.name        AS resource,
    r.description AS resource_desc,
    collect(DISTINCT {
        name: a.name, type: a.type, required: a.required,
        description: a.description
    }) AS attributes,
    collect(DISTINCT {
        name: b.name, nesting_mode: b.nesting_mode,
        min_items: b.min_items, max_items: b.max_items
    }) AS blocks,
    collect(DISTINCT ba.name)[..10] AS sample_block_attrs,
    collect(DISTINCT ex.code)[..1]  AS examples
LIMIT 1
"""


def neo4j_lookup(driver, resource_name: str) -> dict[str, Any] | None:
    """Pull the full schema context for one TFResource from Neo4j."""
    with driver.session() as session:
        result = session.run(_GRAPH_QUERY, resource_name=resource_name)
        record = result.single()
        if record is None:
            return None
        return dict(record)


def infer_resource_from_chroma(hits: list[dict]) -> list[str]:
    """Guess resource names from the ChromaDB top results."""
    seen: list[str] = []
    for h in hits:
        r = h["meta"].get("resource", "")
        if r and r not in seen:
            seen.append(r)
        if len(seen) >= 2:
            break
    return seen

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_LINE = "─" * 72


def _wrap(text: str, width: int = 70, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _print_chroma_hits(hits: list[dict]) -> None:
    for i, h in enumerate(hits, 1):
        meta_str = ", ".join(f"{k}={v}" for k, v in h["meta"].items())
        print(f"  [{i}] dist={h['distance']:.4f}  {meta_str}")
        print(_wrap(h["text"], indent="       "))


def _print_neo4j_result(result: dict | None, resource_name: str) -> None:
    if result is None:
        print(f"  ⚠  No TFResource node found for '{resource_name}'")
        return

    print(f"  Resource : {result['resource']}")
    print(f"  Desc     : {result['resource_desc'] or '(empty)'}")

    attrs = [a for a in result["attributes"] if a.get("name")]
    required = [a for a in attrs if a.get("required")]
    optional = [a for a in attrs if not a.get("required")]

    print(f"\n  Required attributes ({len(required)}):")
    for a in required:
        print(f"    • {a['name']} ({a.get('type','?')}): {(a.get('description') or '')[:80]}")

    print(f"\n  Optional attributes ({len(optional)}) — first 5:")
    for a in optional[:5]:
        print(f"    · {a['name']} ({a.get('type','?')}): {(a.get('description') or '')[:80]}")

    blocks = [b for b in result["blocks"] if b.get("name")]
    if blocks:
        print(f"\n  Blocks ({len(blocks)}):")
        for b in blocks[:5]:
            mode = b.get("nesting_mode", "?")
            mn, mx = b.get("min_items", 0), b.get("max_items", 0)
            cardinality = f"min={mn} max={mx}" if (mn or mx) else "unbounded"
            print(f"    ▸ {b['name']} [{mode}, {cardinality}]")
        if result.get("sample_block_attrs"):
            print(f"      Block attrs sample: {', '.join(result['sample_block_attrs'])}")

    examples = result.get("examples") or []
    if examples and examples[0]:
        preview = examples[0][:300].strip()
        print(f"\n  HCL example (first 300 chars):\n")
        for line in preview.splitlines():
            print(f"    {line}")

# ---------------------------------------------------------------------------
# Verdict heuristic
# ---------------------------------------------------------------------------

def _verdict(chroma_hits: list[dict], neo4j_results: list[dict | None]) -> str:
    """Simple heuristic: does the RAG surface anything useful?"""
    has_chroma = any(h["distance"] < 0.6 for h in chroma_hits)
    has_neo4j  = any(r is not None for r in neo4j_results)
    has_required = any(
        r and any(a.get("required") for a in r.get("attributes", []) if a.get("name"))
        for r in neo4j_results
    )

    if has_neo4j and has_required:
        return "✅  USEFUL — Graph has required-arg schema; LLM can remediate"
    elif has_neo4j:
        return "🟡  PARTIAL — Graph found resource but no required attrs indexed"
    elif has_chroma:
        return "🟡  PARTIAL — ChromaDB has relevant chunks; no graph data"
    else:
        return "❌  WEAK — Neither path returned high-confidence context"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Connecting to ChromaDB ...")
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION, embedding_function=embed_fn
        )
    except Exception as exc:
        print(f"ERROR: Could not open ChromaDB collection '{COLLECTION}': {exc}")
        print("Run scripts 01–05 first to populate the knowledge base.")
        return

    print("Connecting to Neo4j ...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as exc:
        print(f"ERROR: Could not connect to Neo4j: {exc}")
        driver = None

    categories = sorted({tc.category for tc in TEST_CASES})
    totals: dict[str, dict[str, int]] = {c: {"useful": 0, "partial": 0, "weak": 0, "total": 0} for c in categories}

    for tc in TEST_CASES:
        print(f"\n{'═' * 72}")
        print(f"  [{tc.category.upper()}] {tc.label}")
        print("═" * 72)
        print(f"\nQuery:\n{_wrap(tc.query, indent='  ')}\n")

        # --- ChromaDB path ---
        print(f"── ChromaDB (semantic, top-{TOP_K}) {'─' * 40}")
        chroma_hits = chroma_search(collection, tc.query, n_results=TOP_K)
        _print_chroma_hits(chroma_hits)

        # --- Determine resources for graph lookup ---
        resource_names = tc.resource_hints or infer_resource_from_chroma(chroma_hits)

        # --- Neo4j path ---
        print(f"\n── Neo4j (graph lookup for: {resource_names}) {'─' * 20}")
        neo4j_results: list[dict | None] = []
        if driver:
            for rname in resource_names:
                result = neo4j_lookup(driver, rname)
                print(f"\n  ▶ {rname}")
                _print_neo4j_result(result, rname)
                neo4j_results.append(result)
        else:
            print("  (Neo4j unavailable — skipped)")

        # --- Verdict ---
        verdict = _verdict(chroma_hits, neo4j_results)
        print(f"\n{_LINE}")
        print(f"  VERDICT: {verdict}")
        print(_LINE)

        # Tally
        totals[tc.category]["total"] += 1
        if "✅" in verdict:
            totals[tc.category]["useful"] += 1
        elif "🟡" in verdict:
            totals[tc.category]["partial"] += 1
        else:
            totals[tc.category]["weak"] += 1

    # --- Summary ---
    print(f"\n\n{'═' * 72}")
    print("  SUMMARY")
    print("═" * 72)
    for cat, counts in totals.items():
        t = counts["total"]
        u = counts["useful"]
        p = counts["partial"]
        w = counts["weak"]
        print(
            f"  {cat:12s}  {t} queries —  "
            f"✅ useful: {u}  🟡 partial: {p}  ❌ weak: {w}"
        )
    print()

    if driver:
        driver.close()


if __name__ == "__main__":
    main()
