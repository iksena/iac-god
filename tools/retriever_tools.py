"""tools/retriever_tools.py

LangChain @tool that wraps the full hybrid-RAG retrieval pipeline.

The Remediator LLM calls `retrieve_schema_context` directly with its own
generated retrieval queries.  The tool executes the two-stage retrieval
(ChromaDB semantic search + Neo4j graph traversal) and returns the assembled
schema context string as the tool result.

Design constraints:
  - No imports from agents/  (unidirectional: tools <- agents).
  - All heavy I/O (DB connections) is deferred to call-time.
  - Intentionally stateless: all context passed as arguments.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class RetrieveSchemaContextInput(BaseModel):
    """Input schema for retrieve_schema_context."""

    retrieval_queries: Annotated[
        list[str],
        Field(
            description=(
                "Targeted AWS CloudFormation schema-lookup queries you generated "
                "based on the annotated template and validation errors. "
                "Each query should name a specific Resource.Property pair "
                "(e.g. 'AWS::S3::Bucket.BucketEncryption') or describe a "
                "schema constraint you need to look up. "
                "Do NOT include queries already used in previous iterations."
            )
        ),
    ]
    template_yaml: Annotated[
        str,
        Field(
            description=(
                "The current CloudFormation YAML template string (annotated or raw). "
                "Used to seed the Neo4j graph traversal with resource types "
                "present in the template so related schema blocks are fetched "
                "even when the vector search misses them."
            )
        ),
    ]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool(args_schema=RetrieveSchemaContextInput)
def retrieve_schema_context(
    retrieval_queries: list[str],
    template_yaml: str,
) -> str:
    """Retrieve official AWS CloudFormation schema context for the current template errors.

    Runs a two-stage hybrid retrieval pipeline:
      Stage 1 — ChromaDB semantic search over pre-indexed CFN property chunks.
      Stage 2 — Neo4j graph traversal for each resource type found in the template.

    Call this tool when cfn-lint or deployment errors are present and you need
    the official schema (required properties, types, constraints) to fix them.
    Pass the queries YOU generated based on the annotated template and errors.

    Do NOT call this tool for YAML syntax errors or pure security violations
    (checkov / trivy IDs) — those do not benefit from schema context.

    Returns a multi-section schema context string to use in your fix objectives.
    """
    # Deferred imports — keep module-level import cost negligible.
    from tools.cfn_hybrid_rag import execute_hybrid_retrieval          # noqa: PLC0415
    from tools.template_annotator import annotate_template, extract_resource_types  # noqa: PLC0415

    # Seed Neo4j with resource types extracted from the template.
    annotation = annotate_template(file_path="template.yaml", content=template_yaml)
    seed_resources = extract_resource_types(annotation)

    n_resources = len(annotation.resources) if annotation else 0
    print(f"[Tool:retrieve_schema_context] Annotation: {n_resources} resources parsed.")
    print(f"[Tool:retrieve_schema_context] Running hybrid retrieval with {len(retrieval_queries)} queries...")

    context = execute_hybrid_retrieval(
        retrieval_queries=retrieval_queries,
        seed_resources=seed_resources,
    )

    print(f"[Tool:retrieve_schema_context] Context: {len(context)} chars returned.")
    return context
