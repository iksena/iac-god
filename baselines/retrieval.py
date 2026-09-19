"""IaCGOD's Retriever with its query-generation LLM call removed.

Backs the retrieve_context MCP tool for the Harness + Retriever ablation arm.

agents/retriever.py:retriever_agent does two things: it asks a separate LLM for
schema queries, then runs deterministic retrieval with them (ChromaDB property
chunks -> RRF -> Neo4j schema graph, plus Neo4j security rules). This module
keeps only the second half, and takes the queries from the harness's own LLM
instead. Every step below is imported from agents/retriever.py or tools/, not
reimplemented, so the only variable between this arm and IaCGOD's Retriever is
who writes the queries.

Imports are deferred into the functions so runs without --enable-retrieval
never load langchain/chromadb/neo4j or need those services running.
"""
from __future__ import annotations

from typing import Any

# parse_query_response (tools/retriever_helpers.py) caps the Retriever LLM's
# output at 8 queries; the fallback below uses errors[:8] for the same reason.
MAX_SCHEMA_QUERIES = 8


def retrieve_for_scenario(
    *,
    iac_type: str,
    schema_queries: list[str],
    template: str,
    validation_results: list[dict[str, Any]],
    deploy_result: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Mirror of agents/retriever.py:retriever_agent from the point its LLM
    call returns — same error extraction, routing, seeding, retrieval entry
    points and context assembly.

    Returns (context, meta). context is exactly what IaCGOD's Retriever would
    hand to the Remediator for the same queries, template and errors.
    """
    import config  # noqa: F401  loads .env (CHROMA_*, NEO4J_*, EMBEDDING_PROVIDER) as benchmark.py does
    from agents.retriever import (
        _annotate_safely,
        _build_tf_seed_resources,
        _extract_error_resources,
        _get_active_error_types,
    )
    from tools.retriever_helpers import extract_errors
    from tools.template_annotator import extract_resource_types

    errors = extract_errors(validation_results, deploy_result)
    annotation = _annotate_safely(template=template, smell_report=None, iac_type=iac_type)
    has_schema, has_security = _get_active_error_types(
        {"validation_results": validation_results, "deploy_validation_result": deploy_result}
    )

    queries = [q.strip() for q in schema_queries if q.strip()][:MAX_SCHEMA_QUERIES]
    used_fallback = False
    iac_context = ""

    if has_schema:
        if not queries:
            queries = errors[:MAX_SCHEMA_QUERIES]
            used_fallback = True

        if iac_type == "terraform":
            seed_resources = _build_tf_seed_resources(annotation)
        else:
            seed_resources = extract_resource_types(annotation)

        error_resources = _extract_error_resources(
            errors=errors,
            annotation=annotation,
            deploy_validation_result=deploy_result,
        ) or None

        if iac_type == "terraform":
            from tools.tf_hybrid_rag import execute_terraform_retrieval

            iac_context = execute_terraform_retrieval(
                retrieval_queries=queries,
                seed_resources=seed_resources,
                error_resources=error_resources,
            )
        else:
            from tools.cfn_hybrid_rag import execute_hybrid_retrieval

            iac_context = execute_hybrid_retrieval(
                retrieval_queries=queries,
                seed_resources=seed_resources,
                error_resources=error_resources,
            )
    else:
        # Same as the Retriever: with only security findings active, schema
        # retrieval is skipped and any harness-written queries go unused.
        queries = []

    security_context = ""
    security_ids: list[str] = []
    if has_security:
        from tools.security_hybrid_rag import (
            execute_security_retrieval,
            extract_trivy_check_ids,
        )

        security_ids = extract_trivy_check_ids(errors)
        security_context = execute_security_retrieval(raw_errors=errors, iac_type=iac_type)

    context = "\n\n".join(part for part in (iac_context, security_context) if part.strip())
    meta = {
        "schema_retrieval": has_schema,
        "security_retrieval": has_security,
        "schema_queries": queries,
        "used_fallback": used_fallback,
        "security_ids": security_ids,
        "error_count": len(errors),
    }
    return context, meta


_PREAMBLE = """
## Knowledge-base retrieval (retrieve_context)

You also have a `retrieve_context` tool. It searches the knowledge base:
{corpus}, the resource schema graph, and remediation guidance for security
rules.

**Every time `validate_iac` or `deploy_iac` reports a failure, call
`retrieve_context` once before you edit the template.** Then fix the template
using what it returns and call `validate_iac` again.

- It is available only after a `validate_iac` or `deploy_iac` call has failed,
  at most once per iteration (once per `validate_iac` call). It never uses up an
  iteration.
- It works from the template you last passed to `validate_iac` and the errors
  that call (or the `deploy_iac` after it) reported. You do not pass a file path.
- Pass `schema_queries`: at most 8 short retrieval queries, written by following
  the query-planning guide below. Pass an empty list to search with the raw
  error messages instead.
- Security findings (trivy) are looked up automatically from their rule IDs. Do
  not write queries for them.
- Your earlier `retrieve_context` queries in this conversation are your "Prior
  Retrieval Queries". Do not repeat them.

### Query-planning guide

These are the instructions to write the retrieval queries. 
Follow them, but ignore their "Output format" section: pass the list as
the `schema_queries` argument of `retrieve_context` instead of replying with JSON.

"""

_CORPUS = {
    "cloudformation": "official AWS CloudFormation resource specification property records",
    "terraform": "official Terraform AWS provider schema argument and attribute records",
}


def build_retrieval_addendum(iac_type: str) -> str:
    """System-prompt addendum for runs with retrieval enabled.

    Embeds get_query_gen_system(iac_type) verbatim rather than paraphrasing
    it, so the harness writes queries under exactly the rules the Retriever's
    LLM does and the two can't drift apart.
    """
    from prompts.retriever_prompt import get_query_gen_system

    preamble = _PREAMBLE.format(corpus=_CORPUS.get(iac_type, _CORPUS["cloudformation"]))
    return preamble + get_query_gen_system(iac_type)
