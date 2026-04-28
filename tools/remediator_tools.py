"""tools/remediator_tools.py

LangChain @tool definitions for the Remediator agent's ToolNode.

Currently exposes:
  build_retrieval_queries  — wraps the query-generation LLM call that was
                             previously owned by retriever_agent. The Remediator
                             LLM can call this tool when cfn-lint or deployment
                             errors are present to obtain targeted schema-lookup
                             queries before calling hybrid_rag_search.

Design constraints:
  - No imports from agents/  (unidirectional dependency: tools ← agents).
  - All heavy I/O (LLM, DB) is deferred to call-time so module import is fast.
  - The tool is intentionally stateless: all context it needs must be passed
    as arguments, making it unit-testable without GraphState.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class BuildRetrievalQueriesInput(BaseModel):
    """Input schema for build_retrieval_queries."""

    annotated_template: Annotated[
        str,
        Field(
            description=(
                "CloudFormation YAML string. When errors carry line numbers the "
                "template will have inline `# ERROR:` comments injected at the "
                "exact reported lines — use these as the primary signal for which "
                "Resource.Property pairs need schema context. Pass the raw YAML "
                "if no annotation is available."
            )
        ),
    ]
    validation_errors: Annotated[
        list[str],
        Field(
            description=(
                "Flat list of cfn-lint and/or deployment error strings. "
                "Do NOT include checkov or trivy security-policy findings — "
                "those are handled by a separate policy-context path and do not "
                "benefit from CloudFormation schema retrieval."
            )
        ),
    ]
    prior_queries: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Retrieval queries already used in previous iterations of the "
                "repair loop. Generated queries must cover different "
                "Resource.Property combinations to avoid redundant lookups."
            ),
        ),
    ] = []


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool(args_schema=BuildRetrievalQueriesInput)
def build_retrieval_queries(
    annotated_template: str,
    validation_errors: list[str],
    prior_queries: list[str] = [],
) -> dict:
    """Generate targeted AWS CloudFormation schema-retrieval queries.

    Analyses the annotated CloudFormation template and cfn-lint / deployment
    validation errors to produce a minimal, precise list of
    ``Resource.Property`` schema-lookup queries.

    Call this tool FIRST when cfn-lint or deployment errors are present.
    Pass the returned ``queries`` list to ``hybrid_rag_search`` to fetch the
    official AWS schema documentation needed to fix the violations.

    Returns a JSON-serialisable dict::

        {"queries": ["<query1>", "<query2>", ...]}
    """
    # Deferred imports — keep module-level import cost negligible.
    from agents.llm_client import _build_client, _call_llm_with_history  # noqa: PLC0415
    from agents.retriever import build_retrieval_prompt                   # noqa: PLC0415
    from prompts.retriever_prompt import QUERY_GEN_SYSTEM                 # noqa: PLC0415
    from tools.retriever_helpers import parse_query_response              # noqa: PLC0415

    # build_retrieval_prompt expects a RemediationHistory list for prior-query
    # deduplication.  We synthesise a single synthetic entry when prior_queries
    # is non-empty so the existing pure-function interface is reused unchanged.
    synthetic_history: list[dict] = []
    if prior_queries:
        synthetic_history = [{"iteration": 0, "retrieval_queries": prior_queries}]

    user_content = build_retrieval_prompt(
        errors=validation_errors,
        template_yaml=annotated_template,
        annotation=None,          # annotation only used for resource-type seeding
        remediation_history=synthetic_history,  # type: ignore[arg-type]
    )

    client, model = _build_client()
    raw_response, _ = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )

    queries = parse_query_response(raw_response) or validation_errors[:8]
    print(f"[Tool:build_retrieval_queries] Generated {len(queries)} queries.")
    return {"queries": queries}
