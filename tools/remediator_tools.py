"""tools/remediator_tools.py

Legacy tool definitions for the Remediator agent.

NOTE: `build_retrieval_queries` is intentionally NOT registered as a
LangChain @tool and NOT bound to any LLM in the current architecture.
The Remediator now calls `retrieve_schema_context` (tools/retriever_tools.py)
directly — the LLM generates its own retrieval queries as arguments to that
tool rather than through a separate query-generation tool call.

This module is kept for reference and potential future use.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Legacy: build_retrieval_queries (NOT a LangChain tool — do not bind to LLM)
# ---------------------------------------------------------------------------

def build_retrieval_queries(
    annotated_template: str,
    validation_errors: list[str],
    prior_queries: list[str] | None = None,
) -> dict:
    """Generate targeted AWS CloudFormation schema-retrieval queries.

    LEGACY — not called by any agent in the current architecture.
    Kept as a plain Python function for reference only.

    Analyses the annotated CloudFormation template and cfn-lint / deployment
    validation errors to produce a minimal, precise list of
    ``Resource.Property`` schema-lookup queries.

    Returns a dict::

        {"queries": ["<query1>", "<query2>", ...]}
    """
    if prior_queries is None:
        prior_queries = []

    # Deferred imports — keep module-level import cost negligible.
    from agents.llm_client import _build_client, _call_llm_with_history  # noqa: PLC0415
    from agents.retriever import build_retrieval_prompt                   # noqa: PLC0415
    from prompts.retriever_prompt import QUERY_GEN_SYSTEM                 # noqa: PLC0415
    from tools.retriever_helpers import parse_query_response              # noqa: PLC0415

    synthetic_history: list[dict] = []
    if prior_queries:
        synthetic_history = [{"iteration": 0, "retrieval_queries": prior_queries}]

    user_content = build_retrieval_prompt(
        errors=validation_errors,
        template_yaml=annotated_template,
        annotation=None,
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
    print(f"[build_retrieval_queries] Generated {len(queries)} queries.")
    return {"queries": queries}
