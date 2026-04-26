from __future__ import annotations

from state import GraphState, RemediationHistory
from agents.llm_client import _build_client, _call_llm_with_history
from tools.template_annotator import (
    TemplateAnnotation,
    annotate_template,
    attach_smells,
    render_annotated_template,
    extract_resource_types,
)
from tools.cfn_hybrid_rag import (
    QUERY_GEN_SYSTEM,
    execute_hybrid_retrieval,
)
from tools.retriever_helpers import extract_errors, parse_query_response
from tracking.recorder import ResearchRecorder


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def _annotate_safely(
    template_yaml: str,
    smell_report: list[dict] | None,
) -> TemplateAnnotation | None:
    """Parse and annotate the current template, attaching any smell report.

    Returns None on parse failure so callers can degrade gracefully rather
    than propagating exceptions through the graph.
    """
    if not template_yaml:
        return None
    try:
        annotation = annotate_template(file_path="<in-memory>", content=template_yaml)
        if smell_report:
            annotation = attach_smells(annotation, smell_report)
        print(f"[Retriever] Annotation: {len(annotation.resources)} resources parsed.")
        return annotation
    except Exception as exc:
        print(f"[Retriever] Annotation failed (non-fatal): {exc}")
        return None


# ---------------------------------------------------------------------------
# Retrieval prompt builder
# ---------------------------------------------------------------------------

def build_retrieval_prompt(
    errors: list[str],
    template_yaml: str | None,
    annotation: TemplateAnnotation | None,
    remediation_history: list[RemediationHistory],
) -> str:
    """Assemble the single user-turn message for the query-generation LLM call.

    Sections (in order):
      1. Validation errors list.
      2. Annotated CloudFormation template with inline # ERROR comments, or a
         plain template snippet as fallback when annotation is unavailable.
      3. Prior retrieval-query history block so the LLM diversifies queries
         across iterations and avoids redundant Resource.Property combinations.

    This function is pure (no I/O, no LLM calls) and can be unit-tested in
    isolation.
    """
    parts: list[str] = [
        "## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)
    ]

    # --- Template block ---
    if annotation and annotation.resources:
        annotated_yaml = render_annotated_template(
            annotation=annotation,
            errors=errors,
            include_security_smells=False,
        )
        parts.append(
            "## Annotated CloudFormation Template\n"
            "Each resource block has inline # ERROR comments showing which errors\n"
            "apply to that specific resource. Use these as the primary signal for\n"
            "which Resource.Property combinations need schema retrieval.\n"
            f"```yaml\n{annotated_yaml}\n```"
        )
    elif template_yaml:
        parts.append(
            "## Template Snippet (for resource type context)\n"
            f"```yaml\n{template_yaml}\n```"
        )

    # --- Prior retrieval history block ---
    history_lines: list[str] = []
    for entry in remediation_history:
        if not entry.get("retrieval_queries"):
            continue
        queries_str = "\n".join(f"  - {q}" for q in entry["retrieval_queries"])
        history_lines.append(f"### Iteration {entry['iteration']} queries used:\n{queries_str}")

    if history_lines:
        parts.append(
            "## Prior Retrieval Queries\n"
            "These queries were already used in previous iterations.\n"
            "Generate DIFFERENT queries that target unexplored Resource.Property combinations.\n"
            "\n" + "\n\n".join(history_lines)
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM query generator
# ---------------------------------------------------------------------------

def _call_query_generator(
    user_content: str,
) -> tuple[str, str, str, dict | None]:
    """Send the retrieval prompt to the LLM and return the raw response.

    Prompt construction lives in build_retrieval_prompt; response parsing
    lives in parse_query_response. This function owns only the LLM call.

    Returns:
        (model, raw_response, user_content, usage)
    """
    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        # Single-turn: full context is in the prompt; no conversation history needed.
        messages=[{"role": "user", "content": user_content}],
    )
    return model, raw_response, user_content, usage


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Dedicated retrieval agent.

    Orchestration steps:
      1. Annotate the current template (safe — non-fatal on parse failure).
      2. Build the HyDE retrieval prompt (pure, no I/O).
      3. Call the LLM to generate targeted schema queries (recorded as an LLM call).
      4. Execute ChromaDB + Neo4j hybrid retrieval.
      5. Return retriever_context and retriever_queries into state.

    Context diversity across iterations is handled by injecting structured
    remediation_history into the prompt rather than replaying full conversation
    turns — keeping the prompt focused and the token budget predictable.
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    # Step 1 — Annotate
    annotation = _annotate_safely(
        template_yaml=state.get("cloudformation_template", ""),
        smell_report=state.get("smell_report"),
    )

    # Step 2 — Build prompt
    user_content = build_retrieval_prompt(
        errors=errors,
        template_yaml=state.get("cloudformation_template"),
        annotation=annotation,
        remediation_history=state.get("remediation_history", []),
    )

    # Step 3 — Call LLM
    model, raw_response, _, usage = _call_query_generator(user_content)
    retrieval_queries = parse_query_response(raw_response) or errors[:8]

    llm_record = recorder.record_llm_call(
        state=state,
        agent="retriever",
        model=model,
        prompt=f"SYSTEM:\n{QUERY_GEN_SYSTEM}\n\nUSER:\n{user_content}",
        response=raw_response,
        token_usage=usage,
    )

    # Step 4 — Hybrid retrieval
    # extract_resource_types() centralises the annotation → resource-type-set
    # conversion that previously appeared here AND inside execute_hybrid_retrieval.
    seed_resources = extract_resource_types(annotation)
    cfn_context = execute_hybrid_retrieval(
        retrieval_queries=retrieval_queries,
        seed_resources=seed_resources,
    )

    print(
        f"[Retriever] Context: {len(cfn_context)} chars, "
        f"{len(retrieval_queries)} queries used."
    )

    return {
        "retriever_context":  cfn_context,
        "retriever_queries":  retrieval_queries,
        "llm_call_log":       state["llm_call_log"] + [llm_record],
    }
