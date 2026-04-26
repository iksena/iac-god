from __future__ import annotations

from state import GraphState, RemediationHistory
from agents.llm_client import _build_client, _call_llm_with_history
from tools.template_annotator import (
    TemplateAnnotation,
    annotate_template,
    attach_smells,
    render_annotated_template,
)
from tools.cfn_hybrid_rag import (
    QUERY_GEN_SYSTEM,
    _execute_hybrid_retrieval,
)
from tools.retriever_helpers import _extract_errors, _parse_query_response
from tracking.recorder import ResearchRecorder


def _build_retriever_history_context(remediation_history: list[RemediationHistory]) -> str:
    """Build a compact history block so the retriever diversifies its queries.

    Surfaces which retrieval queries were used in prior iterations, allowing
    the LLM to avoid redundant queries and explore different
    Resource.Property facets. This replaces passing retriever_history
    conversation turns to the LLM.
    """
    if not remediation_history:
        return ""

    lines: list[str] = [
        "## Prior Retrieval Queries",
        "These queries were already used in previous iterations.",
        "Generate DIFFERENT queries that target unexplored Resource.Property combinations.",
        "",
    ]

    for entry in remediation_history:
        if not entry.get("retrieval_queries"):
            continue
        queries_str = "\n".join(f"  - {q}" for q in entry["retrieval_queries"])
        lines.append(f"### Iteration {entry['iteration']} queries used:\n{queries_str}")
        lines.append("")

    return "\n".join(lines).strip()


def _build_retriever_user_content(
    errors: list[str],
    template_yaml: str | None,
    annotation: TemplateAnnotation | None,
    remediation_history: list[RemediationHistory],
) -> str:
    """Assemble the user-turn content for the retrieval query-generation call.

    Injects an annotated CFN YAML with inline # ERROR comments anchored to
    each resource, replacing the plain annotation summary.
    Falls back to the plain template snippet when annotation fails.
    """
    user_parts = ["## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)]

    if annotation and annotation.resources:
        annotated_yaml = render_annotated_template(
            annotation=annotation,
            errors=errors,
            include_security_smells=False,
        )
        user_parts.append(
            "## Annotated CloudFormation Template\n"
            "Each resource block has inline # ERROR comments showing which errors\n"
            "apply to that specific resource. Use these as the primary signal for\n"
            "which Resource.Property combinations need schema retrieval.\n"
            f"```yaml\n{annotated_yaml}\n```"
        )
    elif template_yaml:
        user_parts.append(
            f"## Template Snippet (for resource type context)\n"
            f"```yaml\n{template_yaml}\n```"
        )

    history_context = _build_retriever_history_context(remediation_history)
    if history_context:
        user_parts.append(history_context)

    return "\n\n".join(user_parts)


def _generate_retrieval_queries(
    errors: list[str],
    template_yaml: str | None,
    annotation: TemplateAnnotation | None,
    remediation_history: list[RemediationHistory],
) -> tuple[str, str, str, list[str], dict | None]:
    """Build the HyDE prompt, call the LLM, and parse retrieval queries.

    Returns:
        (model, user_content, raw_response, retrieval_queries, usage)

    Uses structured remediation history (not conversation turns) to guide
    query diversity across iterations.
    """
    user_content = _build_retriever_user_content(
        errors=errors,
        template_yaml=template_yaml,
        annotation=annotation,
        remediation_history=remediation_history,
    )

    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        # Single-turn: no conversation history — full context is in the prompt.
        messages=[{"role": "user", "content": user_content}],
    )
    retrieval_queries = _parse_query_response(raw_response)
    if not retrieval_queries:
        retrieval_queries = errors[:8]

    return model, user_content, raw_response, retrieval_queries, usage


def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Dedicated retrieval agent.

    1. Annotates the current template.
    2. Uses LLM to generate HyDE retrieval queries (recorded as an LLM call)
       — informed by structured remediation_history, NOT conversation turns.
    3. Executes ChromaDB + Neo4j retrieval.
    4. Returns cfn_context and retrieval_queries into state.
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = _extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    annotation: TemplateAnnotation | None = None
    template_yaml = state.get("cloudformation_template", "")
    if template_yaml:
        try:
            annotation = annotate_template(
                file_path="<in-memory>",
                content=template_yaml,
            )
            smell_report = state.get("smell_report")
            if smell_report:
                annotation = attach_smells(annotation, smell_report)
            print(f"[Retriever] Annotation: {len(annotation.resources)} resources parsed.")
        except Exception as exc:
            print(f"[Retriever] Annotation failed (non-fatal): {exc}")
            annotation = None

    model, user_content, raw_response, retrieval_queries, usage = _generate_retrieval_queries(
        errors=errors,
        template_yaml=template_yaml,
        annotation=annotation,
        remediation_history=state.get("remediation_history", []),
    )

    llm_record = recorder.record_llm_call(
        state=state,
        agent="retriever",
        model=model,
        prompt=f"SYSTEM:\n{QUERY_GEN_SYSTEM}\n\nUSER:\n{user_content}",
        response=raw_response,
        token_usage=usage,
    )

    cfn_context = _execute_hybrid_retrieval(
        retrieval_queries=retrieval_queries,
        annotation=annotation,
    )

    print(
        f"[Retriever] Context: {len(cfn_context)} chars, "
        f"{len(retrieval_queries)} queries used."
    )

    return {
        "retriever_context": cfn_context,
        "retriever_queries": retrieval_queries,
        "llm_call_log": state["llm_call_log"] + [llm_record],
    }