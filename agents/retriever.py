from __future__ import annotations

import re

from state import GraphState, RemediationHistory
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.retriever_prompt import QUERY_GEN_SYSTEM
from tools.template_annotator import (
    TemplateAnnotation,
    annotate_template,
    attach_smells,
    render_annotated_template,
    extract_resource_types,
)
from tools.cfn_hybrid_rag import execute_hybrid_retrieval
from tools.retriever_helpers import extract_errors, parse_query_response
from tracking.recorder import ResearchRecorder

# Matches cfn-lint line references such as ":12" or ":12:3" embedded in error strings.
_LINE_REF_RE = re.compile(r":\d+(:\d+)?")


def _errors_have_line_numbers(errors: list[str]) -> bool:
    """Return True if at least one error string contains a line:col reference.

    cfn-lint errors embed line numbers (e.g. 'Resources/Bucket/Type:12:3').
    Deployment and YAML errors do not — annotation against line numbers adds
    no signal when those are the only failures present.
    """
    return any(_LINE_REF_RE.search(e) for e in errors)


def _annotate_safely(
    template_yaml: str,
    smell_report: list[dict] | None,
) -> TemplateAnnotation | None:
    """Parse and annotate the template, attaching any smell report.

    Returns None on parse failure so callers degrade gracefully rather than
    propagating exceptions through the graph.
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


def build_retrieval_prompt(
    errors: list[str],
    template_yaml: str | None,
    annotation: TemplateAnnotation | None,
    remediation_history: list[RemediationHistory],
) -> str:
    """Assemble the single user-turn message for the query-generation LLM call.

    Sections (in order):
      1. Validation errors (cfn-lint + deployment; security stages already filtered
         by extract_errors()).
      2. Annotated CloudFormation template — ONLY included when errors carry line
         numbers (cfn-lint), because annotation anchors errors to specific lines.
         Falls back to a plain template snippet when no line numbers are present.
      3. Prior retrieval-query history so the LLM generates diverse queries across
         iterations and avoids redundant Resource.Property lookups.

    Pure function — no I/O, no LLM calls, fully unit-testable.
    """
    parts: list[str] = [
        "## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)
    ]

    # Template block: annotated view only when errors have line numbers.
    if annotation and annotation.resources and _errors_have_line_numbers(errors):
        annotated_yaml = render_annotated_template(
            annotation=annotation,
            errors=errors,
            include_security_smells=False,
        )
        parts.append(
            "## Annotated CloudFormation Template\n"
            "Each resource block carries inline `# ERROR:` comments anchored to\n"
            "the exact line where cfn-lint found a violation. Use these as the\n"
            "primary signal for which Resource.Property pairs need schema retrieval.\n"
            f"```yaml\n{annotated_yaml}\n```"
        )
    elif template_yaml:
        parts.append(
            "## CloudFormation Template (no line-number annotations available)\n"
            "Use the error messages above to identify which resource types and\n"
            "properties need schema retrieval.\n"
            f"```yaml\n{template_yaml}\n```"
        )

    # Prior retrieval history block.
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
            "Generate DIFFERENT queries targeting unexplored Resource.Property combinations.\n"
            "\n" + "\n\n".join(history_lines)
        )

    return "\n\n".join(parts)


def _call_query_generator(
    user_content: str,
) -> tuple[str, str, str, dict | None]:
    """Send the retrieval prompt to the LLM and return the raw response.

    Returns: (model, raw_response, user_content, usage)
    """
    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return model, raw_response, user_content, usage


def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Dedicated retrieval agent.

    Orchestration steps:
      1. Extract cfn-lint + deploy errors (security stages excluded by extract_errors).
      2. Annotate the template ONLY when errors carry line numbers — cfn-lint
         embeds line:col refs; deploy/YAML errors do not.
      3. Build the single-turn retrieval prompt (pure, no I/O).
      4. Call the LLM to generate targeted schema queries.
      5. Execute ChromaDB (semantic) + Neo4j (graph) hybrid retrieval.
      6. Return retriever_context and retriever_queries into state.
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    # Step 2 — Annotate only when at least one error carries a line number.
    if _errors_have_line_numbers(errors):
        annotation = _annotate_safely(
            template_yaml=state.get("cloudformation_template", ""),
            smell_report=state.get("smell_report"),
        )
        print("[Retriever] Line numbers detected — annotated template will be included in prompt.")
    else:
        annotation = None
        print("[Retriever] No line numbers in errors — skipping annotation; plain template used.")

    # Step 3 — Build prompt
    user_content = build_retrieval_prompt(
        errors=errors,
        template_yaml=state.get("cloudformation_template"),
        annotation=annotation,
        remediation_history=state.get("remediation_history", []),
    )

    # Step 4 — Call LLM
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

    # Step 5 — Hybrid retrieval
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
        "retriever_context": cfn_context,
        "retriever_queries": retrieval_queries,
        "llm_call_log":      state["llm_call_log"] + [llm_record],
    }
