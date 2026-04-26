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

# ---------------------------------------------------------------------------
# Line-number detection
# ---------------------------------------------------------------------------

# Pattern 1: colon-separated line references emitted by some validators
#   e.g. "Resources/Bucket/Type:12:3" or "template.yaml:45"
_COLON_LINE_RE = re.compile(r":\d+(:\d+)?")

# Pattern 2: cfn-lint dict-repr location embedded in the error string
#   e.g. "{'ColumnNumber': 7, 'LineNumber': 115}"
#   cfn-lint serialises its Location namedtuple via str(), producing this format.
_DICT_LINE_RE = re.compile(r"'LineNumber'\s*:\s*\d+")


def _errors_have_line_numbers(errors: list[str]) -> bool:
    """Return True if at least one error string contains a line number reference.

    Handles two formats emitted by different validators:
      - Colon-separated: 'Resources/Bucket/Type:12:3'  (generic validators)
      - Dict-repr:       "{'LineNumber': 115, ...}"     (cfn-lint)

    Deployment and YAML parse errors do not carry line numbers, so annotation
    against line numbers adds no signal when those are the only failures present.
    """
    for e in errors:
        if _COLON_LINE_RE.search(e) or _DICT_LINE_RE.search(e):
            return True
    return False


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
      1. Validation errors.
      2. Annotated template (only when errors have cfn-lint line numbers) or
         plain template fallback.
      3. Prior retrieval-query history from remediation_history, so the LLM
         avoids repeating Resource.Property lookups already covered.

    Pure function — no I/O, no LLM calls, fully unit-testable.
    """
    parts: list[str] = [
        "## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)
    ]

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
      2. Annotate the template ONLY when errors carry line numbers.
      3. Build the single-turn retrieval prompt (pure, no I/O).
      4. Call the LLM to generate targeted schema queries.
      5. Execute ChromaDB (semantic) + Neo4j (graph) hybrid retrieval.
      6. Append this invocation to retriever_history.txt via the recorder.
      7. Return retriever_context and retriever_queries into state.
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    if _errors_have_line_numbers(errors):
        annotation = _annotate_safely(
            template_yaml=state.get("cloudformation_template", ""),
            smell_report=state.get("smell_report"),
        )
        print("[Retriever] Line numbers detected — annotated template included in prompt.")
    else:
        annotation = None
        print("[Retriever] No line numbers in errors — plain template used.")

    user_content = build_retrieval_prompt(
        errors=errors,
        template_yaml=state.get("cloudformation_template"),
        annotation=annotation,
        remediation_history=state.get("remediation_history", []),
    )

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

    seed_resources = extract_resource_types(annotation)
    cfn_context = execute_hybrid_retrieval(
        retrieval_queries=retrieval_queries,
        seed_resources=seed_resources,
    )

    # Append this invocation to retriever_history.txt. The retriever has no
    # rolling conversation history, so the recorder appends a fresh dated block
    # for every call instead of overwriting the file.
    recorder.append_retriever_history_entry(
        iteration=iteration,
        prompt=f"SYSTEM:\n{QUERY_GEN_SYSTEM}\n\nUSER:\n{user_content}",
        response=raw_response,
        retrieval_queries=retrieval_queries,
        context_chars=len(cfn_context),
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
