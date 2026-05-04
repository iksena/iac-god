from __future__ import annotations

import re

from state import GraphState, RemediationHistory, Message, append_and_cap
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

# Format produced by validate_cfn_lint:  "[W3005] line 42 | ..."
_WORD_LINE_RE  = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
# Legacy colon-separated format (yamllint, other tools):  ":115" or ":115:7"
_COLON_LINE_RE = re.compile(r":\d+(:\d+)?")
# Defensive fallback for any remaining raw-dict repr (should no longer appear).
_DICT_LINE_RE  = re.compile(r"'LineNumber'\s*:\s*\d+")


def _errors_have_line_numbers(errors: list[str]) -> bool:
    """Return True if at least one error string contains a line number reference.

    Checks three formats in priority order:
      1. Word form:   'line 42'          (cfn-lint via _format_cfn_lint_finding)
      2. Colon form:  ':115' or ':115:7' (yamllint and other validators)
      3. Dict repr:   "{'LineNumber': 42}" (legacy fallback, should not appear)
    """
    for e in errors:
        if _WORD_LINE_RE.search(e) or _COLON_LINE_RE.search(e) or _DICT_LINE_RE.search(e):
            return True
    return False


def _annotate_safely(
    template_yaml: str,
    smell_report: list[dict] | None,
) -> TemplateAnnotation | None:
    """Parse and annotate the template for resource-type seeding.

    Returns None on parse failure so callers degrade gracefully.
    Only used to extract resource types for Neo4j seeding — rendering
    is now done directly against the raw template string.
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
      1. Validation errors — rich format: [RuleId] line N | Resource: X | message | description
      2. Full template with inline ERROR comments at the exact reported lines
         (when errors carry line numbers), or plain template fallback.
      3. Prior retrieval-query history to avoid duplicate lookups.

    Pure function — no I/O, no LLM calls, fully unit-testable.
    """
    parts: list[str] = [
        "## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)
    ]

    if template_yaml and _errors_have_line_numbers(errors):
        annotated = render_annotated_template(
            template_yaml=template_yaml,
            errors=errors,
        )
        parts.append(
            "## CloudFormation Template (errors annotated at reported lines)\n"
            "Lines prefixed with `# ERROR:` mark the exact location cfn-lint\n"
            "reported a violation. Use the Resource name and rule description\n"
            "from the error list above as the primary signal for schema retrieval.\n"
            f"```yaml\n{annotated}\n```"
        )
    elif template_yaml:
        parts.append(
            "## CloudFormation Template\n"
            "No line-number annotations available — use the error messages\n"
            "above to identify which resource types and properties need schema retrieval.\n"
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
    history: list[Message],
) -> tuple[str, str, dict | None]:
    """Send the retrieval prompt to the LLM with conversation history.

    Returns (model, raw_response, token_usage).
    """
    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        messages=history + [{"role": "user", "content": user_content}],
    )
    return model, raw_response, usage


def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Dedicated retrieval agent.

    Orchestration steps:
      1. Extract cfn-lint + deploy errors (security stages excluded).
      2. Annotate the template to seed resource types for Neo4j.
      3. Build the retrieval prompt (pure, no I/O).
      4. Call the LLM (with rolling retriever_history) to generate queries.
      5. Execute ChromaDB (semantic) + Neo4j (graph) hybrid retrieval.
      6. Persist prompt, response, queries, and full schema context to
         retriever_history.txt via the recorder.
      7. Return retriever_context, retriever_queries, and updated
         retriever_history into state.
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    template_yaml = state.get("cloudformation_template", "")
    annotation = _annotate_safely(
        template_yaml=template_yaml,
        smell_report=state.get("smell_report"),
    )

    has_line_numbers = _errors_have_line_numbers(errors)
    print(
        f"[Retriever] {'Line numbers detected — annotated template' : <45} "
        f"{'included' if has_line_numbers else 'NOT included (plain template used)'}."
    )

    user_content = build_retrieval_prompt(
        errors=errors,
        template_yaml=template_yaml,
        annotation=annotation,
        remediation_history=state.get("remediation_history", []),
    )

    model, raw_response, usage = _call_query_generator(
        user_content=user_content,
        history=state.get("retriever_history", []),
    )
    retrieval_queries = parse_query_response(raw_response) or errors[:8]

    user_msg: Message = {"role": "user", "content": user_content}
    assistant_msg: Message = {"role": "assistant", "content": raw_response}

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

    recorder.append_retriever_history_entry(
        iteration=iteration,
        prompt=f"SYSTEM:\n{QUERY_GEN_SYSTEM}\n\nUSER:\n{user_content}",
        response=raw_response,
        retrieval_queries=retrieval_queries,
        context_chars=len(cfn_context),
        retrieved_context=cfn_context,
    )

    print(
        f"[Retriever] Context: {len(cfn_context)} chars, "
        f"{len(retrieval_queries)} queries used."
    )

    return {
        "retriever_context":  cfn_context,
        "retriever_queries":  retrieval_queries,
        "llm_call_log":       state["llm_call_log"] + [llm_record],
        "retriever_history":  append_and_cap(
            state.get("retriever_history", []), user_msg, assistant_msg
        ),
    }
