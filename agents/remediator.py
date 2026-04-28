from __future__ import annotations

import re
import json
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history, _build_langchain_chat_model
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context
from tools.remediator_tools import build_retrieval_queries
from tools.retriever_helpers import (
    get_latest_stage_result,
    format_cfn_lint_errors,
    format_deploy_errors,
    extract_errors,
)
from tools.template_annotator import render_annotated_template
from tracking.recorder import ResearchRecorder


# ---------------------------------------------------------------------------
# Internal helpers — error extraction
# ---------------------------------------------------------------------------

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _extract_check_ids_from_errors(errors: list[object]) -> set[str]:
    check_ids: set[str] = set()
    for error in errors:
        text = str(error or "")
        for match in re.findall(r"\[([A-Z0-9_-]+)\]", text):
            check_ids.add(match.strip().upper())
        for match in re.findall(r"\b(?:AVD-)?AWS-\d{4}\b", text, flags=re.IGNORECASE):
            check_ids.add(match.strip().upper())
        for match in re.findall(r"\bCKV2?_[A-Z0-9_]+\b", text, flags=re.IGNORECASE):
            check_ids.add(match.strip().upper())
    return check_ids


# ---------------------------------------------------------------------------
# Policy source context builders (Checkov / Trivy)
# ---------------------------------------------------------------------------

def _extract_security_findings(
    validation_results: list[dict],
    stage: str,
    results_key: str,
    items_path: list[str],
) -> list[dict[str, str]]:
    """Extract check-ID findings from a security tool's raw JSON output."""
    findings: list[dict[str, str]] = []
    result = get_latest_stage_result(validation_results, stage)
    if not result:
        return findings

    raw = result.get("raw_output", "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings

    top = data.get(results_key, {})

    if isinstance(top, dict):
        for item in top.get(items_path[0], []):
            check_id = str(item.get("check_id") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})

    elif isinstance(top, list):
        for entry in top:
            for item in entry.get(items_path[0], []):
                check_id = str(item.get("ID") or "").strip()
                if check_id:
                    findings.append({"check_id": check_id})

    return findings


def _filter_findings_by_check_ids(
    findings: list[dict[str, str]],
    allowed_check_ids: set[str],
) -> list[dict[str, str]]:
    if not allowed_check_ids:
        return findings

    allowed = {cid.strip().upper() for cid in allowed_check_ids if cid}
    seen: set[str] = set()
    filtered: list[dict[str, str]] = []

    for finding in findings:
        check_id = str(finding.get("check_id") or finding.get("rule_id") or "").strip()
        if not check_id:
            continue
        normalized = check_id.upper()
        if normalized not in allowed or normalized in seen:
            continue
        seen.add(normalized)
        filtered.append({"check_id": check_id})

    return filtered


def _build_policy_source_context(validation_results: list[dict]) -> str:
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    allowed_check_ids: set[str] = set()
    for result in latest_by_stage.values():
        allowed_check_ids.update(_extract_check_ids_from_errors(result.get("errors", [])))

    if not allowed_check_ids:
        return ""

    checkov_findings = _extract_security_findings(
        validation_results,
        stage="checkov",
        results_key="results",
        items_path=["failed_checks"],
    )
    trivy_findings = _extract_security_findings(
        validation_results,
        stage="trivy",
        results_key="Results",
        items_path=["Misconfigurations"],
    )

    checkov_context = get_checkov_policy_context(
        _filter_findings_by_check_ids(checkov_findings, allowed_check_ids)
    )
    trivy_context = get_trivy_policy_context(
        _filter_findings_by_check_ids(trivy_findings, allowed_check_ids)
    )

    sections: list[str] = []
    if checkov_context:
        sections.append(f"### Checkov Policy Source\n{checkov_context}")
    if trivy_context:
        sections.append(f"### Trivy Policy Source\n{trivy_context}")
    if not sections:
        return ""

    return "## Relevant Policy Source Context (Checkov/Trivy)\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Validation error section builder
# ---------------------------------------------------------------------------

def _build_validation_errors_text(state: GraphState) -> str:
    """Build the full validation error section for the remediator user prompt."""
    error_blocks: list[str] = []

    validation_results = state.get("validation_results", [])
    latest_by_stage: dict[str, dict] = {}
    for result in validation_results:
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    for result in latest_by_stage.values():
        if result["passed"]:
            continue
        deduped = _dedupe_preserve_order(
            [str(e) for e in result.get("errors", []) if str(e).strip()]
        )
        if not deduped:
            continue

        stage = result["stage"]
        if stage == "cfn-lint":
            errors_text = format_cfn_lint_errors(deduped)
        else:
            errors_text = "\n".join(f"  - {e}" for e in deduped)

        error_blocks.append(f"### {stage.upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if (
        deploy_result
        and not deploy_result["passed"]
        and deploy_result["target"] != "skipped"
    ):
        error_blocks.append(
            f"### DEPLOYABILITY Errors\n{format_deploy_errors(deploy_result)}"
        )

    return "\n\n".join(error_blocks) if error_blocks else "No validation errors reported."


# ---------------------------------------------------------------------------
# Context inclusion guards
# ---------------------------------------------------------------------------

def should_include_remediation_context(state: GraphState) -> bool:
    """Return True only when cfn-lint or deployment failures are present.

    YAML parse errors are excluded: they indicate malformed YAML syntax, not
    CloudFormation schema violations, so schema RAG context provides no
    actionable signal — the LLM just needs to fix the YAML structure.

    Security-only failures (checkov, trivy) are also excluded — they are
    handled by the policy source context path instead.
    """
    validation_results = state.get("validation_results", [])
    cfn_lint_result = get_latest_stage_result(validation_results, "cfn-lint")

    if cfn_lint_result and not cfn_lint_result.get("passed", True):
        return True

    deploy_result = state.get("deploy_validation_result")
    if (
        deploy_result
        and not deploy_result.get("passed", True)
        and deploy_result.get("target") != "skipped"
    ):
        return True

    return False


def _should_include_policy_source_context(state: GraphState) -> bool:
    validation_results = state.get("validation_results", [])
    for stage in ("trivy", "checkov"):
        result = get_latest_stage_result(validation_results, stage)
        if result and not result.get("passed", True):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool-call result extraction helpers
# ---------------------------------------------------------------------------

_TOOLS = [build_retrieval_queries]
_TOOL_NODE = ToolNode(_TOOLS)
_REMEDIATOR_LLM = None  # lazy-initialised to avoid import-time side effects


def _get_bound_llm():
    """Lazy-init the LangChain model with build_retrieval_queries bound.

    Deferred so that heavy LangChain imports and credential checks only
    happen when the remediator is actually invoked, not at module import.
    """
    global _REMEDIATOR_LLM  # noqa: PLW0603
    if _REMEDIATOR_LLM is None:
        _REMEDIATOR_LLM = _build_langchain_chat_model().bind_tools(_TOOLS)
    return _REMEDIATOR_LLM


def _extract_tool_queries(messages: list) -> list[str]:
    """Pull retrieval queries produced by build_retrieval_queries ToolMessages.

    Parses the JSON content of every ToolMessage whose name matches
    'build_retrieval_queries' and collects the 'queries' list.
    """
    queries: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "build_retrieval_queries":
            continue
        try:
            result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if isinstance(result, dict):
                queries.extend(result.get("queries", []))
        except (json.JSONDecodeError, TypeError):
            pass
    return queries


def _run_tool_loop(
    system_msg: SystemMessage,
    initial_messages: list,
) -> tuple[str, list, list[str]]:
    """Run the LLM → ToolNode agentic loop.

    The LLM is invoked up to MAX_TOOL_ROUNDS times. Each round:
      1. Call the LLM with [system_msg] + accumulated messages.
      2. If the response contains tool_calls, execute them via ToolNode
         and append the ToolMessages to the conversation.
      3. If the response has no tool_calls, break — final answer ready.

    Returns:
        final_content  : The text of the last AIMessage (the fix objectives).
        all_messages   : Full message list including tool results (for audit).
        retrieval_queries: Queries extracted from build_retrieval_queries calls.
    """
    MAX_TOOL_ROUNDS = 2
    llm = _get_bound_llm()
    messages = list(initial_messages)

    ai_msg: AIMessage | None = None
    for round_idx in range(MAX_TOOL_ROUNDS):
        ai_msg = llm.invoke([system_msg] + messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            print(
                f"[Remediator] Tool loop finished after {round_idx} tool round(s). "
                "Proceeding to final answer."
            )
            break

        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        print(f"[Remediator] Tool round {round_idx + 1}: calling {tool_names}")

        tool_result = _TOOL_NODE.invoke({"messages": messages})
        # ToolNode returns {"messages": [*existing*, *new_tool_msgs*]}
        # Slice off only the newly appended ToolMessages.
        new_tool_msgs = tool_result["messages"][len(messages):]
        messages.extend(new_tool_msgs)
    else:
        # Exhausted MAX_TOOL_ROUNDS — do one final call without tools available
        # so the LLM is forced to produce its text answer.
        print(
            f"[Remediator] Reached MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}). "
            "Forcing final answer call."
        )
        final_llm = _build_langchain_chat_model()  # unbound — no tools
        ai_msg = final_llm.invoke([system_msg] + messages)
        messages.append(ai_msg)

    final_content = ai_msg.content if ai_msg else ""
    if isinstance(final_content, list):
        # Some providers return a list of content blocks
        final_content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in final_content
        ).strip()

    retrieval_queries = _extract_tool_queries(messages)
    return str(final_content), messages, retrieval_queries


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])),
    )

    include_remediation = should_include_remediation_context(state)
    include_policy      = _should_include_policy_source_context(state)

    policy_source_context = (
        _build_policy_source_context(state["validation_results"])
        if include_policy
        else ""
    )

    flat_errors = extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )
    annotated_template = render_annotated_template(
        template_yaml=state.get("cloudformation_template", ""),
        errors=flat_errors,
    )
    formatted_errors = _build_validation_errors_text(state)

    # Prior retrieval queries across all previous iterations — passed to the
    # tool so it can generate non-duplicate queries.
    prior_queries: list[str] = []
    for entry in state.get("remediation_history", []):
        prior_queries.extend(entry.get("retrieval_queries", []))

    # Build the REMEDIATOR_USER content.
    # cfn_graph_context is intentionally left blank here: when RAG context
    # is needed the LLM will obtain it by calling build_retrieval_queries
    # (and, in a future iteration, hybrid_rag_search) via the ToolNode.
    # The state's retriever_context from the legacy retriever path is still
    # forwarded as a fallback for callers that pre-populated it.
    cfn_graph_context = state.get("retriever_context", "") if include_remediation else ""

    if include_remediation:
        print(
            f"[Remediator] RAG tool available. Prior queries: {len(prior_queries)}. "
            f"Legacy context: {len(cfn_graph_context)} chars."
        )
    else:
        print("[Remediator] CFN schema context skipped (YAML/security-only failure).")

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        annotated_template=annotated_template,
        validation_errors=formatted_errors,
        policy_source_context=policy_source_context,
        cfn_graph_context=cfn_graph_context,
        remediation_history_context="",
    )

    # -----------------------------------------------------------------------
    # Build system message with tool-usage policy injected when RAG is useful.
    # -----------------------------------------------------------------------
    tool_policy = ""
    if include_remediation:
        prior_q_block = (
            "\n".join(f"  - {q}" for q in prior_queries)
            if prior_queries
            else "  (none yet)"
        )
        tool_policy = (
            "\n## Tool Usage Policy\n"
            "You have access to `build_retrieval_queries`.\n"
            "Call it when cfn-lint or deployment errors are present to generate "
            "targeted AWS CloudFormation schema-lookup queries.\n"
            "Pass the annotated template and the flat list of cfn-lint/deploy "
            "error strings as arguments.\n"
            f"Prior retrieval queries (DO NOT repeat these):\n{prior_q_block}\n"
            "Do NOT call the tool for YAML syntax errors or pure security "
            "violations (checkov/trivy IDs).\n"
            "After receiving the tool result, use the queries to ground your "
            "Root Cause Analysis and Fix Objectives.\n"
        )

    system_with_policy = system + tool_policy
    system_msg = SystemMessage(content=system_with_policy)

    # -----------------------------------------------------------------------
    # Seed the message thread with remediator_history (rolling window) + user.
    # -----------------------------------------------------------------------
    # remediator_history stores {"role": ..., "content": ...} dicts;
    # convert to LangChain message objects for the ToolNode loop.
    lc_history: list = []
    for msg in state.get("remediator_history", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content))

    user_lc_msg = HumanMessage(content=user_content)
    initial_messages = lc_history + [user_lc_msg]

    # -----------------------------------------------------------------------
    # Run the ToolNode agentic loop.
    # -----------------------------------------------------------------------
    content, all_messages, retrieval_queries_from_tool = _run_tool_loop(
        system_msg=system_msg,
        initial_messages=initial_messages,
    )

    # Merge tool-produced queries with any legacy queries from state.
    retrieval_queries = retrieval_queries_from_tool or state.get("retriever_queries", [])

    # -----------------------------------------------------------------------
    # Audit logging via recorder.
    # -----------------------------------------------------------------------
    model = DEFAULT_CONFIG.model
    llm_record = recorder.record_llm_call(
        state=state,
        agent="remediator",
        model=model,
        prompt=f"SYSTEM:\n{system_with_policy}\n\nUSER:\n{user_content}",
        response=content,
        token_usage={},  # LangChain path; token counts available via callbacks if needed
    )

    # -----------------------------------------------------------------------
    # Persist state.
    # -----------------------------------------------------------------------
    new_history_entry: RemediationHistory = {
        "iteration":         iteration,
        "errors":            state["validation_results"],
        "flat_errors":       flat_errors,
        "formatted_errors":  formatted_errors,
        "suggestion":        content,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "cfn_context":       cfn_graph_context,
        "retrieval_queries": retrieval_queries,
    }

    # Store history as plain dicts (existing Message type) for compatibility
    # with the rest of the pipeline.
    user_msg: Message = {"role": "user",      "content": user_content}
    assistant_msg: Message = {"role": "assistant", "content": content}

    print("[Remediator] Suggestions generated. Routing back to Engineer.")
    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "current_iteration":   iteration + 1,
        "llm_call_log":        state["llm_call_log"] + [llm_record],
        "remediator_history":  append_and_cap(
            state["remediator_history"], user_msg, assistant_msg
        ),
        "retriever_queries":   retrieval_queries,
    }


# Avoid unused import warning — DEFAULT_CONFIG is used for model name in audit log.
from config import DEFAULT_CONFIG  # noqa: E402
