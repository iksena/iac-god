from __future__ import annotations

import re
import json
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_langchain_chat_model
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context
from tools.retriever_tools import retrieve_schema_context
from tools.retriever_helpers import (
    get_latest_stage_result,
    format_cfn_lint_errors,
    format_deploy_errors,
    extract_errors,
)
from tools.template_annotator import render_annotated_template
from tracking.recorder import ResearchRecorder


# ---------------------------------------------------------------------------
# Reasoning block helpers
# ---------------------------------------------------------------------------

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


def extract_reasoning_block(text: str) -> str:
    """Return the content inside the first <reasoning>...</reasoning> block, or ''."""
    match = _REASONING_RE.search(text)
    return match.group(1).strip() if match else ""


def strip_reasoning_block(text: str) -> str:
    """Remove all <reasoning>...</reasoning> blocks and return the clean text."""
    return _REASONING_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Internal helpers - error extraction
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

def _should_include_policy_source_context(state: GraphState) -> bool:
    validation_results = state.get("validation_results", [])
    for stage in ("trivy", "checkov"):
        result = get_latest_stage_result(validation_results, stage)
        if result and not result.get("passed", True):
            return True
    return False


# ---------------------------------------------------------------------------
# LLM + ToolNode setup
# ---------------------------------------------------------------------------

# retrieve_schema_context is the only tool bound to the Remediator LLM.
_TOOLS = [retrieve_schema_context]
_TOOL_NODE = ToolNode(_TOOLS)
_REMEDIATOR_LLM = None  # lazy-initialised


def _get_bound_llm():
    """Lazy-init the LangChain model with retrieve_schema_context bound."""
    global _REMEDIATOR_LLM  # noqa: PLW0603
    if _REMEDIATOR_LLM is None:
        _REMEDIATOR_LLM = _build_langchain_chat_model().bind_tools(_TOOLS)
    return _REMEDIATOR_LLM


def _extract_tool_context(messages: list) -> str:
    """Collect the schema context string returned by retrieve_schema_context ToolMessages."""
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "retrieve_schema_context":
            continue
        if isinstance(msg.content, str) and msg.content:
            return msg.content
    return ""


def _extract_retrieval_queries(messages: list) -> list[str]:
    """Pull the retrieval_queries argument from retrieve_schema_context AIMessage tool_calls."""
    queries: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "retrieve_schema_context":
                args = tc.get("args", {})
                queries.extend(args.get("retrieval_queries", []))
    return queries


def _extract_tool_call_args(ai_msg: AIMessage) -> dict:
    """Return the args dict of the first retrieve_schema_context tool call, or {}."""
    for tc in getattr(ai_msg, "tool_calls", None) or []:
        if tc.get("name") == "retrieve_schema_context":
            return tc.get("args", {})
    return {}


def _ai_content_str(ai_msg: AIMessage) -> str:
    """Safely coerce AIMessage.content to a plain string."""
    content = ai_msg.content if ai_msg else ""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content)


def _run_tool_loop(
    system_msg: SystemMessage,
    initial_messages: list,
    state: GraphState,
    recorder: ResearchRecorder,
) -> tuple[str, str, list, str, list[str], str]:
    """Run the Remediator LLM -> ToolNode agentic loop.

    Flow per round:
      1. Call the bound LLM with [system] + accumulated messages.
      2. If the AIMessage contains tool_calls for retrieve_schema_context:
         a. Extract pre-tool <reasoning> and call args for audit recording.
         b. Execute via ToolNode; append resulting ToolMessages to the list.
         c. Record the full tool call via recorder.record_rag_tool_call().
         d. Loop back - LLM now sees the ToolMessage result and produces
            the final RCA + Fix Objectives on the next round.
      3. If no tool_calls in the AIMessage, the loop ends with the final answer.

    The local `messages` list accumulates ALL turns including ToolMessages.
    It is intentionally NOT persisted to remediator_history - only the clean
    final user/assistant pair is stored there, ensuring _build_lc_history
    never replays tool-call AIMessages into a future LLM call (which would
    break LangChain message validation requiring alternating ToolCall/ToolResult
    pairs).

    Returns:
        clean_content        : Final answer with <reasoning> stripped (for Engineer).
        raw_content          : Final answer as-is, reasoning included (for audit).
        all_messages         : Full accumulated message list (for audit only).
        tool_context         : Schema context string from the ToolMessage, or ''.
        retrieval_queries    : All queries the LLM passed to the tool.
        tool_message_content : Raw ToolMessage content string, identical to
                               tool_context but named explicitly so callers can
                               use it to reconstruct the LLM's full context window
                               for audit recording.
    """
    MAX_TOOL_ROUNDS = 3
    llm = _get_bound_llm()
    messages = list(initial_messages)

    ai_msg: AIMessage | None = None
    for round_idx in range(MAX_TOOL_ROUNDS):
        ai_msg = llm.invoke([system_msg] + messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            # No tool call: LLM produced its final answer (either directly or
            # after receiving ToolMessage context from a prior round).
            print(
                f"[Remediator] Tool loop finished after {round_idx} tool round(s). "
                "Generating RCA & Fix Objectives."
            )
            break

        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        print(f"[Remediator] Tool round {round_idx + 1}: calling {tool_names}")

        # Extract pre-tool reasoning and call args for recording.
        raw_ai_str = _ai_content_str(ai_msg)
        pre_tool_reasoning = extract_reasoning_block(raw_ai_str)
        tool_args = _extract_tool_call_args(ai_msg)
        queries_for_this_round = tool_args.get("retrieval_queries", [])
        seed_resources_for_this_round = tool_args.get("seed_resources", [])

        # Execute the tool via ToolNode.
        tool_result = _TOOL_NODE.invoke({"messages": messages})
        new_tool_msgs = tool_result["messages"][len(messages):]
        messages.extend(new_tool_msgs)

        context_this_round = _extract_tool_context(new_tool_msgs)

        # Record the full RAG tool call for audit.
        recorder.record_rag_tool_call(
            state=state,
            agent="remediator",
            retrieval_queries=queries_for_this_round,
            seed_resources=seed_resources_for_this_round,
            context_returned=context_this_round,
            reasoning_block=pre_tool_reasoning,
            raw_ai_response=raw_ai_str,
            round_idx=round_idx,
        )

        # ToolMessages are now appended; loop back so the LLM sees the schema
        # context and can produce its final RCA + Fix Objectives.

    else:
        # Exhausted MAX_TOOL_ROUNDS without a clean answer - force a final call
        # on an unbound model so no further tool calls are possible.
        print(
            f"[Remediator] Reached MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}). "
            "Forcing final answer with unbound model."
        )
        final_llm = _build_langchain_chat_model()  # unbound - no tools
        ai_msg = final_llm.invoke([system_msg] + messages)
        messages.append(ai_msg)

    raw_content = _ai_content_str(ai_msg)
    clean_content = strip_reasoning_block(raw_content)

    tool_context = _extract_tool_context(messages)
    retrieval_queries = _extract_retrieval_queries(messages)
    # FIX 2: surface tool_message_content as an explicit return value so
    # _build_return_state can include it verbatim in the recorded prompt,
    # faithfully representing what the LLM saw when producing its final answer.
    tool_message_content = tool_context
    return clean_content, raw_content, messages, tool_context, retrieval_queries, tool_message_content


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """
    Remediator agent with inline RAG tool calling.

    The LLM receives the annotated template and validation errors via the user
    prompt. It has `retrieve_schema_context` bound as a tool.

    Execution flow:
      1. LLM is invoked with system + conversation history + current user message.
      2. If cfn-lint or deployment errors are present, the LLM generates targeted
         Resource.Property queries and extracts the relevant resource types from
         the annotated template, then calls retrieve_schema_context.
      3. The ToolNode executes the hybrid RAG pipeline (ChromaDB + Neo4j) and
         returns the official AWS schema context as a ToolMessage.
      4. The LLM is invoked again with the ToolMessage in context and produces
         the final RCA + Fix Objectives.
      5. If only YAML syntax or security-only errors are present, the LLM skips
         the tool call and produces fix objectives directly.

    Schema context never appears in the Engineer's prompt - it is distilled into
    the remediation suggestion the Engineer receives.

    The <reasoning> block is stripped before storing in remediator_history so
    the Engineer never sees internal deliberation. It is recorded separately for
    audit in rag_tool_calls.jsonl and rag_tool_data_flow.txt.
    """
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])),
    )

    include_policy = _should_include_policy_source_context(state)
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

    # Collect prior retrieval queries so the LLM is told not to repeat them.
    prior_queries: list[str] = []
    for entry in state.get("remediation_history", []):
        prior_queries.extend(entry.get("retrieval_queries", []))

    prior_q_block = (
        "\n".join(f"  - {q}" for q in prior_queries)
        if prior_queries
        else "  (none yet)"
    )

    # Tool guidance is appended to the system prompt so the LLM knows:
    # - when to call retrieve_schema_context (cfn-lint / deploy errors)
    # - what arguments to pass:
    #     retrieval_queries : Resource.Property strings for ChromaDB
    #     seed_resources    : AWS resource types (not the full YAML) for Neo4j
    # - which queries were already used (to avoid redundant retrieval)
    # - when NOT to call it (YAML syntax / pure security violations)
    tool_guidance = (
        "\n## Tool Usage\n"
        "You have access to `retrieve_schema_context`.\n"
        "Call it when cfn-lint or deployment errors are present. Pass:\n"
        "  - `retrieval_queries`: targeted Resource.Property queries you generate "
        "based on the annotated template and errors.\n"
        "  - `seed_resources`: list of AWS CloudFormation resource type strings "
        "related to the current errors (e.g. ['AWS::S3::Bucket', 'AWS::IAM::Role']). "
        "Extract these from the annotated template above - do NOT copy the full YAML.\n"
        "The tool runs hybrid RAG (ChromaDB + Neo4j) and returns official AWS "
        "CloudFormation schema context.\n"
        "Use that context to write accurate, schema-correct Fix Objectives.\n"
        f"Prior retrieval queries already used (DO NOT repeat these):\n{prior_q_block}\n"
        "Do NOT call the tool for YAML syntax errors or pure security violations "
        "(checkov/trivy IDs only) - those do not benefit from schema context.\n"
        "After the tool returns its context, produce the final RCA & Fix Objectives "
        "using that context.\n"
    )

    system_with_guidance = system + tool_guidance
    system_msg = SystemMessage(content=system_with_guidance)

    # Build the user message for this iteration.
    # NOTE: cfn_graph_context is intentionally absent - schema context arrives
    # via ToolMessage in the LLM context window, not injected into the prompt.
    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        annotated_template=annotated_template,
        validation_errors=formatted_errors,
        policy_source_context=policy_source_context,
    )

    # Replay prior clean user/assistant pairs as conversation history.
    # Tool-call AIMessages from prior iterations are NOT replayed here -
    # they were local to each iteration's tool loop and were never persisted
    # to remediator_history. This avoids LangChain validation errors caused
    # by orphaned ToolCall messages without matching ToolResult pairs.
    lc_history = _build_lc_history(state)
    user_lc_msg = HumanMessage(content=user_content)

    clean_content, raw_content, _, tool_context, retrieval_queries, tool_message_content = _run_tool_loop(
        system_msg=system_msg,
        initial_messages=lc_history + [user_lc_msg],
        state=state,
        recorder=recorder,
    )

    if retrieval_queries:
        print(
            f"[Remediator] retrieve_schema_context called with "
            f"{len(retrieval_queries)} quer(ies). Context: {len(tool_context)} chars."
        )
    print("[Remediator] Suggestions generated. Routing to Engineer.")

    return _build_return_state(
        state=state,
        iteration=iteration,
        system_with_guidance=system_with_guidance,
        user_content=user_content,
        clean_content=clean_content,
        raw_content=raw_content,
        flat_errors=flat_errors,
        formatted_errors=formatted_errors,
        tool_context=tool_context,
        retrieval_queries=retrieval_queries,
        tool_message_content=tool_message_content,
        recorder=recorder,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_lc_history(state: GraphState) -> list:
    """Convert remediator_history dicts to LangChain message objects.

    Only clean user/assistant summary pairs are replayed. Tool-call AIMessages
    from prior iterations are never stored in remediator_history (they stay
    local to each _run_tool_loop call), so this function will never encounter
    them and will never accidentally inject orphaned ToolCall messages that
    would break LangChain's message validation.
    """
    lc_history: list = []
    for msg in state.get("remediator_history", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content))
    return lc_history


def _build_return_state(
    state: GraphState,
    iteration: int,
    system_with_guidance: str,
    user_content: str,
    clean_content: str,
    raw_content: str,
    flat_errors: list[str],
    formatted_errors: str,
    tool_context: str,
    retrieval_queries: list[str],
    tool_message_content: str,
    recorder: ResearchRecorder,
) -> GraphState:
    """Assemble the state update dict returned by the remediator agent."""
    from config import DEFAULT_CONFIG  # local import to avoid circular
    model = DEFAULT_CONFIG.model

    # FIX 2+3: Build the recorded prompt to faithfully reflect the LLM's full
    # context window. When the tool was called, the LLM saw:
    #   SYSTEM + USER -> [decided to call tool] -> TOOL_RESULT -> FINAL_ANSWER
    # Previously only SYSTEM + USER was recorded, making the audit trail
    # misleading: the recorded prompt did not contain the schema context that
    # actually drove the LLM's RCA and Fix Objectives.
    if tool_message_content:
        recorded_prompt = (
            f"SYSTEM:\n{system_with_guidance}\n\n"
            f"USER:\n{user_content}\n\n"
            f"TOOL_RESULT (retrieve_schema_context):\n{tool_message_content}"
        )
    else:
        recorded_prompt = f"SYSTEM:\n{system_with_guidance}\n\nUSER:\n{user_content}"

    # LLM call log stores the raw response (with reasoning) for full audit.
    llm_record = recorder.record_llm_call(
        state=state,
        agent="remediator",
        model=model,
        prompt=recorded_prompt,
        response=raw_content,
        token_usage={},
    )

    reasoning = extract_reasoning_block(raw_content)

    new_history_entry: RemediationHistory = {
        "iteration":         iteration,
        "errors":            state["validation_results"],
        "flat_errors":       flat_errors,
        "formatted_errors":  formatted_errors,
        # suggestion is always clean - no <reasoning> block
        "suggestion":        clean_content,
        # reasoning stored separately for traceability; never forwarded to Engineer
        "reasoning":         reasoning,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "cfn_context":       tool_context,
        "retrieval_queries": retrieval_queries,
    }

    user_msg: Message      = {"role": "user",      "content": user_content}
    # assistant message stored clean - Engineer must not see <reasoning> block
    assistant_msg: Message = {"role": "assistant", "content": clean_content}

    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "llm_call_log":        state["llm_call_log"] + [llm_record],
        "remediator_history":  append_and_cap(
            state.get("remediator_history", []), user_msg, assistant_msg
        ),
        "retriever_context":   tool_context,
        "retriever_queries":   retrieval_queries,
    }
