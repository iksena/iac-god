from datetime import datetime, timezone
import json
import os
from state import GraphState, RemediationHistory, Message, compact_message_history, append_and_cap
from prompts.remediator_prompt import REMEDIATOR_SYSTEM, REMEDIATOR_USER
from tracking.recorder import ResearchRecorder
from agents.engineer import _build_client, _call_llm_with_history
from tools.checkov_context import get_checkov_policy_context
from tools.trivy_context import get_trivy_policy_context
from tools.cfn_graph_context_rag import get_cfn_schema_context
from tools.cfn_aws_doc_context import get_cfn_aws_doc_context_for_state

# ---------------------------------------------------------------------------
# CFN context strategy selector
#
# Set env var  CFN_CONTEXT_STRATEGY=aws_doc_mcp  to use the live AWS
# Documentation MCP Server instead of the offline GraphRAG index.
#
# Default ("rag") uses the locally-built FAISS/BM25/graph index.
# "aws_doc_mcp" fetches live CloudFormation documentation per resource type
# via the awslabs.aws-documentation-mcp-server HTTP endpoint.
#
# The two strategies expose the same signature so swapping is zero-cost:
#   get_cfn_schema_context(validation_results, deploy_validation_result, template_yaml)
# ---------------------------------------------------------------------------

_CFN_STRATEGY = os.environ.get("CFN_CONTEXT_STRATEGY", "rag").lower()


def _get_cfn_schema_context(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
    template_yaml: str | None,
) -> tuple[str, str]:
    """
    Return (context_text, strategy_label) using the configured strategy.

    strategy_label is logged so it is always clear which backend was used.
    """
    if _CFN_STRATEGY == "aws_doc_mcp":
        ctx = get_cfn_aws_doc_context_for_state(
            validation_results=validation_results,
            deploy_validation_result=deploy_validation_result,
            template_yaml=template_yaml,
        )
        return ctx, "aws_doc_mcp"

    # Default: offline GraphRAG (FAISS + BM25 + graph)
    ctx = get_cfn_schema_context(
        queries="none",
        template_yaml=template_yaml,
    )
    return ctx, "rag"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_checkov_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for result in validation_results:
        if result.get("stage") != "checkov":
            continue
        raw = result.get("raw_output", "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        failed = data.get("results", {}).get("failed_checks", [])
        for item in failed:
            check_id = str(item.get("check_id") or "").strip()
            if check_id:
                findings.append({"check_id": check_id})
    return findings


def _extract_trivy_findings(validation_results: list[dict]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for result in validation_results:
        if result.get("stage") != "trivy":
            continue
        raw = result.get("raw_output", "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for r in data.get("Results", []):
            for misconfig in r.get("Misconfigurations", []):
                check_id = str(misconfig.get("ID") or "").strip()
                if check_id:
                    findings.append({"check_id": check_id})
    return findings


def _build_policy_source_context(validation_results: list[dict]) -> str:
    checkov_context = get_checkov_policy_context(
        _extract_checkov_findings(validation_results)
    )
    trivy_context = get_trivy_policy_context(
        _extract_trivy_findings(validation_results)
    )

    sections: list[str] = []
    if checkov_context:
        sections.append(f"### Checkov Policy Source\n{checkov_context}")
    if trivy_context:
        sections.append(f"### Trivy Policy Source\n{trivy_context}")
    if not sections:
        return ""
    
    policy_context = "## Relevant Policy Source Context (Checkov/Trivy)\n\n" + "\n\n".join(sections)

    return policy_context


def _build_validation_errors_text(state: GraphState) -> str:
    error_blocks: list[str] = []

    for result in state["validation_results"]:
        if result["passed"]:
            continue
        deduped_errors = _dedupe_preserve_order(
            [str(error) for error in result.get("errors", []) if str(error).strip()]
        )
        if not deduped_errors:
            continue
        errors_text = "\n".join(f"  - {error}" for error in deduped_errors)
        error_blocks.append(f"### {result['stage'].upper()} Errors\n{errors_text}")

    deploy_result = state.get("deploy_validation_result")
    if deploy_result and not deploy_result["passed"] and deploy_result["target"] != "skipped":
        target = deploy_result["target"].upper()
        lines: list[str] = []

        failed = deploy_result.get("failed_resources", [])
        if failed:
            lines.append("**Failed resources:**")
            for fr in failed:
                name   = fr.get("logical_name") or fr.get("resource") or "unknown"
                reason = fr.get("status_reason") or fr.get("reason") or "no reason provided"
                lines.append(f"  - `{name}`: {reason}")
        else:
            if deploy_result.get("error_message"):
                lines.append(f"**Error:** {deploy_result['error_message']}")

        completed = deploy_result.get("completed_resources", [])
        if completed:
            lines.append(f"**Resources that completed successfully:** {', '.join(f'`{r}`' for r in completed)}")

        deploy_logs = deploy_result.get("deployment_logs", [])
        actionable_log_lines = [
            line for line in deploy_logs
            if any(kw in str(line) for kw in ("FAILED", "ERROR", "timed out", "does not exist", "InvalidAMI", "parameter"))
        ]
        if actionable_log_lines:
            lines.append("**Deployment event log (errors only):**")
            for log_line in actionable_log_lines:
                lines.append(f"  - {log_line}")
        elif not failed and deploy_logs:
            lines.append("**Last deployment events:**")
            for log_line in deploy_logs[-5:]:
                lines.append(f"  - {log_line}")

        if not lines:
            lines.append("Deployment failed with no structured error details.")

        error_blocks.append(
            f"### DEPLOYABILITY Errors ({target})\n" + "\n".join(lines)
        )

    if not error_blocks:
        return "No validation errors reported."
    return "\n\n".join(error_blocks)

def remediator_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Remediator] Analyzing errors (iteration {iteration})...")

    system = REMEDIATOR_SYSTEM.format(
        user_request=state["user_request"],
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    policy_source_context  = _build_policy_source_context(state["validation_results"])

    # cfn_graph_context, strategy = _get_cfn_schema_context(
    #     validation_results=state.get("validation_results"),
    #     deploy_validation_result=state.get("deploy_validation_result"),
    #     template_yaml=state["cloudformation_template"],
    # )
    # print(f"[Remediator] CFN schema context ({strategy}): {len(cfn_graph_context)} chars")

    user_content = REMEDIATOR_USER.format(
        iteration=iteration,
        template=state["cloudformation_template"],
        validation_errors=_build_validation_errors_text(state),
        policy_source_context=policy_source_context,
        cfn_graph_context="",
    )
    user_msg: Message = {"role": "user", "content": user_content}

    messages = compact_message_history(state["remediator_history"]) + [user_msg]

    client, model = _build_client()
    content, usage = _call_llm_with_history(client, model, system, messages)
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state, agent="remediator", model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content, token_usage=usage,
    )

    new_history_entry: RemediationHistory = {
        "iteration": iteration,
        "errors": state["validation_results"],
        "suggestion": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[Remediator] Suggestions generated. Routing back to Engineer.")
    return {
        "remediation_history": state["remediation_history"] + [new_history_entry],
        "current_iteration": iteration + 1,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "remediator_history": append_and_cap(state["remediator_history"], user_msg, assistant_msg),
    }
