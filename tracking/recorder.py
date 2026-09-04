import json
from datetime import datetime, timezone
from pathlib import Path
from state import LLMCallRecord, GraphState


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _extract_policy_metrics(validation_results: list[dict]) -> dict:
    total_policies = 0
    passed_policies = 0
    filtered_failed_policies = 0

    for result in validation_results:
        if result.get("stage") not in {"checkov", "trivy"}:
            continue

        stats = result.get("policy_stats") or {}
        total_policies += _safe_int(stats.get("total_policies"))
        passed_policies += _safe_int(stats.get("passed_policies"))
        filtered_failed_policies += _safe_int(stats.get("filtered_failed_policies"))

    if total_policies > 0:
        scenario_ppr = passed_policies / total_policies
        scenario_fcr = (total_policies - filtered_failed_policies) / total_policies
    else:
        scenario_ppr = 1.0
        scenario_fcr = 1.0

    return {
        "total_policies": total_policies,
        "passed_policies": passed_policies,
        "filtered_failed_policies": filtered_failed_policies,
        "scenario_policy_pass_rate": scenario_ppr,
        "filtered_compliance_rate": scenario_fcr,
    }


class ResearchRecorder:
    def __init__(self, run_id: str, output_dir: str = "./runs"):
        self.run_id = run_id
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_llm_call(
        self,
        state: GraphState,
        agent: str,
        model: str,
        prompt: str,
        response: str,
        token_usage: dict | None = None,
    ) -> LLMCallRecord:
        record: LLMCallRecord = {
            "agent": agent,
            "iteration": state["current_iteration"],
            "model": model,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_usage": token_usage,
        }
        with open(self.output_dir / "llm_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def record_tool_call(
        self,
        state: GraphState,
        agent: str,
        tool_name: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
    ) -> dict:
        """Record a tool invocation with inputs and outputs."""
        record = {
            "agent": agent,
            "iteration": state["current_iteration"],
            "tool_name": tool_name,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.output_dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def record_deployment_log(
        self,
        iteration: int,
        iac_type: str,
        target: str,
        deployment_logs: list[str],
        passed: bool,
        duration_seconds: float,
        failed_resources: list[dict] | None = None,
    ) -> None:
        """Persist the full raw deployment log for a single validator run.

        Writes to deployment_log_<iteration:03d>.txt so every tflocal/terraform
        apply stdout+stderr is available for offline analysis, independent of
        what the LLM-facing format_deploy_errors() chooses to surface.

        File format:
          Header block  — run metadata (iteration, iac_type, target, result)
          Failed block  — failed resource addresses + reasons (if any)
          Raw log       — every line from deployment_logs, line-numbered

        Args:
            iteration:         Graph iteration number at time of deploy call.
            iac_type:          "terraform" or "cloudformation".
            target:            Deploy target string (e.g. "localstack", "aws").
            deployment_logs:   Raw log lines from DeployValidationResult.
            passed:            Whether the deployment succeeded.
            duration_seconds:  Wall-clock seconds the deploy stage took.
            failed_resources:  List of {"logical_name", "status_reason"} dicts.
        """
        log_path = self.output_dir / f"deployment_log_{iteration:03d}.txt"
        timestamp = datetime.now(timezone.utc).isoformat()
        status_label = "PASSED" if passed else "FAILED"

        header_lines = [
            f"{'=' * 72}",
            f"Deployment Log",
            f"Run ID    : {self.run_id}",
            f"Iteration : {iteration}",
            f"Timestamp : {timestamp}",
            f"IAC Type  : {iac_type}",
            f"Target    : {target.upper()}",
            f"Result    : {status_label}",
            f"Duration  : {duration_seconds:.2f}s",
            f"{'=' * 72}",
            "",
        ]

        failed_lines: list[str] = []
        if failed_resources:
            failed_lines.append("Failed Resources:")
            for fr in failed_resources:
                name = fr.get("logical_name") or "unknown"
                reason = fr.get("status_reason") or "no reason provided"
                failed_lines.append(f"  {name}: {reason}")
            failed_lines.append("")

        raw_log_lines = [
            "Raw Deployment Log:",
            f"  ({len(deployment_logs)} lines total)",
            "",
        ]
        for i, line in enumerate(deployment_logs, start=1):
            raw_log_lines.append(f"  {i:>4}: {line}")

        content = (
            "\n".join(header_lines)
            + "\n".join(failed_lines)
            + "\n".join(raw_log_lines)
            + "\n"
        )
        log_path.write_text(content, encoding="utf-8")
        print(f"[Recorder] Deployment log saved: {log_path.name} ({len(deployment_logs)} lines, {status_label})")

    def save_iteration_snapshot(self, state: GraphState):
        """Save full state snapshot at each iteration boundary."""
        iteration = state["current_iteration"]
        snapshot_path = self.output_dir / f"iteration_{iteration:03d}.json"
        policy_metrics = _extract_policy_metrics(state.get("validation_results", []))
        snapshot = {
            "iteration": iteration,
            "objectives": state["objectives"],
            "iac_template": state["iac_template"],
            "validation_results": state["validation_results"],
            "validation_passed": state["validation_passed"],
            "policy_metrics": policy_metrics,
            "deploy_validation_result": state.get("deploy_validation_result"),
            "remediation_history": state["remediation_history"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2))

        # Conversation-history agents: overwrite with the latest full history
        # on every snapshot (the history list IS the source of truth).
        for agent in ("planner", "engineer", "remediator", "retriever"):
            self._write_agent_history(agent, state.get(f"{agent}_history", []))

    def append_retriever_history_entry(
        self,
        iteration: int,
        prompt: str,
        response: str,
        retrieval_queries: list[str],
        context_chars: int,
        retrieved_context: str = "",
    ) -> None:
        """Append one retriever invocation to retriever_history.txt.

        Unlike conversation-history agents (planner, engineer, remediator)
        whose history files are rewritten wholesale on every snapshot, the
        retriever's history file is append-only. Each call to this method
        writes a single dated block so no prior invocation is lost.

        Each block contains:
          - Run metadata (iteration, timestamp, run ID)
          - Retrieval queries used
          - Full assembled schema context returned by the hybrid RAG tool
          - LLM prompt and raw response for the query-generation call

        Args:
            iteration:         Graph iteration number at time of call.
            prompt:            Full prompt sent to the query-generation LLM.
            response:          Raw LLM response (JSON with retrieval_queries).
            retrieval_queries: Parsed query list actually used for retrieval.
            context_chars:     Character count of the assembled CFN context.
            retrieved_context: Full schema context string returned by the
                               hybrid RAG tool (ChromaDB + Neo4j output).
        """
        history_path = self.output_dir / "retriever_history.txt"
        timestamp = datetime.now(timezone.utc).isoformat()

        queries_str = (
            "\n".join(f"  {i+1}. {q}" for i, q in enumerate(retrieval_queries))
            if retrieval_queries
            else "  (none — fell back to raw error strings)"
        )

        context_section = retrieved_context.strip() if retrieved_context else "(empty)"

        block = (
            f"{'=' * 72}\n"
            f"Iteration : {iteration}\n"
            f"Timestamp : {timestamp}\n"
            f"Run ID    : {self.run_id}\n"
            f"Context   : {context_chars} chars assembled\n"
            f"Queries   :\n{queries_str}\n"
            f"{'- ' * 36}\n"
            f"[prompt]\n{prompt}\n"
            f"{'- ' * 36}\n"
            f"[response]\n{response}\n"
            f"{'- ' * 36}\n"
            f"[retrieved schema context]\n{context_section}\n"
        )

        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(block)

    def save_final_report(self, state: GraphState):
        """Save complete research report at end of run."""
        policy_metrics = _extract_policy_metrics(state.get("validation_results", []))
        report = {
            "run_id": self.run_id,
            "user_request": state["user_request"],
            "total_iterations": state["current_iteration"],
            "final_passed": state["validation_passed"],
            "objectives": state["objectives"],
            "final_template": state["final_template"],
            "remediation_history": state["remediation_history"],
            "llm_calls_total": len(state["llm_call_log"]),
            "llm_call_log": state["llm_call_log"],
            "validation_results": state["validation_results"],
            "policy_metrics": policy_metrics,
            "deploy_validation_result": state.get("deploy_validation_result"),
        }
        (self.output_dir / "final_report.json").write_text(
            json.dumps(report, indent=2)
        )
        print(f"\n[Recorder] Run complete. Report saved to: {self.output_dir}/final_report.json")

    def _write_agent_history(self, agent: str, history: list[dict]) -> None:
        """Overwrite <agent>_history.txt with the full formatted conversation.

        Used for agents with a rolling conversation list (planner, engineer,
        remediator, retriever). The list itself accumulates all turns via
        append_and_cap(), so overwriting on each snapshot is correct.
        """
        history_path = self.output_dir / f"{agent}_history.txt"
        history_path.write_text(
            self._format_history(agent, history),
            encoding="utf-8",
        )

    def _format_history(self, agent: str, history: list[dict]) -> str:
        lines: list[str] = [
            f"Agent: {agent}",
            f"Run ID: {self.run_id}",
            f"Updated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        if not history:
            lines.append("No conversation history recorded yet.")
            return "\n".join(lines) + "\n"

        turn = 1
        for index in range(0, len(history), 2):
            user_msg = history[index]
            assistant_msg = history[index + 1] if index + 1 < len(history) else None

            lines.append(f"Turn {turn}")
            lines.append(f"[user]\n{user_msg['content']}")
            if assistant_msg is not None:
                lines.append(f"[assistant]\n{assistant_msg['content']}")
            lines.append("")
            turn += 1

        return "\n".join(lines).rstrip() + "\n"
