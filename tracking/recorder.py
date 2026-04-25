# tracking/recorder.py
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
        # Append to JSONL file for streaming access
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
        # Append to JSONL file for streaming access
        with open(self.output_dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def save_iteration_snapshot(self, state: GraphState):
        """Save full state snapshot at each iteration boundary."""
        iteration = state["current_iteration"]
        snapshot_path = self.output_dir / f"iteration_{iteration:03d}.json"
        policy_metrics = _extract_policy_metrics(state.get("validation_results", []))
        snapshot = {
            "iteration": iteration,
            "objectives": state["objectives"],
            "cloudformation_template": state["cloudformation_template"],
            "validation_results": state["validation_results"],
            "validation_passed": state["validation_passed"],
            "policy_metrics": policy_metrics,
            "deploy_validation_result": state.get("deploy_validation_result"),
            "remediation_history": state["remediation_history"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
        self._write_agent_history("planner", state["planner_history"])
        self._write_agent_history("engineer", state["engineer_history"])
        self._write_agent_history("remediator", state["remediator_history"])
        self._write_agent_history("retriever", state.get("retriever_history", []))

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
            "retriever_history": state.get("retriever_history", []),
        }
        (self.output_dir / "final_report.json").write_text(
            json.dumps(report, indent=2)
        )
        print(f"\n[Recorder] Run complete. Report saved to: {self.output_dir}/final_report.json")

    def _write_agent_history(self, agent: str, history: list[dict]) -> None:
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