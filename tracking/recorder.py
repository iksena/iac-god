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

    def record_rag_tool_call(
        self,
        state: GraphState,
        agent: str,
        retrieval_queries: list[str],
        seed_resources: list[str],
        context_returned: str,
        reasoning_block: str,
        raw_ai_response: str,
        round_idx: int,
    ) -> dict:
        """Record a retrieve_schema_context tool invocation in full detail.

        Captures:
        - The <reasoning> block the LLM produced before deciding to call the tool
        - The exact queries passed to the tool (for ChromaDB semantic search)
        - The seed_resources list passed to the tool (for Neo4j graph traversal)
        - The full schema context string returned by the tool
        - The raw AIMessage content (reasoning + tool_call decision, pre-result)
        - Round index within the tool loop

        Written to two sinks:

        1. rag_tool_calls.jsonl  (append)
           Machine-readable; one JSON line per invocation across all iterations
           and rounds. Stores the full context_returned string (not just char
           count) so benchmarks and replays have the exact retrieved content.
           Never overwritten - full audit trail.

        2. rag_tool_data_flow.txt  (append)
           Human-readable; one dated block per invocation appended in order.
           Each entry is separated by a clear header showing run_id, iteration,
           and round, so the full tool-call history for a run is visible in one
           file without losing earlier rounds or iterations.
           Owned entirely by this method; never touched by save_iteration_snapshot.
        """
        iteration = state["current_iteration"]
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "agent": agent,
            "iteration": iteration,
            "tool_name": "retrieve_schema_context",
            "round_idx": round_idx,
            "retrieval_queries": retrieval_queries,
            "seed_resources": seed_resources,
            # FIX 1: store the full context string, not just the char count.
            # Previously only context_returned_chars was persisted, making it
            # impossible to benchmark or replay against the actual retrieved content.
            "context_returned": context_returned,
            "context_returned_chars": len(context_returned),
            "reasoning_block": reasoning_block,
            "raw_ai_response": raw_ai_response,
            "timestamp": timestamp,
        }

        # Sink 1: machine-readable append-only log
        with open(self.output_dir / "rag_tool_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        # Sink 2: human-readable append-only log - all invocations accumulated
        self._write_rag_tool_data_flow(
            iteration=iteration,
            round_idx=round_idx,
            retrieval_queries=retrieval_queries,
            seed_resources=seed_resources,
            context_returned=context_returned,
            reasoning_block=reasoning_block,
            timestamp=timestamp,
        )

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

        # Conversation-history agents: overwrite with the latest full history
        # on every snapshot (the history list IS the source of truth).
        for agent in ("planner", "engineer", "remediator"):
            self._write_agent_history(agent, state.get(f"{agent}_history", []))

        # Retriever has no rolling conversation history - each invocation is a
        # fresh single-turn call. Its history is appended live by
        # append_retriever_history_entry(); nothing to do here.

    def append_retriever_history_entry(
        self,
        iteration: int,
        prompt: str,
        response: str,
        retrieval_queries: list[str],
        context_chars: int,
    ) -> None:
        """Append one retriever invocation to retriever_history.txt.

        Unlike conversation-history agents (planner, engineer, remediator)
        whose history files are rewritten wholesale on every snapshot, the
        retriever's history file is append-only. Each call to this method
        writes a single dated block so no prior invocation is lost.

        Args:
            iteration:         Graph iteration number at time of call.
            prompt:            Full prompt sent to the query-generation LLM.
            response:          Raw LLM response (JSON with retrieval_queries).
            retrieval_queries: Parsed query list actually used for retrieval.
            context_chars:     Character count of the assembled CFN context.
        """
        history_path = self.output_dir / "retriever_history.txt"
        timestamp = datetime.now(timezone.utc).isoformat()

        queries_str = (
            "\n".join(f"  {i+1}. {q}" for i, q in enumerate(retrieval_queries))
            if retrieval_queries
            else "  (none - fell back to raw error strings)"
        )

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

        Correct for agents with a rolling conversation list (planner, engineer,
        remediator) because the list itself accumulates all turns.
        NOT used for the retriever - use append_retriever_history_entry() instead.
        NOT used for RAG tool calls - use record_rag_tool_call() instead.
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

    def _write_rag_tool_data_flow(
        self,
        iteration: int,
        round_idx: int,
        retrieval_queries: list[str],
        seed_resources: list[str],
        context_returned: str,
        reasoning_block: str,
        timestamp: str,
    ) -> None:
        """Append one RAG tool invocation block to rag_tool_data_flow.txt.

        Shows the complete data flow through retrieve_schema_context:
          INPUT  : retrieval queries generated by the LLM (for ChromaDB)
          INPUT  : seed_resources extracted by the LLM (for Neo4j) - NOT the full YAML
          OUTPUT : schema context string returned by the hybrid RAG pipeline

        This file is append-only (open mode "a") so every tool call across all
        rounds and iterations is accumulated in order. Each entry is separated
        by a clear ====== header showing run_id, iteration, and round.
        For the machine-readable audit trail see rag_tool_calls.jsonl.
        """
        data_flow_path = self.output_dir / "rag_tool_data_flow.txt"

        queries_str = (
            "\n".join(f"  {i+1}. {q}" for i, q in enumerate(retrieval_queries))
            if retrieval_queries
            else "  (none)"
        )

        resources_str = (
            "\n".join(f"  {i+1}. {r}" for i, r in enumerate(seed_resources))
            if seed_resources
            else "  (none)"
        )

        reasoning_section = (
            reasoning_block.strip()
            if reasoning_block.strip()
            else "(no <reasoning> block extracted)"
        )

        block = (
            f"{'=' * 72}\n"
            f"RAG Tool Call\n"
            f"{'=' * 72}\n"
            f"Run ID          : {self.run_id}\n"
            f"Iteration       : {iteration}\n"
            f"Round           : {round_idx + 1}\n"
            f"Timestamp       : {timestamp}\n"
            f"{'=' * 72}\n"
            f"\n"
            f"[pre-tool reasoning]\n"
            f"{'-' * 72}\n"
            f"{reasoning_section}\n"
            f"\n"
            f"[INPUT: retrieval queries ({len(retrieval_queries)}) -> ChromaDB semantic search]\n"
            f"{'-' * 72}\n"
            f"{queries_str}\n"
            f"\n"
            f"[INPUT: seed resources ({len(seed_resources)}) -> Neo4j graph traversal]\n"
            f"{'-' * 72}\n"
            f"{resources_str}\n"
            f"\n"
            f"[OUTPUT: schema context ({len(context_returned)} chars)]\n"
            f"{'-' * 72}\n"
            f"{context_returned}\n"
            f"\n"
        )

        with open(data_flow_path, "a", encoding="utf-8") as fh:
            fh.write(block)
