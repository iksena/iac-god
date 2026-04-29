# state.py
from typing import TypedDict, Optional, NotRequired


class PolicyStats(TypedDict):
    total_policies: int
    passed_policies: int
    failed_policies: int
    filtered_failed_policies: int

class ValidationResult(TypedDict):
    stage: str          # "yaml" | "cfn-lint" | "checkov" | "trivy"
    passed: bool
    errors: list[str]
    raw_output: str
    policy_stats: NotRequired[PolicyStats]
    scenario_policy_pass_rate: NotRequired[float]
    filtered_compliance_rate: NotRequired[float]

class DeployValidationResult(TypedDict):
    target: str                     # "localstack" | "aws" | "skipped"
    passed: bool
    stack_id: Optional[str]
    completed_resources: list[str]
    failed_resources: list[dict]    # [{"resource": str, "reason": str}]
    error_message: Optional[str]
    duration_seconds: float
    deployment_logs: list[str]

class RemediationHistory(TypedDict):
    iteration: int
    errors: list[ValidationResult]  # Raw validation snapshots (for audit / re-processing)
    flat_errors: list[str]          # Flat error strings used by render_annotated_template()
    formatted_errors: str           # Human-readable error context for LLM prompt
    suggestion: str                 # Remediator's fix suggestion
    timestamp: str
    cfn_context: str
    retrieval_queries: list[str]

class LLMCallRecord(TypedDict):
    agent: str
    iteration: int
    model: str
    prompt: str
    response: str
    timestamp: str
    token_usage: Optional[dict]

class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


def compact_message_history(history: list[Message]) -> list[Message]:
    compacted: list[Message] = []

    for message in history:
        if compacted and compacted[-1] == message:
            continue
        compacted.append(message)

    return compacted

MAX_HISTORY_PAIRS = 10  # keep last 5 user+assistant pairs = 10 messages

def append_and_cap(history: list[Message], user_msg: Message, assistant_msg: Message) -> list[Message]:
    updated = history + [user_msg, assistant_msg]
    # Always keep system msg if present, then cap the tail
    pairs = updated[-MAX_HISTORY_PAIRS * 2:]
    return pairs

class GraphState(TypedDict):
    # --- Core inputs ---
    user_request: str

    # --- Grounded Objectives Document (shared across all agents) ---
    objectives: list[str]           # Planner output (CGO-style)
    cloudformation_template: str    # Engineer output (latest YAML)

    # --- Validation state ---
    validation_results: list[ValidationResult]
    validation_passed: bool
    deploy_validation_result: Optional[DeployValidationResult]

    # --- Static analysis smell report (populated by smell detector, consumed by retriever) ---
    smell_report: NotRequired[list[dict]]

    # --- Remediation history (all iterations) ---
    remediation_history: list[RemediationHistory]

    # --- Iteration control ---
    current_iteration: int
    max_iterations: int

    # --- Research tracking (all LLM conversations) ---
    llm_call_log: list[LLMCallRecord]

    # --- Per-agent conversation histories ---
    # Kept for debugging and recording only.
    # The Remediator and Retriever agents do NOT pass these as LLM context;
    # structured remediation_history is injected via the prompt instead.
    planner_history:    list[Message]
    engineer_history:   list[Message]
    remediator_history: list[Message]
    retriever_history:  list[Message]

    # --- Retriever outputs ---
    retriever_context: str
    retriever_queries: list[str]

    # --- Remediator routing flag ---
    # Set to True by the Remediator's first pass (tool-call phase) to signal
    # graph.py that the Retriever should run next.  Cleared (False) by the
    # Remediator's second pass (synthesis phase) once context has been consumed.
    awaiting_retriever: bool

    # --- Final output ---
    final_template: Optional[str]
    run_id: str
