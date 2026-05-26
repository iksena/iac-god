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


class FailedResource(TypedDict):
    """Canonical shape for a single resource-level deploy failure.

    This matches what deploy_validator.py emits in every code path:
      - _drain_stack_events():    {"logical_name": rid,       "status_reason": reason}
      - parameter pre-check:      {"logical_name": param_key, "status_reason": "..."}
      - ClientError / unexpected: {"logical_name": "stack",   "status_reason": msg}
    """
    logical_name: str   # CloudFormation LogicalResourceId (or "stack"/"template" sentinel)
    status_reason: str  # CloudFormation ResourceStatusReason or error message


class DeployValidationResult(TypedDict):
    target: str                         # "localstack" | "aws" | "skipped"
    passed: bool
    stack_id: Optional[str]
    completed_resources: list[str]
    failed_resources: list[FailedResource]
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
    retriever_context: str
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


# ---------------------------------------------------------------------------
# Stage grouping for per-stage error counting
# ---------------------------------------------------------------------------
# Maps a logical group name to the set of ValidationResult stage strings
# that belong to it.  The validator increments stage_error_counts[group]
# whenever any member stage has failures in a given iteration.
#
# Groups:
#   "yaml-cfn-lint" — structural / schema correctness (always active)
#   "security"      — policy violations (checkov, trivy — currently skipped
#                     in run_all_validators but wired here for when re-enabled)
#   "deploy"        — live deployability (localstack / AWS)
STAGE_GROUPS: dict[str, set[str]] = {
    "yaml-cfn-lint": {"yaml", "cfn-lint"},
    "security":      {"checkov", "trivy"},
    "deploy":        {"deploy"},
}


def classify_failing_stages(
    validation_results: list["ValidationResult"],
    deploy_validation_result: Optional["DeployValidationResult"],
) -> set[str]:
    """Return the set of STAGE_GROUPS keys that have failures this iteration."""
    failing: set[str] = set()

    failed_static = {
        r["stage"]
        for r in validation_results
        if not r.get("passed", True)
    }

    for group, members in STAGE_GROUPS.items():
        if group == "deploy":
            continue  # handled separately below
        if failed_static & members:
            failing.add(group)

    if (
        deploy_validation_result
        and not deploy_validation_result.get("passed", True)
        and deploy_validation_result.get("target") != "skipped"
    ):
        failing.add("deploy")

    return failing


def any_stage_in_moderate_mode(
    stage_error_counts: dict[str, int],
    failing_stages: set[str],
    threshold: int,
) -> bool:
    """Return True if any currently-failing stage has reached the moderate threshold.

    A stage is in moderate mode when its cumulative error-iteration count
    (incremented by the validator on every failing iteration) equals or
    exceeds *threshold*.  Only stages that are *currently* failing are
    checked — a stage that passed this iteration does not trigger moderate
    mode even if its historical count is high.
    """
    return any(
        stage_error_counts.get(stage, 0) >= threshold
        for stage in failing_stages
    )


def compact_message_history(history: list[Message]) -> list[Message]:
    compacted: list[Message] = []
    for message in history:
        if compacted and compacted[-1] == message:
            continue
        compacted.append(message)
    return compacted


MAX_HISTORY_PAIRS = 15  # keep last 15 user+assistant pairs = 30 messages


def append_and_cap(
    history: list[Message],
    user_msg: Message,
    assistant_msg: Message,
) -> list[Message]:
    updated = history + [user_msg, assistant_msg]
    return updated[-MAX_HISTORY_PAIRS * 2:]


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

    # --- Per-stage error iteration counters ---
    # Keyed by STAGE_GROUPS group name ("yaml-cfn-lint", "security", "deploy").
    # Incremented by the validator on every iteration where that group fails.
    # Used by the router to decide between simple and moderate remediation.
    stage_error_counts: dict[str, int]

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
    planner_history:    list[Message]
    engineer_history:   list[Message]
    remediator_history: list[Message]
    retriever_history:  list[Message]

    # --- Retriever outputs ---
    retriever_context: str
    retriever_queries: list[str]

    # --- Final output ---
    final_template: Optional[str]
    run_id: str
