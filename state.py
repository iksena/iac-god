# state.py
from typing import TypedDict, Annotated, Optional
import operator

class ValidationResult(TypedDict):
    stage: str          # "yaml" | "cfn-lint" | "checkov" | "trivy"
    passed: bool
    errors: list[str]
    raw_output: str

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
    errors: list[ValidationResult]
    suggestion: str     # Remediator's fix suggestion
    timestamp: str

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
    
    # --- Remediation history (all iterations) ---
    remediation_history: list[RemediationHistory]
    
    # --- Iteration control ---
    current_iteration: int
    max_iterations: int
    
    # --- Research tracking (all LLM conversations) ---
    llm_call_log: Annotated[list[LLMCallRecord], operator.add]

    # --- Per-agent conversation histories (NEW) ---
    # Each agent returns only the new [user_msg, assistant_msg] pair.
    # operator.add means LangGraph concatenates the returned pair onto the accumulated history.
    planner_history:    Annotated[list[Message], operator.add]
    engineer_history:   Annotated[list[Message], operator.add]
    remediator_history: Annotated[list[Message], operator.add]
    
    # --- Final output ---
    final_template: Optional[str]
    run_id: str