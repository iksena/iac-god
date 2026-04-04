# state.py
from typing import TypedDict, Annotated, Optional
import operator

class ValidationResult(TypedDict):
    stage: str          # "yaml" | "cfn-lint" | "checkov" | "trivy"
    passed: bool
    errors: list[str]
    raw_output: str

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

class GraphState(TypedDict):
    # --- Core inputs ---
    user_request: str

    # --- Grounded Objectives Document (shared across all agents) ---
    objectives: list[str]           # Planner output (CGO-style)
    cloudformation_template: str    # Engineer output (latest YAML)
    
    # --- Validation state ---
    validation_results: list[ValidationResult]
    validation_passed: bool
    
    # --- Remediation history (all iterations) ---
    remediation_history: list[RemediationHistory]
    
    # --- Iteration control ---
    current_iteration: int
    max_iterations: int
    
    # --- Research tracking (all LLM conversations) ---
    llm_call_log: Annotated[list[LLMCallRecord], operator.add]
    
    # --- Final output ---
    final_template: Optional[str]
    run_id: str