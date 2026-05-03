# prompts/engineer_prompt.py

ENGINEER_SYSTEM = """You are an expert AWS CloudFormation engineer.
You generate syntactically correct, secure, deployable, and production-ready CloudFormation YAML templates.
Always follow AWS best practices. Do NOT include any rule suppressions or workarounds for known issues.

## Original User Request
{user_request}

## Grounded Objectives
{objectives}

Output ONLY the raw CloudFormation YAML. No explanation, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Path A: clean generation on iteration 1
# ---------------------------------------------------------------------------
ENGINEER_USER_INITIAL = (
    "Generate the CloudFormation template that fully satisfies all objectives above."
)

# ---------------------------------------------------------------------------
# Path B: simple self-correction
# The engineer receives rich cfn-lint errors (rule ID, line number, resource
# logical ID, human-readable description) and the annotated template with
# inline # ERROR: comments at the reported lines.  No schema RAG context is
# provided — the errors carry enough signal for the model to self-correct.
# ---------------------------------------------------------------------------
ENGINEER_USER_SIMPLE_FIX = """\
Iteration {iteration} — The validator found errors in your template.
Fix ALL errors listed below directly. No Remediator guidance is provided at this stage.

## Current Template
Lines prefixed with `# ERROR:` are injected at the exact line numbers reported
by cfn-lint. Each error line also shows the Resource logical ID and rule
description so you can resolve the issue without additional context.

```yaml
{annotated_template}
```

## Validation Errors
Each error is in the format:
  [RuleId] line N | Resource: LogicalId | Error message | Rule description | See: <doc url>

Use the rule description and documentation URL to apply the correct fix.

{validation_errors}

Rules:
- Fix every error listed. Do not suppress or comment out checks.
- Preserve all resources, properties, and logic that are NOT related to the errors.
- Output the complete corrected CloudFormation YAML.
"""

# ---------------------------------------------------------------------------
# Path C: moderate remediation with schema context from retriever
# ---------------------------------------------------------------------------
ENGINEER_USER_REMEDIATION = """\
Iteration {iteration} — Apply these fixes based on validation errors and remediation suggestions from the Remediator agent:

## Current Template
The template below has `# ERROR:` comments injected at the exact line numbers
reported by cfn-lint and the deployment validator. Apply fixes precisely at the
marked locations.
```yaml
{annotated_template}
```

## Validation Errors
{error_context}

## Remediation Suggestion
{remediation_suggestion}

## Retrieved CloudFormation Schema Context
The following schema context was retrieved specifically for the errors above.
Use it to produce property-correct, constraint-aware YAML:

{cfn_context}

{remediation_history_context}

The final template must also satisfy Original User Request and Grounded Objectives.
These fix objectives can override or extend previous fix objectives and Grounded Objectives.
Do NOT repeat changes that are marked as already applied in the history above.
Output the complete corrected CloudFormation YAML.
"""
