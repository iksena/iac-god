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

# Iteration 1: clean generation request — no history context needed
ENGINEER_USER_INITIAL = "Generate the CloudFormation template that fully satisfies all objectives above."

# Iteration 2+: full context in prompt — no conversation history passed to LLM
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
