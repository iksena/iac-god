# prompts/engineer_prompt.py

ENGINEER_SYSTEM = """You are an expert AWS CloudFormation engineer.
You generate syntactically correct, secure, and production-ready CloudFormation
YAML templates. Always follow AWS best practices.

You will be given:
- A list of OBJECTIVES (functional requirements)
- Optionally: a PREVIOUS TEMPLATE and REMEDIATION SUGGESTION (for iteration)

Output ONLY the raw CloudFormation YAML. No explanation, no markdown fences.
"""

ENGINEER_USER = """## Grounded Objectives
{objectives}

{remediation_context}

Generate the CloudFormation template that fully satisfies all objectives.
"""

ENGINEER_REMEDIATION_CONTEXT = """
--- Remediation Directive (Iteration {iteration}) ---
Apply the following fixes to your previous template:
{remediation_suggestion}
"""