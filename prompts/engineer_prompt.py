# prompts/engineer_prompt.py

ENGINEER_SYSTEM = """You are an expert AWS CloudFormation engineer.
You generate syntactically correct, secure, and production-ready CloudFormation
YAML templates. Always follow AWS best practices.

## Grounded Objectives (fixed for this run)
{objectives}

Output ONLY the raw CloudFormation YAML. No explanation, no markdown fences.
"""

# Iteration 1: clean generation request — no history context needed
ENGINEER_USER_INITIAL = "Generate the CloudFormation template that fully satisfies all objectives above."

# Iteration 2+: ONLY the new fix directive — previous template is in history[-1] assistant turn
ENGINEER_USER_REMEDIATION = """\
Iteration {iteration} fix directive — apply these changes to your previous template:

{remediation_suggestion}

Output the complete corrected CloudFormation YAML."""