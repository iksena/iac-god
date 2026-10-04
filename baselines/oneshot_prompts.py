"""Prompts for the one-shot baseline, copied verbatim from the original IaCGen.

Source: IaCGen/Code/generation/prompts/prompt_for_cloud.py (the first-iteration
prompt in IaCGen/Code/main.py::process_template is
FORMATE_SYSTEM_PROMPT as system + TOP_PROMPT + <business need> + BOTTOM_PROMPT
as the single user turn). The Terraform pair is the IaC-Eval prompt IaCGen
carries in the same file. Do not reword these: the point of the baseline is
that the only difference from IaCGen's own first generation is the model/API
under test.
"""

# --- CloudFormation (IaCGen) -------------------------------------------------

TOP_PROMPT = "You are an expert AWS DevOps engineer with extensive experience in creating CloudFormation templates. Your task is to generate a valid, production ready and deployable AWS CloudFormation YAML template based on the following business need:\n\n<business_need>\n"

BOTTOM_PROMPT = "\n</business_need>\n\nInstructions:\n1. Analyze the business need carefully.\n2. Generate a complete CloudFormation YAML template that fulfills this need.\n3. Be sure to fully specified all resources needed in the Resources section.\n4. Ensure high accuracy and deployment success by following these guidelines:\n\n   a. Start the template with 'AWSTemplateFormatVersion'.\n   b. Include all necessary resources to meet the business need.\n   c. Provide all required properties for each resource.\n   d. Use proper YAML syntax and indentation.\n   e. Follow AWS CloudFormation best practices and cloudformation-linter rules.\n   f. End the template with the last property of the last resource.\n\n5. Do not include any explanations, markdown formatting, or backticks in your output.\n\nBefore generating the final template, wrap your planning process in <template_planning> tags. In this section:\n\n1. List the key AWS services mentioned or implied in the business need.\n2. Outline the main sections of the CloudFormation template (e.g., Parameters, Mappings, Resources, Outputs).\n3. Consider potential dependencies between resources and how to order them.\n4. Think about any parameters or mappings that might be needed for flexibility.\n5. Consider any outputs that would be useful for the user after stack creation.\n\nThis planning process will help reduce errors and improve deployment success rate. It's okay for this section to be quite long.\n\nAfter your planning process, provide the complete CloudFormation YAML template as your final output."

FORMATE_SYSTEM_PROMPT = "You are an expert in AWS CloudFormation template generation. Your task is to generate and improve templates based on feedback. Please write your complete CloudFormation YAML template inside <iac_template></iac_template> tags."

# --- Terraform (IaC-Eval, as carried in IaCGen's prompt file) ----------------

FORMATE_SYSTEM_PROMPT_TF = "You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response to each of my Instructions. Make sure the configuration is deployable. Create IAM roles as needed. If variables are used, make sure default values are supplied. Be sure to include a valid provider configuration within a valid region. Make sure there are no undeclared resources (e.g., as references) or variables, i.e., all resources and variables needed in the configuration should be fully specified. Please write your complete HCL template inside <iac_template></iac_template> tags."

TOP_PROMPT_TF = "Here is the actual business need description which you should follow to build and deploy Cloud Infrastructure written in Terraform HCL: "


def build_messages(iac_type: str, business_need: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for IaCGen's first generation."""
    if iac_type == "terraform":
        return FORMATE_SYSTEM_PROMPT_TF, TOP_PROMPT_TF + business_need
    return FORMATE_SYSTEM_PROMPT, TOP_PROMPT + business_need + BOTTOM_PROMPT


def extract_template(content: str, iac_type: str) -> tuple[str, str]:
    """Pull the template out of a raw completion; returns (template, method).

    CloudFormation mirrors IaCGen's own extraction (main.py
    generate_template_with_history): drop the <template_planning> block, take
    what is inside <iac_template> tags, else fall back to the text from
    'AWSTemplateFormatVersion' up to any trailing triple backticks. Terraform
    has no IaCGen-specific extraction, so it takes the <iac_template> body and
    otherwise unwraps a markdown code fence.

    `method` records which path produced the template, so a run with many
    fallback/raw extractions is visible in the results rather than silently
    scored as if the model had followed the output format.
    """
    start_tag, end_tag = "<template_planning>", "</template_planning>"
    start_pos = content.find(start_tag)
    end_pos = content.find(end_tag, start_pos)
    if start_pos != -1 and end_pos != -1:
        content = content[:start_pos] + content[end_pos + len(end_tag):]

    iac_start_tag, iac_end_tag = "<iac_template>", "</iac_template>"
    iac_start_pos = content.find(iac_start_tag)
    iac_end_pos = content.find(iac_end_tag, iac_start_pos)
    if iac_start_pos != -1 and iac_end_pos != -1:
        return content[iac_start_pos + len(iac_start_tag):iac_end_pos].strip(), "iac_template_tags"

    if iac_type == "terraform":
        lines = content.strip().split("\n")
        if lines and lines[0].startswith("```"):
            fence_end = next(
                (i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith("```")), len(lines)
            )
            return "\n".join(lines[1:fence_end]).strip(), "code_fence"
        return content.strip(), "raw"

    aws_version_pos = content.find("AWSTemplateFormatVersion")
    if aws_version_pos != -1:
        content = content[aws_version_pos:].strip()
        backticks_pos = content.find("```")
        if backticks_pos != -1:
            content = content[:backticks_pos].strip()
        return content, "aws_version_fallback"
    return content.strip(), "raw"
