# agents/engineer.py
from state import GraphState
from config import DEFAULT_CONFIG, LLMProvider
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM, ENGINEER_USER,
    ENGINEER_REMEDIATION_CONTEXT,
)
from tracking.recorder import ResearchRecorder

def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """
    CGO Stage 2: Code Generation.
    Generates CloudFormation template grounded in objectives (+ remediation if iterating).
    """
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    # Build remediation context if this is a re-generation
    remediation_context = ""
    if state["remediation_history"]:
        latest = state["remediation_history"][-1]
        remediation_context = ENGINEER_REMEDIATION_CONTEXT.format(
            iteration=latest["iteration"],
            previous_template=state["cloudformation_template"],
            remediation_suggestion=latest["suggestion"],
        )

    objectives_text = "\n".join(
        f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"])
    )
    prompt = ENGINEER_USER.format(
        objectives=objectives_text,
        remediation_context=remediation_context,
    )

    content, usage = _call_llm(client, model, ENGINEER_SYSTEM, prompt)

    # Strip markdown fences if model added them
    template = _strip_yaml_fences(content)

    llm_record = recorder.record_llm_call(
        state=state,
        agent="engineer",
        model=model,
        prompt=f"SYSTEM:\n{ENGINEER_SYSTEM}\n\nUSER:\n{prompt}",
        response=content,
        token_usage=usage,
    )

    print(f"[Engineer] Template generated ({len(template.splitlines())} lines).")
    return {
        **state,
        "cloudformation_template": template,
        "llm_call_log": state["llm_call_log"] + [llm_record],
    }

def _strip_yaml_fences(text: str) -> str:
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def _build_client():
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        from openai import OpenAI
        return OpenAI(
            api_key=DEFAULT_CONFIG.openrouter_api_key,
            base_url=DEFAULT_CONFIG.openrouter_base_url,
        ), DEFAULT_CONFIG.model
    else:
        import anthropic
        return anthropic.Anthropic(
            api_key=DEFAULT_CONFIG.anthropic_api_key
        ), DEFAULT_CONFIG.model

def _call_llm(client, model, system, prompt):
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        return r.choices[0].message.content, {
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
        }
    else:
        import anthropic as ant
        r = client.messages.create(
            model=model, system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        return r.content[0].text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
        }