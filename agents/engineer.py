# agents/engineer.py
from state import GraphState, Message, compact_message_history
from config import DEFAULT_CONFIG, LLMProvider
from prompts.engineer_prompt import (
    ENGINEER_SYSTEM, ENGINEER_USER_INITIAL,
    ENGINEER_USER_REMEDIATION,
)
from tracking.recorder import ResearchRecorder


def engineer_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    iteration = state["current_iteration"]
    print(f"\n[Engineer] Generating CFN template (iteration {iteration})...")

    client, model = _build_client()

    system = ENGINEER_SYSTEM.format(
        objectives="\n".join(f"{i+1}. {obj}" for i, obj in enumerate(state["objectives"]))
    )

    is_remediation = bool(state["remediation_history"])

    if not is_remediation:
        # Iteration 1: simple generation request — no history yet
        user_content = ENGINEER_USER_INITIAL
    else:
        # Iteration 2+: only carry NEW information — the latest fix directive
        # The previous template is already in engineer_history[-1] assistant turn
        # The suggestion text is already in engineer_history[-2] user turn (via prior remediation context)
        latest = state["remediation_history"][-1]
        user_content = ENGINEER_USER_REMEDIATION.format(
            iteration=latest["iteration"],
            remediation_suggestion=latest["suggestion"],
        )

    user_msg: Message = {"role": "user", "content": user_content}

    # Full accumulated history + new user turn (no duplication — history carries prior turns)
    messages = compact_message_history(state["engineer_history"]) + [user_msg]

    content, usage = _call_llm_with_history(client, model, system, messages)
    template = _strip_yaml_fences(content)
    assistant_msg: Message = {"role": "assistant", "content": content}

    llm_record = recorder.record_llm_call(
        state=state,
        agent="engineer",
        model=model,
        prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user_content}",
        response=content,
        token_usage=usage,
    )

    print(f"[Engineer] Template generated ({len(template.splitlines())} lines).")
    return {
        **state,
        "cloudformation_template": template,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "engineer_history": [user_msg, assistant_msg],
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


def _call_llm_with_history(client, model, system, messages):
    """Call LLM with full message history (multi-turn)."""
    if DEFAULT_CONFIG.provider == LLMProvider.OPENROUTER:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
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
            messages=messages,
            temperature=DEFAULT_CONFIG.temperature,
            max_tokens=DEFAULT_CONFIG.max_tokens,
        )
        return r.content[0].text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
        }