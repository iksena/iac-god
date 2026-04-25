# agents/retriever.py
from state import GraphState, Message, append_and_cap, compact_message_history
from agents.engineer import _build_client, _call_llm_with_history
from tracking.recorder import ResearchRecorder
from tools.template_annotator import annotate_template, attach_smells
from tools.cfn_hybrid_rag import (
    _extract_errors,
    _build_annotation_summary,
    _parse_query_response,
    QUERY_GEN_SYSTEM,
    _execute_hybrid_retrieval,
)


def _generate_retrieval_queries(
    errors: list[str],
    template_yaml: str | None,
    annotation,
    history: list[Message],
) -> tuple[str, str, list[str], dict | None]:
    """Build the HyDE prompt, call the LLM, and parse retrieval queries."""
    user_parts = ["## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)]
    if annotation and annotation.resources:
        # if annotation.parse_error:
        #     user_parts.append(
        #         "## Template Parse Note\n"
        #         f"Best-effort structural scan was used because YAML parsing failed: {annotation.parse_error}"
        #     )
        user_parts.append(
            "## Template Resource Annotation\n"
            "(Logical IDs, resource types, property keys present, detected smells)\n"
            + _build_annotation_summary(annotation)
        )
    elif template_yaml:
        user_parts.append(
            f"## Template Snippet (for resource type context)\n"
            f"```yaml\n{template_yaml}\n```"
        )

    user_content = "\n\n".join(user_parts)
    user_msg: Message = {"role": "user", "content": user_content}
    messages = compact_message_history(history) + [user_msg]

    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=QUERY_GEN_SYSTEM,
        messages=messages,
    )
    retrieval_queries = _parse_query_response(raw_response)
    if not retrieval_queries:
        retrieval_queries = errors[:8]

    return model, user_content, raw_response, retrieval_queries, usage


def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """
    Dedicated retrieval agent that:
      1. Annotates the current template
      2. Uses LLM to generate HyDE retrieval queries (recorded as LLM call)
      3. Executes ChromaDB + Neo4j retrieval
      4. Returns CFN context + retrieval_queries into state
    """
    iteration = state["current_iteration"]
    print(f"\n[Retriever] Building CFN context (iteration {iteration})...")

    errors = _extract_errors(
        state.get("validation_results", []),
        state.get("deploy_validation_result"),
    )

    annotation = None
    template_yaml = state.get("cloudformation_template", "")
    if template_yaml:
        try:
            annotation = annotate_template(
                file_path="<in-memory>",
                content=template_yaml,
            )
            smell_report = state.get("smell_report")
            if smell_report:
                annotation = attach_smells(annotation, smell_report)
            print(f"[Retriever] Annotation: {len(annotation.resources)} resources parsed.")
        except Exception as exc:
            print(f"[Retriever] Annotation failed (non-fatal): {exc}")
            annotation = None

    model, user_content, raw_response, retrieval_queries, usage = _generate_retrieval_queries(
        errors=errors,
        template_yaml=template_yaml,
        annotation=annotation,
        history=state.get("retriever_history", []),
    )

    llm_record = recorder.record_llm_call(
        state=state,
        agent="retriever",
        model=model,
        prompt=f"SYSTEM:\n{QUERY_GEN_SYSTEM}\n\nUSER:\n{user_content}",
        response=raw_response,
        token_usage=usage,
    )

    cfn_context = _execute_hybrid_retrieval(
        retrieval_queries=retrieval_queries,
        annotation=annotation,
        template_yaml=template_yaml,
    )

    print(
        f"[Retriever] Context: {len(cfn_context)} chars, "
        f"{len(retrieval_queries)} queries used."
    )

    user_msg: Message = {"role": "user", "content": user_content}
    assistant_msg: Message = {"role": "assistant", "content": raw_response}

    return {
        "retriever_context": cfn_context,
        "retriever_queries": retrieval_queries,
        "llm_call_log": state["llm_call_log"] + [llm_record],
        "retriever_history": append_and_cap(
            state.get("retriever_history", []), user_msg, assistant_msg
        ),
    }