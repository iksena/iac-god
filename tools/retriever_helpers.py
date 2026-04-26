"""tools/retriever_helpers.py

State-extraction and LLM-response parsing utilities used exclusively by the
retriever agent.

Extracted from cfn_hybrid_rag.py so that:
- cfn_hybrid_rag.py owns only DB connectivity and retrieval logic.
- retriever.py owns only LLM orchestration and prompt assembly.
- These helpers, which depend only on state shape and stdlib, have no
  heavyweight dependencies and can be unit-tested without DB fixtures.
"""
from __future__ import annotations

import json
import re

# Validation stages that emit security policy violations.
# Excluded from CFN schema retrieval — the remediator already knows the fix
# (e.g. enable encryption) without needing property-level schema context.
SECURITY_STAGES = {"checkov", "trivy"}


def _extract_errors(
    validation_results: list[dict],
    deploy_validation_result: dict | None,
) -> list[str]:
    """Extract a flat list of error strings from the validation state.

    Security stages (checkov, trivy) are excluded — their findings are policy
    violations, not schema errors, and do not require CFN schema retrieval.
    """
    errors: list[str] = []

    for result in validation_results:
        stage = str(result.get("stage") or "").strip().lower()
        if stage in SECURITY_STAGES:
            continue
        if not result.get("passed"):
            for err in result.get("errors", []):
                if str(err).strip():
                    errors.append(str(err))

    if deploy_validation_result and not deploy_validation_result.get("passed"):
        if deploy_validation_result.get("error_message"):
            errors.append(deploy_validation_result["error_message"])
        for fr in deploy_validation_result.get("failed_resources", []):
            name = fr.get("logical_name") or fr.get("resource") or ""
            reason = fr.get("status_reason") or fr.get("reason") or ""
            if name or reason:
                errors.append(f"{name} {reason}".strip())

    return errors


def _parse_query_response(raw: str, max_queries: int = 8) -> list[str]:
    """Parse the LLM's query-generation response.

    Accepts both {"queries": [...]} object form and bare [...] array form.
    Strips markdown fences before parsing.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[RAG Tool] Query parse error (JSONDecodeError): {e}. Raw: {cleaned[:200]}")
        return []

    if isinstance(parsed, dict):
        queries = parsed.get("queries") or parsed.get("query") or []
    elif isinstance(parsed, list):
        queries = parsed
    else:
        print(f"[RAG Tool] Unexpected query response type: {type(parsed)}")
        return []

    if not isinstance(queries, list):
        print(f"[RAG Tool] 'queries' field is not a list: {queries}")
        return []

    result = [str(q).strip() for q in queries if str(q).strip()][:max_queries]
    print(f"[RAG Tool] Parsed {len(result)} retrieval queries from LLM response.")
    return result