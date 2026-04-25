"""Shared prompt-context builders for remediation history.

Extracted so both engineer.py and remediator.py can import
_build_remediation_history_context without a circular dependency.
"""
from __future__ import annotations

from state import RemediationHistory


def _build_remediation_history_context(remediation_history: list[RemediationHistory]) -> str:
    """Build a structured, read-only history block from past RemediationHistory entries.

    Injected as a compact document so both the Remediator and Engineer agents
    can avoid repeating failed strategies without the token overhead of verbatim
    conversation transcripts.
    """
    if not remediation_history:
        return ""

    lines: list[str] = [
        "## Prior Remediation Attempts",
        "The following fix strategies were already attempted. Do NOT repeat them.",
        "Use this history to choose a different approach if a strategy failed or was insufficient.",
        "",
    ]

    for entry in remediation_history:
        iteration = entry["iteration"]
        lines.append(f"### Attempt {iteration} ({entry['timestamp'][:10]})")

        if entry.get("formatted_errors"):
            error_summary = entry["formatted_errors"][:800]
            if len(entry["formatted_errors"]) > 800:
                error_summary += "\n  ... (truncated)"
            lines.append(f"**Errors present:**\n{error_summary}")

        if entry.get("suggestion"):
            suggestion_summary = entry["suggestion"][:600]
            if len(entry["suggestion"]) > 600:
                suggestion_summary += "\n  ... (truncated)"
            lines.append(f"**Fix objectives suggested:**\n{suggestion_summary}")

        if entry.get("retrieval_queries"):
            queries_str = ", ".join(f'"{q}"' for q in entry["retrieval_queries"])
            lines.append(f"**Retrieval queries used:** {queries_str}")

        lines.append("")

    return "\n".join(lines).strip()
