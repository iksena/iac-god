"""stdio MCP server exposing IaCGOD's validators as tools for a coding harness.

Spawned by the harness (Claude Code) once per scenario via a generated
.mcp.json; scenario context arrives through that file's env block. All the
logic lives in baselines/evaluation.py — this module is only the JSON-RPC
transport.

Implemented against the wire protocol directly rather than the `mcp` SDK: the
surface needed is three methods, and this repo pins langchain/chromadb, so
adding an SDK with its own pydantic constraints risks destabilising the
multi-agent pipeline the baseline exists to measure.

stdout is reserved exclusively for JSON-RPC frames. The validators and the
recorder both print progress (see tools/validators.validate_tflint,
tools/deploy_validator), so the real stdout is captured at import time and the
module-level sys.stdout is redirected to stderr — any stray print() from the
imported stack lands in the harness's stderr log instead of corrupting the
protocol stream.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

# --- stdout protection, before importing anything that might print ---------
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.evaluation import ScenarioConfig, ScenarioLedger, template_filename  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "iacgod-eval", "version": "1.0.0"}


def _tool_definitions(iac_type: str) -> list[dict[str, Any]]:
    fname = template_filename(iac_type)
    lang = "Terraform HCL" if iac_type == "terraform" else "CloudFormation YAML"
    path_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": f"Absolute path to the {fname} file.",
            }
        },
        "required": ["file_path"],
    }
    return [
        {
            "name": "validate_iac",
            "description": (
                f"Run static validation on a {lang} file and return every error found. "
                "Each call counts as one iteration against this scenario's budget. "
                "Call this after every edit until it reports PASSED."
            ),
            "inputSchema": path_schema,
        },
        {
            "name": "deploy_iac",
            "description": (
                "Deploy the template to verify it actually provisions. Only available "
                "once validate_iac has passed for the template's current contents; "
                "does not count as an extra iteration."
            ),
            "inputSchema": path_schema,
        },
        {
            "name": "submit_template",
            "description": (
                "Submit the final template and end work on this scenario. Call this "
                "once validation and deployment pass, or immediately if told the "
                "iteration cap has been reached."
            ),
            "inputSchema": path_schema,
        },
    ]


def _send(payload: dict[str, Any]) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_OUT.flush()


def _result(rid: Any, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": rid, "result": result})


def _error(rid: Any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def _text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _dispatch_tool(ledger: ScenarioLedger, name: str, args: dict[str, Any]) -> dict[str, Any]:
    file_path = args.get("file_path")
    if not file_path:
        return _text_result("file_path is required.", is_error=True)

    if name == "validate_iac":
        outcome = ledger.validate(file_path)
    elif name == "deploy_iac":
        outcome = ledger.deploy(file_path)
    elif name == "submit_template":
        outcome = ledger.submit(file_path)
    else:
        return _text_result(f"Unknown tool: {name}", is_error=True)

    # isError is reserved for protocol/tool faults. A failing validation is a
    # successful tool call reporting a negative result: flagging it as an error
    # makes some harnesses retry the call rather than fix the template.
    return _text_result(outcome.text, is_error=False)


def main() -> int:
    try:
        config = ScenarioConfig.from_env()
    except Exception as exc:
        print(f"[mcp_server] fatal: {exc}", file=sys.stderr)
        return 1

    ledger: ScenarioLedger | None = None
    tools = _tool_definitions(config.iac_type)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method")
        rid = request.get("id")
        is_notification = rid is None

        try:
            if method == "initialize":
                requested = (request.get("params") or {}).get("protocolVersion")
                _result(
                    rid,
                    {
                        "protocolVersion": requested or PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": SERVER_INFO,
                    },
                )
            elif method == "notifications/initialized":
                # Defer ledger construction until the harness is live, so a
                # probe that only performs a handshake leaves no run artifacts.
                if ledger is None:
                    ledger = ScenarioLedger(config=config)
            elif method == "ping":
                _result(rid, {})
            elif method == "tools/list":
                _result(rid, {"tools": tools})
            elif method == "tools/call":
                if ledger is None:
                    ledger = ScenarioLedger(config=config)
                params = request.get("params") or {}
                _result(
                    rid,
                    _dispatch_tool(
                        ledger,
                        params.get("name", ""),
                        params.get("arguments") or {},
                    ),
                )
            elif is_notification:
                continue
            else:
                _error(rid, -32601, f"Method not found: {method}")
        except Exception as exc:  # never let one bad call kill the scenario
            print(
                f"[mcp_server] error handling {method}: {exc}\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            if not is_notification:
                _error(rid, -32603, f"Internal error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
