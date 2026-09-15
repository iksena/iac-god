"""Local reverse proxy that transparently retries empty/degenerate completions.

Root cause (confirmed empirically, see baselines/README.md): OpenRouter
load-balances deepseek-v4-flash across at least 8 upstream providers
(StreamLake, GMICloud, DigitalOcean, DeepInfra, Novita, SiliconFlow, NextBit,
Alibaba, ...), chosen essentially at random per request. In a controlled
15-call burst against the raw /v1/messages endpoint, 14/15 (~93%) returned a
healthy completion; the one failure (SiliconFlow) burned its entire output
budget on a "thinking" block and returned zero visible content. This matches
the stalled-session signature exactly: fast, healthy-looking HTTP response,
just empty.

Rather than pin a single provider (which would change what is actually being
benchmarked — "deepseek-v4-flash via OpenRouter" includes this provider
heterogeneity) or rely on Claude Code's own one-shot "no visible output"
nudge (which asks the SAME flaky draw to try again, not a fresh one), this
proxy sits between Claude Code and OpenRouter, and when a response comes back
with no real content (no tool_use, no non-empty text — i.e. exactly the
signature above), transparently re-issues the SAME request before Claude Code
ever sees the failure. Each retry gets a fresh provider draw from OpenRouter,
so retrying is expected to resolve the large majority of cases.

Usage: point ANTHROPIC_BASE_URL at this proxy instead of directly at
OpenRouter (http://127.0.0.1:<port> in place of https://openrouter.ai/api);
everything else about the request/response is passed through unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

UPSTREAM_BASE = "https://openrouter.ai/api"

# Headers that must not be forwarded verbatim (either hop-by-hop, or
# recomputed by `requests`/the server itself for the new destination).
_STRIP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_STRIP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection"}


def _parse_sse_message(raw: bytes) -> dict[str, Any] | None:
    """Reconstruct the final assembled message from a raw SSE byte stream.

    Only extracts what is needed to judge emptiness (content block types and
    text), not a full reimplementation of the Anthropic streaming protocol.
    Returns None if the bytes are not a recognisable SSE stream (caller then
    falls back to treating it as a plain JSON body).
    """
    text = raw.decode("utf-8", errors="replace")
    if "event:" not in text:
        return None

    blocks: dict[int, dict[str, Any]] = {}
    stop_reason = None
    provider = None
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("event:"):
            continue
        lines = block.split("\n")
        event_type = lines[0].split(":", 1)[1].strip()
        data_line = next((l for l in lines[1:] if l.startswith("data:")), None)
        if not data_line:
            continue
        try:
            data = json.loads(data_line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            continue

        if event_type == "message_start":
            provider = (data.get("message") or {}).get("provider")
        elif event_type == "content_block_start":
            idx = data.get("index", 0)
            cb = data.get("content_block") or {}
            blocks[idx] = {"type": cb.get("type"), "text": cb.get("text", ""), "has_input": bool(cb.get("input"))}
        elif event_type == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta") or {}
            if idx not in blocks:
                blocks[idx] = {"type": None, "text": "", "has_input": False}
            if delta.get("type") == "text_delta":
                blocks[idx]["text"] += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                if delta.get("partial_json"):
                    blocks[idx]["has_input"] = True
        elif event_type == "message_delta":
            stop_reason = (data.get("delta") or {}).get("stop_reason", stop_reason)

    return {"blocks": list(blocks.values()), "stop_reason": stop_reason, "provider": provider}


def _is_healthy(parsed: dict[str, Any] | None, fallback_json: dict[str, Any] | None) -> bool:
    """A completion is healthy iff it has a real tool_use or non-empty text.

    Thinking-only (or fully empty) content — the confirmed stall signature —
    is unhealthy regardless of how many tokens it spent getting there.
    """
    blocks: list[dict[str, Any]] = []
    if parsed is not None:
        blocks = parsed["blocks"]
    elif fallback_json is not None:
        for cb in fallback_json.get("content") or []:
            blocks.append(
                {
                    "type": cb.get("type"),
                    "text": cb.get("text", ""),
                    "has_input": bool(cb.get("input")),
                }
            )
    else:
        return False

    for b in blocks:
        if b.get("type") == "tool_use" and (b.get("has_input") or b.get("type") == "tool_use"):
            return True
        if b.get("type") == "text" and b.get("text", "").strip():
            return True
    return False


class RetryProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # class-level config, set by main() before the server starts
    max_retries: int = 3
    log_path: Path | None = None
    request_timeout: float = 120.0

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr noise
        pass

    def _log(self, event: dict[str, Any]) -> None:
        if not self.log_path:
            return
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def do_POST(self) -> None:  # noqa: N802 (http.server naming convention)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        forward_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS
        }
        target_url = UPSTREAM_BASE + self.path

        try:
            req_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req_json = None

        last_raw: bytes = b""
        last_status = 502
        last_headers: dict[str, str] = {}
        attempts = 0

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                resp = requests.post(
                    target_url,
                    data=body,
                    headers=forward_headers,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                self._log({"event": "request_error", "attempt": attempts, "error": str(exc)})
                last_status, last_raw = 502, json.dumps({"error": str(exc)}).encode()
                if attempt < self.max_retries:
                    continue
                break

            last_status = resp.status_code
            last_raw = resp.content
            last_headers = dict(resp.headers)

            if resp.status_code != 200:
                self._log(
                    {
                        "event": "upstream_error",
                        "attempt": attempts,
                        "status": resp.status_code,
                        "body": last_raw.decode("utf-8", "replace")[:500],
                    }
                )
                # 5xx/429 are transient — same family as a connection error,
                # worth retrying. Other 4xx (bad request, auth) indicate a
                # real client-side problem retrying would not fix.
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    continue
                break

            parsed_sse = _parse_sse_message(last_raw)
            fallback_json = None
            if parsed_sse is None:
                try:
                    fallback_json = json.loads(last_raw)
                except json.JSONDecodeError:
                    fallback_json = None

            healthy = _is_healthy(parsed_sse, fallback_json)
            provider = (
                (parsed_sse or {}).get("provider")
                if parsed_sse
                else (fallback_json or {}).get("provider")
            )
            self._log(
                {
                    "event": "attempt",
                    "attempt": attempts,
                    "healthy": healthy,
                    "provider": provider,
                    "stop_reason": (parsed_sse or {}).get("stop_reason")
                    if parsed_sse
                    else (fallback_json or {}).get("stop_reason"),
                }
            )

            if healthy or attempt >= self.max_retries:
                break

        if attempts > 1:
            self._log({"event": "resolved", "attempts_used": attempts, "final_status": last_status})

        self.send_response(last_status)
        for k, v in last_headers.items():
            if k.lower() not in _STRIP_RESPONSE_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(last_raw)))
        self.end_headers()
        self.wfile.write(last_raw)

    def do_GET(self) -> None:  # noqa: N802
        # Pass-through for any non-POST calls (model listing, health checks).
        forward_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS
        }
        resp = requests.get(UPSTREAM_BASE + self.path, headers=forward_headers, timeout=self.request_timeout)
        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() not in _STRIP_RESPONSE_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp.content)))
        self.end_headers()
        self.wfile.write(resp.content)


def serve(port: int, max_retries: int, log_path: Path | None) -> ThreadingHTTPServer:
    RetryProxyHandler.max_retries = max_retries
    RetryProxyHandler.log_path = log_path
    server = ThreadingHTTPServer(("127.0.0.1", port), RetryProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=0, help="0 = pick a free port and print it")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--log-file", type=Path, default=None)
    args = p.parse_args()

    server = serve(args.port or 0, args.max_retries, args.log_file)
    actual_port = server.server_address[1]
    print(actual_port, flush=True)  # for the parent process to capture
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
