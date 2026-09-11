# Single-Agent Harness Baseline

A control condition for IaCGOD. The same benchmark scenarios, the same
validators, the same scoring — but the multi-agent LangGraph pipeline is
replaced by an off-the-shelf coding harness driving IaCGOD's validators
itself. The harness is pluggable (`--harness claude_code` or `--harness
opencode`, see "Harness drivers" below) so the baseline is not tied to one
vendor's agent or one model family.

The experimental variable is the **orchestration**. Everything downstream of
generation is shared code, not a reimplementation.

## Architecture

Control is inverted relative to `benchmark.py`. Python does not run the repair
loop; the harness does.

```
harness_baseline.py                       mcp_server.py  ──►  evaluation.py
  ├─ selects rows from the CSV                  ▲                  │
  ├─ mints run_id, makes a scratch workdir      │                  ├─ run_all_validators()
  ├─ picks a HarnessDriver (drivers/)           │ MCP             ├─ validate_deployment()
  ├─ driver builds argv/env/config file  ────────────────►  coding harness    ├─ iteration counter + cap
  └─ scores the artifact it left behind        (owns the loop)     └─ ResearchRecorder
```

The MCP server is the instrumentation: it *is* the validator, the recorder, the
iteration counter and the cap enforcer. The harness experiences it as three
ordinary tools, regardless of which harness is driving.

| Tool | Behaviour |
|---|---|
| `validate_iac(file_path)` | Static validation. **One call = one iteration.** Refuses past the cap and tells the harness to submit. |
| `deploy_iac(file_path)` | Live deploy. Gated: refuses unless *this exact content* passed `validate_iac`. Attaches to the current iteration, never opens one. |
| `submit_template(file_path)` | Records the final answer. A claim, not a verdict. |

## Usage

```bash
# CloudFormation, real AWS, DeepSeek V4 Flash through OpenRouter
python -m baselines.harness_baseline \
  --iac-type cloudformation \
  --dataset data/cfn_eval_benchmark_real_aws.csv \
  --model deepseek/deepseek-v4-flash \
  --deploy-target aws --max-iterations 30

# Terraform, static only, a 10-row pilot to calibrate cost
python -m baselines.harness_baseline \
  --iac-type terraform \
  --dataset data/tf_eval_benchmark_real_aws.csv \
  --model deepseek/deepseek-v4-flash \
  --deploy-target none --max-rows 10 --keep-workspace

# Same scenario through OpenCode instead of Claude Code
python -m baselines.harness_baseline \
  --harness opencode \
  --iac-type cloudformation \
  --dataset data/cfn_eval_benchmark_real_aws.csv \
  --model deepseek/deepseek-v4-flash \
  --deploy-target none --max-rows 10 --keep-workspace
```

`--harness` selects the driver (`claude_code`, default, or `opencode`); see
"Harness drivers" below for what differs between them. `--rows`,
`--start-row`, `--max-rows`, `--exclude-completed-csv` and `--retry-errors`
behave exactly as in `benchmark.py` — they share the implementation in
`benchmark_common.py`, harness-independently.

### Model routing

`--model` is mapped onto **every** Claude Code model alias
(`ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL`)
so background, subagent and main-loop calls all route to the model under test.
Without that, a run is not attributable to a single model. Verified: a completed
run reports exactly one key in `modelUsage`.

`--native-auth` skips the override entirely and uses Claude Code's own
authentication and models. OpenCode has no equivalent flag — it always routes
through `--model`/`--base-url`/`--api-key-env`.

## Harness drivers

Everything specific to one coding harness — the CLI invocation, its config
file format, its own state/session isolation, and parsing its event stream
into the common telemetry shape — lives behind the `HarnessDriver` protocol
(`baselines/drivers/base.py`) in one file per harness
(`baselines/drivers/claude_code.py`, `baselines/drivers/opencode.py`). The
scenario loop, retry logic, scoring, and CSV/summary writers in
`harness_baseline.py` never see a harness-specific detail; adding a third
harness means adding a third driver, not touching that code.

Both drivers were verified end-to-end (a full write → validate (fail) → edit
→ validate (pass) → deploy → submit session, scored `PASS`) before being
considered done — not just checked to run without a Python exception.

**`ClaudeCodeDriver`** — the original, most-exercised path; the "Known
asymmetries" and "Root cause of the stalled-session failure mode" sections
below are specific to it (Anthropic-compat OpenRouter routing, Claude Code's
own cost/token accounting, its particular empty-completion failure mode).

**`OpenCodeDriver`** — routes through OpenRouter's native OpenAI-compatible
endpoint (`@ai-sdk/openai-compatible`, `/v1/chat/completions`) rather than the
Anthropic-compat shim Claude Code uses. This matters: the Anthropic-compat
path has a confirmed ~7% (1/15) empty-completion rate across randomly
load-balanced upstream providers (see the root-cause section below); a burst
of 20 calls against the OpenAI-compat path hit 0 failures. Not proof of zero,
but suggestive that OpenCode may not need the same stall mitigation Claude
Code does — **no OpenCode stall signature has actually been observed and
confirmed yet**, unlike Claude Code's. `stalled` in `OpenCodeDriver.parse_stream`
is therefore a structural definition (LLM calls happened, no real content
came back), not a confirmed-real failure mode being guarded against.

Tool access uses OpenCode's own deny-all-then-allowlist permission model
(`agent.iacgod.permission: {"*": "deny", ...}` in the generated
`opencode.json`), verified to actually block a denied tool (an explicit
permission error, not silent fallthrough) — the same guarantee Claude Code's
`--tools`/`--allowedTools` split provides, via a differently-shaped
mechanism. Isolation uses `XDG_DATA_HOME`/`XDG_CONFIG_HOME`/`XDG_STATE_HOME`
redirected under the scenario's scratch workdir (OpenCode's equivalent of
Claude Code's `CLAUDE_CONFIG_DIR`), verified to leave nothing in the real
installation. OpenCode's automatic session-title generation call is pinned to
the model under test via `small_model` in the generated config — without it,
title generation silently used a different vendor's model
(`google/gemini-3.8-flash`), which would have polluted per-model attribution
the same way an unpinned Claude Code subagent alias would.

**Found and fixed while building the OpenCode driver: a subprocess `PWD`
bug in `run_harness()` itself** (harness-agnostic, so it could as easily have
bitten a future third driver). `subprocess.Popen(cmd, cwd=workdir, ...)` sets
the child process's real working directory, but leaves the `PWD` environment
variable it inherits from `os.environ` pointing at wherever the Python
process itself was launched from. Most tools never notice (they call the
real `getcwd()`), but OpenCode reads `process.env.PWD` during startup and,
when it disagreed with the actual cwd, bootstrapped a second, conflicting
instance against the researcher's real repo checkout instead of the isolated
workdir — surfacing as an opaque `{"error":{"name":"UnknownError","message":
"Unexpected server error..."}}` with no further detail. It did not reproduce
manually, because `cd`ing into the workdir in an interactive shell keeps
`PWD` correct automatically; only the Python-spawned subprocess had the
mismatch. Fixed by setting `env["PWD"]` to the resolved workdir right
alongside `cwd=` in `run_harness()`, so the two can never drift apart again,
for any driver.

**Not yet ported for OpenCode: the retry proxy** (`baselines/retry_proxy.py`,
see below) understands only Anthropic-shaped SSE. It is a no-op risk for
OpenCode today only because no confirmed stall mode has forced the question
yet — if one turns up, the proxy will need an OpenAI-shaped (`/v1/chat/completions`
streaming-JSON) parsing path before it can help OpenCode runs the way it
already helps Claude Code runs.

## What is held constant, and what is not

**Held constant** — `run_all_validators()` is the exact function
`agents/validator.py` calls, so stage order, skip semantics and `policy_stats`
are identical. The error text the harness reads is produced by
`agents.engineer._build_simple_fix_errors`, imported rather than reimplemented,
so it is character-identical to what IaCGOD's Engineer reads. One
`validate_iac` call equals one iteration, matching `agents/validator.py`, so
`iterations_used == 1` means the same thing in both systems and `pass@1` is
comparable.

**Deliberately different** — no Planner, so no grounded objectives (the system
prompt in `prompts/` is the Engineer's prompt minus that section); no
Retriever, so no GraphRAG schema context; no Remediator, so no root-cause
analysis step.

## Known asymmetries (Claude Code)

These are measurement caveats, established empirically for `ClaudeCodeDriver`
specifically. Report them alongside any numbers from that harness; they do
not automatically apply to `OpenCodeDriver` (see "Harness drivers" above).

1. **Cost.** Claude Code prices every run with Anthropic rates. For
   `deepseek-v4-flash` it overstated by ~60× (claimed $2.12 against an actual
   $0.035). `total_cost_usd` is therefore discarded; `cost_usd_openrouter` is
   recomputed from token counts and OpenRouter's published rates.

2. **Token counts come from `modelUsage`.** Per-message `usage` returns all
   zeros through OpenRouter's Anthropic-compat endpoint, so the aggregate in
   the result event is used instead. The `token_usage_source` field records
   which path a row used. Baseline rows populate `input_tokens`/`output_tokens`
   while multi-agent OpenRouter rows populate `prompt_tokens`/
   `completion_tokens` — **`token_all_tokens` is the column that compares
   directly.**

3. **No prompt caching** through the shim (`cacheReadInputTokens: 0`), so the
   full context is re-sent every turn. Token totals are dominated by harness
   overhead: a trivial S3-bucket scenario with 3 repair rounds consumed ~400k
   input tokens. This is a real property of the baseline, not an artifact.

4. **Context window.** Claude Code reports `contextWindow: 200000`, assuming
   Anthropic limits rather than DeepSeek's 1M, and will auto-compact earlier
   than necessary. IaCGOD instead caps history at `MAX_HISTORY_PAIRS = 15`.
   Neither bound is the other's.

5. **The harness rationalises.** Observed in testing: given an error it could
   not fix, it declared the finding *"a parsing artifact rather than a template
   issue"* and submitted anyway. `submit_template` is therefore never trusted.
   `final_verdict_source` records how each row was actually scored:

   | Value | Meaning |
   |---|---|
   | `recorded` | Submitted artifact matched its last validation; recorded results reused. |
   | `deployed_at_finalize` | Static passed but the harness never deployed; deployed during scoring. |
   | `revalidated` | Submitted artifact was never validated in that exact form; validated during scoring. |
   | `no_template_produced` | Nothing usable was written. |

   Re-validation at finalize never increments `iterations_used` — it measures
   the artifact, it is not a cycle the harness performed.

6. **Sessions can stall — and Claude Code still reports success.** Observed
   with `deepseek-v4-flash` in a 5-row sample where every row hit this at
   least once: the model returns a turn with no usable content, either
   genuinely empty (no text, no tool call — seen twice, first turn) or by
   emitting its native tool-call syntax as literal text inside a `thinking`
   block instead of a structured `tool_use`, which the OpenRouter shim does
   not parse (seen twice, mid-repair-loop — one case died retrying a deploy
   it had otherwise correctly diagnosed). Claude Code has exactly one
   built-in recovery: a synthetic nudge
   (`"[Your previous response had no visible output...]"`). When that nudge
   also comes back empty, Claude Code ends the session with
   `is_error: false, subtype: "success"` — a session that never produced a
   real turn is indistinguishable, at that level, from one that legitimately
   tried and failed.

   The runner detects this directly off the stream (a synthetic nudge fired,
   and no `tool_use` appeared in any assistant turn afterward) and retries
   with a **fresh** process — up to `--max-stall-retries` (default 1, so 2
   attempts total) — mirroring the empty-completion retry
   `agents/llm_client.py` already gives the multi-agent path. A row that
   never recovers is scored as `status=harness_stalled`, not `ok`: it is not
   a validation failure on the merits, and `pass_rate` should be computed
   after excluding or separately reporting these rows.
   `harness_stall_retries_used` and `harness_stalled_attempt_run_ids` record
   what happened; stalled attempts' partial artifacts stay on disk under
   their own `run_id` for inspection but are not what gets scored.

   **Incidence is high, not occasional.** A 50-row `--start-row 1 --max-rows
   50` sweep hit it on every one of the first 4 rows, and on two of them
   *both* attempts stalled (default `--max-stall-retries 1` gives 2 tries
   total) — the second attempt's log showed the identical DSML-in-thinking
   leak as the first, this time preceded by ~150 escalating
   `thinking_tokens` ticks (a live token-count estimator Claude Code emits
   during a single long API call) before the malformed turn arrived. That is
   suggestive — unusually long reasoning chains correlating with the
   leak — but not confirmed as causal from these logs alone. This is a
   property of `deepseek-v4-flash` routed through the OpenRouter
   Anthropic-compat shim under Claude Code, not a defect in this runner: the
   retry mechanism is working as designed each time (fresh process, bounded,
   exactly `--max-stall-retries + 1` attempts) — raising `--max-stall-retries`
   as high as 5 was tried and did not resolve it; the same stalls kept
   recurring. `--harness-effort low` (passed through as `claude --effort
   low`) was also tried and did not help — it changes reasoning depth, not
   whether thinking is enabled at all.

   **`--max-thinking-tokens 0` is a partial mitigation, not a fix — it does
   not reliably hold.** The leak in both observed failure signatures happens
   specifically *inside* a `thinking` content block — either the model's
   native tool-call syntax leaking out as literal text there, or (a third
   variant found investigating this) the model hallucinating a fake
   `<environment_injection_message>` and reacting to its own fiction, again
   inside thinking. `--max-thinking-tokens 0` sets `MAX_THINKING_TOKENS=0` in
   the harness subprocess's environment, asking Claude Code to disable
   extended thinking outright (verified: a probe call with it set returns a
   plain `text` block, no `thinking` block at all — not a shrunk budget, no
   thinking channel). Tested on the exact diff345 (difficulty 3-5) rows that
   had stalled repeatedly, including across retries — a first batch of 14
   short, fresh-session rows (the 6 that previously failed even after 2-5
   retries, plus 8 more) **all passed with zero stalls**, single attempt
   each (`--max-stall-retries 0`).

   **That result did not hold on a longer session.** A separately-observed
   case on a deep, cache-heavy mid-repair-loop turn
   (`cache_read_input_tokens` in the tens of thousands) burned **10,935
   thinking tokens with `--max-thinking-tokens 0` still set**, and the DSML
   leak recurred inside that thinking block. The most likely explanation:
   `deepseek-v4-flash`'s own always-reasoning behaviour overrides the
   request once it judges a turn hard enough — plausibly correlated with
   conversation depth or cache size, not something this flag can force. The
   14-row batch that looked clean was, in hindsight, biased toward short
   sessions; it was real evidence, just not evidence of a structural fix.

   Two consequences for using this baseline now:

   - **`thinking_tokens_total`** (CSV column, extracted from `modelUsage`)
     and an inline `⚠ thinking=<N>` console warning above
     `THINKING_TOKENS_WARN_THRESHOLD` (3000, in `harness_baseline.py`) exist
     specifically so this doesn't require grepping thousands of raw
     `thinking_tokens` ping lines in `harness_stream.jsonl` by hand — that
     grep is how this gap was found in the first place. Watch that column on
     any sweep; do not assume `--max-thinking-tokens 0` means 0.
   - Pass `--max-thinking-tokens 0` anyway — it measurably helps on short
     sessions and costs nothing when it doesn't — but keep
     `--max-stall-retries` at a real value (not 0) as the actual safety net,
     and keep watching `pass_rate` vs `pass_rate_excl_stalled` in
     `summary.json`. A different model may not share this exact failure mode
     or respond to this lever at all.

   **Update: the root cause behind all of the above was found and fixed —
   see "Root cause of the stalled-session failure mode" below.** Everything
   above stays true as the investigation trail (thinking-block correlation,
   the DSML leak, the environment-injection hallucination all really
   happened, and `--max-thinking-tokens`/`--max-stall-retries` still help)
   but is superseded as the primary defence by the retry proxy.

## Root cause of the stalled-session failure mode, and the fix

Confirmed with a controlled experiment, not inferred: OpenRouter
load-balances `deepseek-v4-flash` across at least eight upstream providers
(StreamLake, GMICloud, DigitalOcean, DeepInfra, Novita, SiliconFlow,
NextBit, Alibaba — seen in practice; there may be more), chosen essentially
at random per request. A burst of 15 raw calls against `/v1/messages` with
no tools, no system prompt, nothing else in play landed on 7 different
providers; 14/15 returned a healthy completion, and the 1 failure
(`provider: SiliconFlow` that run — a **different** provider failed on a
repeat of the same test, confirming it is not one specific bad backend)
burned its entire output-token budget on a `thinking` block and returned
zero visible content, `stop_reason: "max_tokens"`. That is the exact
stalled-session signature, reproduced outside Claude Code entirely: fast,
HTTP 200, healthy-looking — just empty.

This explains the whole shape of the earlier investigation: why stalls
looked random (which of ~8 providers you draw *is* random), why retries
sometimes worked and sometimes didn't (restarting the whole harness process
just draws again — usually enough, but unlucky runs of bad draws happen),
and why `--max-thinking-tokens 0` helped on some sessions and not others (if
different providers honour the request differently, which provider you
happened to draw would explain the inconsistency).

**The fix: `baselines/retry_proxy.py`**, a small local reverse proxy that
sits between the harness and OpenRouter. `ANTHROPIC_BASE_URL` points at it
instead of directly at OpenRouter; every request passes through unchanged,
but when a response comes back with no real content (no `tool_use`, no
non-empty `text` — exactly the confirmed signature) it transparently
re-issues the *same* request before Claude Code ever sees the failure. Each
retry draws a fresh provider from OpenRouter's own load balancer, so most
failures resolve invisibly. Verified in a direct test (10 calls through the
proxy): 2 hit the empty-completion signature on the first attempt and were
both silently retried and resolved on the second — the caller (my test
script, standing in for Claude Code) saw 10/10 healthy responses and never
knew 2 of them needed a retry. On the two real scenarios that had failed
**100% of the time — all 6 retries, both rows** in an actual benchmark run
(the concrete case that prompted this investigation), both passed cleanly
on the first attempt with the proxy in place.

On by default (`--no-retry-proxy` to disable; `--retry-proxy-max-retries`,
default 3, to tune). One proxy process is shared for the whole benchmark
run — it is stateless, so there is no reason to restart it per scenario —
and its decisions are logged to `runs/retry_proxy_<timestamp>.log`
(`{"event": "attempt", "healthy": bool, "provider": str, ...}` per API call,
plus a `"resolved"` line whenever more than one attempt was needed). Ignored
under `--native-auth`, since that routes to Claude's own API, not
OpenRouter, and the retry-worthy failure mode is specific to OpenRouter's
provider load-balancing.

Distinct from `--max-stall-retries`, not a replacement for it: this retries
one HTTP request at a fraction of the cost of restarting the whole harness
process, and should catch the large majority of cases before
`--max-stall-retries` is ever needed. Keep `--max-stall-retries` at a
non-zero value regardless — retrying a single request 3 times only helps
when *some* provider on the retry list is healthy; if OpenRouter itself has
a bad few minutes across the board, a full process restart (drawn at a
different time) is the deeper fallback.

## Output

Identical to `benchmark.py`, so the existing aggregation scripts read both.

- `benchmark_runs/<name>/results.csv` — `CSV_RESULT_FIELDS`' 18 columns in
  order, then the baseline-specific columns.
- `benchmark_runs/<name>/results.jsonl`, `results.json`, `summary.json`
- `runs/<run_id>/` — `iteration_NNN.json`, `final_report.json`,
  `deployment_log_NNN.txt`, written by the same `ResearchRecorder`. Plus
  `baseline_state.json` (the MCP server's ledger), `harness_stream.jsonl`
  (the raw stream-json), and two debug files (see below):
  `harness_transcript.txt` and `harness_tool_calls.txt`.

`objectives` and `remediation_history` are empty in baseline snapshots by
construction: the baseline has no Planner and no Remediator. That absence is
the experimental condition, not missing data.

## Debug transcripts

Every scenario attempt writes two human-readable `.txt` files alongside the
raw `harness_stream.jsonl`, built independently of the scoring/token-counting
logic so a change there can never silently change what gets recorded:

- **`harness_transcript.txt`** — the full conversation in order: the system
  prompt, the initial user prompt, then every thinking block, text block,
  tool call (with its full arguments — e.g. the exact template content on a
  `Write` call), and tool result, exactly as they occurred, ending with the
  session's final result summary.
- **`harness_tool_calls.txt`** — the same session reduced to just the
  tool calls and their results, for scanning a run without reading the whole
  transcript.

Both are written for every attempt, including stalled or failed ones — a
transcript showing exactly nothing happened, or where a session broke off, is
itself the debugging signal.

**`harness_debug.log`** — the one artifact that shows what happens *below*
the content layer, via Claude Code's own `--debug api --debug-file`. The
transcript files only ever see what Claude Code decided to hand back as
message content, which cannot distinguish two very different failures: a
request that never got a response at all (infra/network layer) from one
that got a response the model/shim produced as empty (content layer). This
log can, from three line types per API call:

```
[DEBUG] [API REQUEST] /api/v1/messages source=sdk
[DEBUG] Stream started - received first chunk
[DEBUG] [API:timing] first byte after <N>ms
```

Reading a stalled session's log: if "Stream started" is **missing** after
the last `API REQUEST` line, the connection never got a response at all —
that is an infrastructure failure (network, timeout, rejected request), not
a model output problem, and `--max-thinking-tokens`/`--max-stall-retries`
would not be the right lever for it. If "Stream started" **is** present but
the corresponding turn in `harness_stream.jsonl` is still empty, the
model/shim produced a response Claude Code parsed as empty — that is the
DSML-leak/empty-completion failure mode in the point above, and the
existing mitigations apply. Written for every attempt, same as the
transcripts.

## Isolation (Claude Code)

The harness subprocess must not pick up the researcher's own Claude Code
state, and the model it routes to must be reliably injectable rather than
whatever the ambient environment happens to resolve. Two mechanisms, both
empirically verified (not assumed):

- **`CLAUDE_CONFIG_DIR`** is set to a scenario-scoped temp directory
  (`workdir/.claude_config`). Verified: with it set, Claude Code writes its
  entire session/project state under that directory and nothing touches the
  real `~/.claude`.
- **`--setting-sources ""`** loads no user/project/local `settings.json` —
  which is where hooks, custom agents, and output styles are configured — and
  was separately verified to block this repo's own `AGENTS.md` from being
  auto-discovered as project context: a probe run explicitly asked whether it
  had been told about any multi-agent architecture reported no awareness of
  it.

**`--bare` was tried and deliberately dropped.** It gives stronger guarantees
on paper (no keychain reads, no plugin sync, no CLAUDE.md auto-discovery
either) and was confirmed compatible with the `ANTHROPIC_AUTH_TOKEN` routing
below — but its own baseline tool set has **no `Write` tool at all**
(confirmed: `--tools Write` under `--bare` is rejected as an unrecognized
name; `Write` exists only outside `--bare`). Without a `Write` tool, the
harness naturally reached for `Bash` (`cat > file <<EOF`) to create the
template instead — which reopens a real methodological risk this baseline
otherwise closes: a harness with `Bash` can validate or deploy by shelling
out directly (e.g. running `cfn-lint` or `aws cloudformation deploy` itself),
bypassing `validate_iac`/`deploy_iac` entirely and breaking the "one
`validate_iac` call == one iteration" comparability the whole design depends
on. `--setting-sources ""` alone already covered the CLAUDE.md-leak concern
`--bare` was mainly wanted for, so `--bare` was not worth trading `Write`
away for.

`--tools "Read,Write,Edit"` is what actually restricts the built-in tool set
— **`--allowedTools` alone does not**: a model given only `--allowedTools`
(with no `--tools`) still sees, and freely uses, `Bash`. Both flags are
passed together: `--tools` sets the built-in roster (no `Bash`, ever),
`--allowedTools` pre-approves the specific tools (the three MCP tools plus
`Read`/`Write`/`Edit`) so `--permission-mode acceptEdits` doesn't need to
prompt for any of them.

## Operational warnings

- **Real-AWS runs must stay serial.** `validate_deployment` performs an
  account-wide greenfield reset — it deletes every non-default VPC and sweeps
  `iac-god-eval-*` stacks. Run against the dedicated research account only, and
  never two runs at once.
- **Budget.** No usable spend cap exists: Claude Code's `--max-budget-usd` is
  computed with Anthropic pricing and is meaningless here. The bounds are the
  server-side iteration cap and `--scenario-timeout` (default 1800s). Pilot
  with `--max-rows 10` before committing to a full 250-row sweep.
- Workspaces are deleted after scoring unless `--keep-workspace` is passed.
