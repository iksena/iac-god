# Single-Agent Harness Baseline

A control condition for IaCGOD. The same benchmark scenarios, the same
validators, the same scoring — but the multi-agent LangGraph pipeline is
replaced by an off-the-shelf coding harness (Claude Code) driving IaCGOD's
validators itself.

The experimental variable is the **orchestration**. Everything downstream of
generation is shared code, not a reimplementation.

## Architecture

Control is inverted relative to `benchmark.py`. Python does not run the repair
loop; the harness does.

```
harness_baseline.py                       mcp_server.py  ──►  evaluation.py
  ├─ selects rows from the CSV                  ▲                  │
  ├─ mints run_id, makes a scratch workdir      │                  ├─ run_all_validators()
  ├─ writes .mcp.json + model env               │ MCP             ├─ validate_deployment()
  ├─ launches `claude -p`  ────────────────►  Claude Code          ├─ iteration counter + cap
  └─ scores the artifact it left behind        (owns the loop)     └─ ResearchRecorder
```

The MCP server is the instrumentation: it *is* the validator, the recorder, the
iteration counter and the cap enforcer. The harness experiences it as three
ordinary tools.

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
```

`--rows`, `--start-row`, `--max-rows`, `--exclude-completed-csv` and
`--retry-errors` behave exactly as in `benchmark.py` — they share the
implementation in `benchmark_common.py`.

### Model routing

`--model` is mapped onto **every** Claude Code model alias
(`ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL`)
so background, subagent and main-loop calls all route to the model under test.
Without that, a run is not attributable to a single model. Verified: a completed
run reports exactly one key in `modelUsage`.

`--native-auth` skips the override entirely and uses Claude Code's own
authentication and models.

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

## Known asymmetries

These are measurement caveats, established empirically. Report them alongside
any numbers.

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
   exactly `--max-stall-retries + 1` attempts), it is just frequently not
   enough. Two consequences for using this baseline:

   - `summary.json` reports both `pass_rate` (over all evaluated rows,
     matching `benchmark.py`'s denominator for direct comparability) and
     `pass_rate_excl_stalled` (over rows that got a fair attempt, i.e.
     excluding `rows_stalled`). Report both — the gap between them is itself
     a finding, not noise to average away.
   - Raise `--max-stall-retries` above the default 1 for a real sweep, and
     expect the wall-clock and OpenRouter-cost total to grow accordingly —
     each retry is a full fresh process, deploy included. `--harness-effort
     low` (passed through as `claude --effort low`) is offered as an
     untested lever worth trying against the correlation above; it is not
     confirmed to reduce the incidence.

## Output

Identical to `benchmark.py`, so the existing aggregation scripts read both.

- `benchmark_runs/<name>/results.csv` — `CSV_RESULT_FIELDS`' 18 columns in
  order, then the baseline-specific columns.
- `benchmark_runs/<name>/results.jsonl`, `results.json`, `summary.json`
- `runs/<run_id>/` — `iteration_NNN.json`, `final_report.json`,
  `deployment_log_NNN.txt`, written by the same `ResearchRecorder`. Plus
  `baseline_state.json` (the MCP server's ledger) and `harness_stream.jsonl`
  (the full harness transcript).

`objectives` and `remediation_history` are empty in baseline snapshots by
construction: the baseline has no Planner and no Remediator. That absence is
the experimental condition, not missing data.

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
