# IaCGOD — Infrastructure-as-Code Generation and Optimization with Delivery

IaCGOD is a multi-agent LLM system that generates **production-deployable** Infrastructure-as-Code from a natural-language request. It targets both **AWS CloudFormation** (YAML) and **HashiCorp Terraform** (HCL), and closes the loop from generation to *verified deployability*: every candidate template is statically linted, security-scanned, and — optionally — actually deployed to LocalStack or live AWS before the run is accepted as passing.

The system is built around a cyclic **LangGraph** state machine that iterates a generate → validate → repair loop, escalating from cheap self-correction to a full retrieval-augmented root-cause-analysis pipeline only when errors persist. Repair grounding comes from a **dual-stream hybrid GraphRAG** design: a ChromaDB semantic-search stream over provider documentation, and a Neo4j deterministic-schema stream over resource/attribute graphs, fused into a single remediation context.

Evaluated on the **DPIaC-Eval** benchmark (~300 real-world infrastructure scenarios spanning CloudFormation and Terraform), IaCGOD's full pipeline reaches over 97% deployability pass rate.

> This README documents the codebase as it exists in this checkout (branch `feature/terraform`). See [`AGENTS.md`](AGENTS.md) for the condensed contributor/agentic-coding guide.

## Table of Contents

- [Architecture](#architecture)
- [The five agents](#the-five-agents)
- [Orchestration: the LangGraph state machine](#orchestration-the-langgraph-state-machine)
- [Dual-stream hybrid GraphRAG](#dual-stream-hybrid-graphrag)
- [Validation & deployment toolchain](#validation--deployment-toolchain)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Usage](#usage)
- [Benchmark data](#benchmark-data)
- [Configuration reference](#configuration-reference)

## Architecture

```
                     ┌───────────┐
        user_request │  Planner  │  Grounded Objectives (numbered, structured)
                     └─────┬─────┘
                           │
                           ▼
                     ┌───────────┐
              ┌─────▶│ Engineer  │  Generates / repairs IaC template
              │      └─────┬─────┘
              │            │
              │            ▼
              │      ┌───────────┐
              │      │ Validator │  cfn-lint/tflint, checkov, trivy, LocalStack/AWS
              │      └─────┬─────┘
              │            │ route_after_validator()
              │  ┌─────────┼───────────────┐
              │  │         │               │
              │  ▼         ▼               ▼
              │ end   engineer_simple   [moderate mode]
              │ (pass  (simple self-        │
              │ /max   correction)          ▼
              │ iters)                ┌───────────┐
              │                       │ Retriever │  HyDE queries + hybrid RAG
              │                       └─────┬─────┘
              │                             ▼
              │                       ┌────────────┐
              └───────────────────────┤ Remediator │  RCA + fix objectives
                                       └────────────┘
```

The graph is defined in [`graph.py`](graph.py) using `langgraph.graph.StateGraph`, checkpointed with `MemorySaver`. Shared state flows through a single `GraphState` `TypedDict` defined in [`state.py`](state.py) — every agent reads and returns partial updates to this one object, and LangGraph merges them via its state reducers.

## The five agents

All five agents live in [`agents/`](agents/) and are LangGraph nodes. Each is a plain function `(state: GraphState, recorder: ResearchRecorder) -> GraphState` bound into the graph with `functools.partial`.

| Agent | File | Role |
|---|---|---|
| **Planner** | `agents/planner.py` | Runs once per pipeline, at the entry point. Takes the natural-language `user_request` and produces a numbered list of **Grounded Objectives** — the CGO ("Chain of Grounded Objectives") stage-1 decomposition that anchors everything downstream. |
| **Engineer** | `agents/engineer.py` | Generates or repairs the actual IaC template (CFN YAML or TF HCL). Operates in one of three prompt paths depending on pipeline state: **Path A** (first generation, from the Planner's objectives), **Path B** (simple self-correction, direct from raw validator errors — used in "simple mode"), and **Path C** (moderate remediation, consuming the Remediator's root-cause analysis and fix objectives — used once a stage escalates past the simple-mode threshold). The graph wires both `engineer` and `engineer_simple` to the same underlying function; the active path is detected from state, not from which graph node was entered. |
| **Validator** | `agents/validator.py` | Runs the full static + live validation pipeline against the current template: linting (`cfn-lint`/`tflint`/`terraform validate`), security scanning (`checkov`, `trivy`), and — if a deploy target is configured — an actual deployment attempt against LocalStack or AWS via `tools/deploy_validator.py`. Populates `validation_results`, `deploy_validation_result`, and increments `stage_error_counts`. |
| **Retriever** | `agents/retriever.py` | Triggered only once a failing stage enters "moderate mode" (see below). Generates Hypothetical Document Embedding (HyDE) queries with an LLM, extracts the specific failing resource IDs out of raw linter/deploy errors via regex, and queries the offline hybrid knowledge bases for grounded schema context (`tools/cfn_hybrid_rag.py` / `tools/tf_hybrid_rag.py`). |
| **Remediator** | `agents/remediator.py` | Consumes the Retriever's schema context plus formatted validation errors, produces a root-cause analysis (RCA) and a concrete fix plan, and hands it to the Engineer (Path C) for the next repair iteration. Also pulls in deterministic security remediation context via `tools/security_hybrid_rag.py` when policy findings are present. |

## Orchestration: the LangGraph state machine

The routing logic lives entirely in [`route_after_validator()`](graph.py) and is driven by per-stage error counters, not just a global iteration count:

1. **Pass or iteration cap** — if `validation_passed` is `True`, or `current_iteration >= max_iterations`, the graph ends.
2. **Stage classification** — `classify_failing_stages()` (in `state.py`) groups every failing validator stage into one of three buckets via `STAGE_GROUPS`:
   - `syntax` — `yaml`, `cfn-lint`, `tflint`, `terraform-validate`
   - `security` — `checkov`, `trivy`
   - `deploy` — live LocalStack/AWS deployment failures
3. **Simple vs. moderate mode** — `any_stage_in_moderate_mode()` checks whether *any currently-failing* stage's cumulative error count (`stage_error_counts`) has reached `SIMPLE_MODE_THRESHOLD` (`config.py`, currently `0`, i.e. every failing stage escalates to moderate mode on its very first failing iteration in this checkout's configuration):
   - **Simple mode** → routes straight back to `engineer_simple` for direct self-correction from raw errors, skipping RAG entirely (cheapest repair path).
   - **Moderate mode** → routes to `retriever` (which always chains into `remediator` in the current graph — the commented-out `should_include_remediation_context` gate is currently short-circuited to always retrieve) for full hybrid-RAG-grounded remediation, then to `engineer` (Path C).

This staged design means trivial errors (a missing quote, an off-by-one YAML indent) are fixed by a fast, cheap, RAG-free loop, while errors that persist or involve unfamiliar resource schemas escalate to the expensive retrieval-grounded repair path — this is also what the repo's ablation studies (`benchmark_runs/*Ablation*`) are designed to measure (e.g. "No Planner", "No HyDE", "No Remediation").

## Dual-stream hybrid GraphRAG

Repair grounding is split across two knowledge stores that are queried differently, per IaC surface:

- **Semantic stream (ChromaDB)** — dense vector search over pre-indexed documentation chunks (`_semantic_search()` in `tools/cfn_hybrid_rag.py` / `tools/tf_hybrid_rag.py`), driven by HyDE queries generated by the Retriever agent. Chunks are filtered to `seed_resources` extracted from the template so unrelated resource types cannot leak in via coincidental similarity, then grouped and deduplicated per resource (`CHROMA_CONTEXT_MODE=compact`, the default, cuts token usage roughly in half versus `raw` mode).
- **Deterministic schema stream (Neo4j)** — a Cypher graph traversal (`_graph_schema_lookup()`) over the provider's resource/attribute/nested-block schema, keyed by exact resource type rather than similarity. For Terraform this walks `(:TFResource)-[:HAS_ATTRIBUTE]->(:TFAttribute)`, `(:TFResource)-[:HAS_BLOCK]->(:TFBlock)`, and `(:TFResource)-[:HAS_EXAMPLE]->(:TFExample)`; the CloudFormation graph mirrors this shape for CFN resource types. When specific failing resources are known, only their schemas are fetched rather than the full template's resource set.
- **Security stream** (`tools/security_hybrid_rag.py`) is a third, purely deterministic lookup: Trivy/Checkov always emit explicit rule IDs (e.g. `AVD-AWS-0086`), so `extract_trivy_check_ids()` regex-extracts these directly and looks them up in a dedicated Neo4j `SecurityCheck` subgraph — no embedding step is needed since there is no ambiguity to resolve.

All three streams are assembled into a single formatted context string consumed by the Remediator's prompt. The two RAG-ingestion pipelines that build these stores are under [`scripts/graphrag/`](scripts/graphrag/) (CFN + shared) and [`scripts/graphrag/terraform/`](scripts/graphrag/terraform/) (TF-specific), each with its own step-numbered scripts (`01_download_*` → `06_test_rag_queries.py`) and README; the security ingestion pipeline lives in `scripts/graphrag/security/`.

## Validation & deployment toolchain

[`tools/validators.py`](tools/validators.py) wraps every static-analysis tool behind a common `ValidationResult` shape (`stage`, `passed`, `errors`, `raw_output`, plus policy pass-rate stats for security stages):

| Stage | Tool | IaC surface |
|---|---|---|
| `yaml` | `yamllint` (via internal YAML parser) | CloudFormation |
| `cfn-lint` | `cfn-lint` | CloudFormation |
| `tflint` | `tflint` | Terraform |
| `terraform-validate` | `terraform validate` | Terraform |
| `checkov` | `checkov` | Both |
| `trivy` | `trivy` | Both |

[`tools/deploy_validator.py`](tools/deploy_validator.py) performs the live-deployment check: it builds a CFN or Terraform client/binary wrapper (`_build_cfn_client`, `_terraform_bin`), resets target state between runs (`_reset_localstack_state`/`_reset_aws_state`, including stray-VPC and stack cleanup to avoid quota exhaustion across repeated benchmark runs), submits the deployment, and reports per-resource pass/fail status (`FailedResource`, keyed by CFN `LogicalResourceId` or Terraform resource address) plus deployment logs. Deployment target is selected via `DeployConfig.target` (`none` / `localstack` / `aws`, `config.py`), defaulting to **LocalStack** for safe, repeatable, cost-free evaluation, with real AWS as an opt-in target (`aws_region`/`aws_profile` from environment).

## Tech stack

- **Language / runtime**: Python (repo uses a `.venv`; no upper Python version pin file is checked in — developed against a recent CPython 3.x).
- **Orchestration**: `langgraph` (`1.1.6`), `langchain` (`1.2.15`), `langchain-core`, `langchain-chroma`, `langchain-huggingface`, `langchain-ollama`.
- **LLM providers** (`agents/llm_client.py`, `config.py`): **OpenRouter** (default provider, any OpenAI-compatible model, default `x-ai/grok-4.1-fast`), **Anthropic Claude** (direct API), **OpenAI direct** (including o-series/codex reasoning models — `o1`, `o3`, `o4`, `codex-*` — auto-detected by `is_openai_reasoning_model()`, which switches to `max_completion_tokens` and omits `temperature` for these models). OpenRouter requests additionally support provider allow-listing and minimum-quantization filtering (`--openrouter-provider-only`, `--openrouter-min-quantization`).
- **Hybrid RAG stack**: `chromadb` (semantic vector store), `neo4j` (deterministic schema graph), `faiss-cpu` + `rank_bm25` (hybrid dense/sparse search over the offline CFN corpus in `data/cfn_rag_*`), `sentence_transformers` (embeddings).
- **IaC validation tools**: `cfn-lint`, `checkov`, `yamllint`, plus externally-invoked `tflint` and `terraform` binaries (not pip packages — must be installed separately, see Setup).
- **Deployment**: `boto3` (AWS/CloudFormation SDK), `terraform-local` (`tflocal`, for LocalStack-targeted Terraform applies).
- **Data / utilities**: `pandas`, `numpy`, `beautifulsoup4` (documentation scraping), `datasets`, `httpx`, `requests`, `python-dotenv`.

Exact pinned versions are in [`requirements.txt`](requirements.txt).

## Repository layout

```text
.
├── agents/                    # LangGraph node implementations (the 5 agents + llm_client + history helpers)
├── prompts/                   # System/user prompt templates, one module per agent
├── tools/                     # RAG execution, template annotation/AST parsing, validators, deploy validator
├── scripts/
│   ├── graphrag/               # Ingestion pipeline: CFN + shared knowledge-base build (01_* … 07_*, README.md)
│   │   ├── terraform/          # Terraform-specific ingestion pipeline (01_* … 06_*, README.md)
│   │   └── security/           # Trivy/Checkov security-graph ingestion pipeline
│   ├── build_cfn_graph.py / build_neo4j.py / build_cfn_rag_index.py   # standalone KB build helpers
│   ├── evaluate_iac_eval_benchmark.py   # Terraform deployability evaluation runner
│   └── aggregate_benchmark_run_data.py  # merges/aggregates benchmark_runs/ output into result CSVs
├── tracking/
│   └── recorder.py            # ResearchRecorder — logs every LLM call + policy/compliance metrics
├── data/                      # Benchmark prompt datasets + offline RAG corpora/indices (see below)
├── benchmark_runs/            # Output of past `benchmark.py` runs (per-run folders, incl. ablation studies)
├── runs/                      # Additional archived benchmark run outputs
├── main.py                    # Single-request CLI entry point
├── benchmark.py                # Batch CSV-driven evaluation entry point
├── graph.py                    # LangGraph StateGraph definition + routing logic
├── state.py                    # GraphState TypedDict + stage classification helpers
├── config.py                   # LLMConfig / DeployConfig / SIMPLE_MODE_THRESHOLD
├── docker-compose.yml           # Neo4j + ChromaDB services (persistent named volumes)
├── localstack-docker-compose.yml # LocalStack Pro service for live-deploy validation
└── AGENTS.md                    # Condensed contributor guide for agentic coding assistants
```

## Setup

1. **Python environment** — create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **External CLI tools** (not installed via pip) — `tflint` and `terraform` must be on `PATH` for Terraform validation/deployment.
3. **Knowledge-base services** — start Neo4j + ChromaDB:
   ```bash
   docker compose up -d
   ```
   Then run the ingestion pipelines under `scripts/graphrag/` (CFN + security) and `scripts/graphrag/terraform/` (Terraform) to populate both stores — see each subfolder's `README.md` for the exact step order. Pre-built artifacts (`cfn_knowledge_graph.json`, `tf_knowledge_graph.json`, `chroma_db/`, `data/cfn_rag_*`) may already be present in this checkout from a prior ingestion run.
4. **LocalStack** (default deploy target) — start it via:
   ```bash
   docker compose -f localstack-docker-compose.yml up -d
   ```
   Requires a `LOCALSTACK_AUTH_TOKEN` (LocalStack Pro is used, for CloudFormation/Terraform provider coverage).
5. **Environment variables** (`.env`) — at minimum one LLM provider key: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` if proxying, e.g. Azure OpenAI). For real AWS deployment, standard `AWS_PROFILE`/`AWS_DEFAULT_REGION` resolution applies.

## Usage

### Single request (`main.py`)

```bash
python main.py --request "A public S3 bucket with versioning and a CloudFront distribution in front of it" \
  --iac-type cloudformation \
  --provider openrouter \
  --deploy-target localstack \
  --max-iterations 30
```

Key flags: `--iac-type` (`cloudformation` | `terraform`), `--provider` (`openrouter` | `claude` | `openai`), `--model` (provider-specific model id), `--deploy-target` (`none` | `localstack` | `aws`), `--localstack-endpoint` (override, default `http://localhost:4566`), `--openrouter-provider-only` / `--openrouter-min-quantization` (OpenRouter routing filters), `--max-iterations` (default `30`).

### Batch benchmarking (`benchmark.py`)

```bash
python benchmark.py --iac-type cloudformation --dataset data/iac_eval_deployable.csv \
  --provider openrouter --deploy-target localstack --max-iterations 30
```

Runs the full pipeline over every row of a prompt CSV (`row_number`, `prompt`[, `ground_truth_path`]), defaulting to `data/iac_basic.csv` for CloudFormation or `data/tf_basic.csv` for Terraform when `--dataset` is omitted. Supports `--start-row`/`--max-rows`/`--rows` for partial/resumable runs (`--rows` takes a comma-separated list of `row_number` values from the CSV, not positional indexes). Results are written per-run under `benchmark_runs/<iac_type>_<timestamp>/` as a CSV with a fixed schema (`CSV_RESULT_FIELDS`): run id, pass/fail status, iterations used, full LLM token accounting (input/output/prompt/completion/total), scenario policy pass rate, filtered/unfiltered compliance rate, duration, and error message/traceback on failure. `scripts/aggregate_benchmark_run_data.py` merges multiple run folders' results into a single consolidated CSV/JSONL for analysis.

## Benchmark data

`data/` holds both the evaluation prompt sets and the offline RAG corpora:

- **Prompt datasets**: `iac_eval_deployable.csv` (90 rows — deployability-focused evaluation prompts, `row_number,prompt`), `iac_basic.csv` (259 rows — `row_number,ground_truth_path,prompt`).
- **RAG corpora / indices**: `cfn_spec.json`, `cfn_rag_corpus.jsonl`, `cfn_rag_faiss.index`, `cfn_rag_bm25.pkl`, `cfn_graph.pkl` (CloudFormation); `avd_scraped.json`, `avd_remediation_map.csv`, `security_checks.json`, `checkov_cfn_policy_map.csv`, `trivy_cfn_policy_map.csv`, `trivy_enriched.csv` (security); `tf_evaluation_results.csv` (Terraform deployability results).
- Top-level `cfn_knowledge_graph.json` / `tf_knowledge_graph.json` / `tf_registry_docs.json` / `tf_schema_raw.json` and `scraped_html/` (1,583 scraped AWS resource-reference pages) / `scraped_avd/` are ingestion-pipeline intermediates feeding the Neo4j + ChromaDB stores.

Past run outputs (`benchmark_runs/`, `runs/`) include labeled evaluation sweeps across different LLM backends (Grok, o3-mini, DeepSeek) and RAG configurations, plus dedicated ablation runs (no-Planner, no-HyDE, no-Remediation) used to isolate each component's contribution to the overall pass rate.

## Configuration reference

`config.py` centralizes all tunables:

- `LLMConfig` — provider (`openrouter` default), model, temperature (`0.0` default), `max_tokens` (`8192`), reasoning toggle, plus per-provider API key/base-URL fields (all resolved from environment variables by default).
- `DeployConfig` — deploy target (`none`/`localstack`/`aws`), LocalStack endpoint (`http://localhost:4566` default) and reset-wait, AWS region/profile, stack creation/deletion timeouts (30 min / 5 min).
- `SIMPLE_MODE_THRESHOLD` — the per-stage error-count threshold at which the router escalates a failing stage from simple self-correction to full hybrid-RAG remediation (see [Orchestration](#orchestration-the-langgraph-state-machine)).
