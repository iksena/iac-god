# IaC-GOD: Agentic Development Guide

Welcome! You are assisting in the development of **IaC-GOD** (Infrastructure-as-Code Generative Orchestration & Debugging). 

This repository contains an advanced, multi-agent generative AI system designed to autonomously generate, statically validate, and iteratively repair Infrastructure as Code (IaC) templates. It natively supports both **AWS CloudFormation** and **HashiCorp Terraform**.

## 🧠 System Architecture

The core of the system is a cyclic, stateful Directed Acyclic Graph (DAG) built using **LangGraph**. The workflow iterates until the infrastructure passes all static and live validations, or hits a maximum iteration cap.

### The 5 Core Agents (`/agents`)
1. **Planner (`planner.py`)**: Analyzes the natural language user request and generates structured, numbered "Grounded Objectives".
2. **Engineer (`engineer.py`)**: Translates the Planner's objectives into raw IaC code (CFN YAML or TF HCL). In repair loops, it fixes code based on provided RAG context or Remediator suggestions.
3. **Validator (`validator.py`)**: An execution environment that runs a rigid pipeline of external tools: syntax linters (`cfn-lint`, `tflint`), security scanners (`Trivy`, `Checkov`), and live emulators (`LocalStack`).
4. **Retriever (`retriever.py`)**: Triggers when validation fails. It uses LLMs to generate Hypothetical Document Embeddings (HyDE queries), extracts failing resource IDs via Regex, and queries the offline Knowledge Bases.
5. **Remediator (`remediator.py`)**: Digests the RAG context and errors, writes a Root Cause Analysis (RCA), and formulates a precise fix plan for the Engineer.

### The RAG Engine (`/tools` & `/scripts/graphrag`)
We use a sophisticated **Hybrid GraphRAG** pipeline to prevent hallucination:
* **Dense RAG (ChromaDB)**: Embeds human-readable AWS/Terraform documentation. Queried via HyDE.
* **Graph RAG (Neo4j)**: Stores deterministic provider schemas (Required Attributes, Nested Blocks). Queried deterministically using Regex-extracted resource IDs.
* **Security RAG**: Deterministic lookup mapping Trivy vulnerability IDs (e.g., `AVD-AWS-0086`) directly to compliant code snippets.

## 📂 Repository Structure

```text
.
├── agents/                 # LangGraph node definitions (the brains)
├── prompts/                # System and user prompt templates for all agents
├── tools/                  # Auxiliary logic (RAG execution, AST code annotation, validation parsers)
├── scripts/                # Utility scripts
│   ├── graphrag/           # Ingestion scripts: builds Neo4j/ChromaDB for CFN, TF, and Security
│   └── evaluate_*.py       # Benchmark evaluation runners
├── tracking/               # Telemetry and logging (ResearchRecorder)
├── data/                   # Datasets, intermediate scraped data, and benchmarks
├── main.py                 # Single-shot execution entry point
├── benchmark.py            # Bulk evaluation entry point (runs CSV datasets)
├── graph.py                # LangGraph DAG definition and routing logic
├── state.py                # TypedDict defining the global GraphState
└── config.py               # Global thresholds and deployment configurations

```

## 🛠️ Development Guidelines & Rules

When writing or modifying code in this repository, you must adhere to the following principles:

### 1. State Management is Sacred

* All agent inputs and outputs flow through `state.py` (`GraphState`).
* If you add a new capability or return value to an agent, you **must** update the `TypedDict` in `state.py`. Handle optional fields carefully using `NotRequired`.
* Do not mutate the state directly in place if returning a list; use the `append_and_cap` helper or standard list concatenation `state["log"] + [new_item]` to ensure LangGraph state reducers work properly.

### 2. Dual IaC Support

* This system supports both CloudFormation and Terraform.
* Whenever you modify an agent, a prompt, or a RAG tool, check the `state.get("iac_type")`. You must ensure your logic handles both `cloudformation` and `terraform` paths gracefully.
* *Example:* CFN uses `cfn-lint` and YAML parsers. TF uses `tflint`, `terraform validate`, and HCL regex extractors.

### 3. Modularity and Ablation

* We frequently run **Ablation Studies** (e.g., removing the Planner, bypassing the Remediator, turning off HyDE).
* Keep agent logic highly decoupled. An agent should rely entirely on what is passed to it in the `GraphState` and should not make assumptions about previous agents running (e.g., always use `.get("key", default)` safely).

### 4. RAG Precision over Volume

* We optimize for token efficiency. Do not dump raw, unfiltered documentation into the LLM context.
* If modifying RAG tools (`tools/*_hybrid_rag.py`), maintain the strict metadata filtering (e.g., `seed_resources`) to ensure we only retrieve schema docs for resources that actually exist in the user's template.

### 5. LLM Client Abstraction

* All LLM calls must go through `agents/llm_client.py` (`_call_llm_with_history` or `_build_client`).
* Always record LLM usage by returning the `llm_record` object and appending it to `state["llm_call_log"]` using the `recorder` tool.

### 6. LocalStack & Emulation

* The Validator agent supports local deployment via LocalStack or directly to AWS.

**Your Goal:** Help the user refine Agentic loops, improve GraphRAG retrieval precision, fix LangGraph routing bugs, and expand benchmark coverage without breaking the delicate multi-agent state flow.