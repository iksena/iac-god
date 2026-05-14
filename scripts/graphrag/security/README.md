# Security Remediation GraphRAG Pipeline

This directory contains the security context pipeline that mirrors and extends the CFN schema GraphRAG pipeline at `scripts/graphrag/`.

## Overview

The goal is to build a **security remediation knowledge graph** from [Aqua Security AVD](https://avd.aquasec.com/misconfig/aws/) Trivy check data, enabling the IaC generation LLM to produce CloudFormation templates that are both structurally correct (from the CFN graph) and security-compliant (from this graph).

## Architecture

```
trivy_enriched.csv  ──► 01_load_trivy_csv.py ──► data/security_checks.json
                                                         │
                         02_scrape_avd_docs.py ◄─────────┤ (enriches in-place)
                                                         │
                         03_build_security_chromadb.py ◄─┘
                              │
                              ▼
                    data/chroma_security_db/     ← ChromaDB: 'security_checks' collection
                    (Stage 2 – coming next)
                              │
                    04_import_security_to_neo4j.py
                              │
                              ▼
                    Neo4j: SecurityCheck, Impact, Remediation, GoodExample,
                           RegoPolicy, AwsService nodes
                           + APPLIES_TO_RESOURCE edges into CFN graph
```

## Stage 1 – Data Loading & ChromaDB (this PR)

### Scripts

| Script | Purpose |
|---|---|
| `01_load_trivy_csv.py` | Parse `data/trivy_enriched.csv` → `data/security_checks.json` |
| `02_scrape_avd_docs.py` | Scrape AVD HTML pages → `data/scraped_avd/` + enrich JSON |
| `03_build_security_chromadb.py` | Embed checks → `data/chroma_security_db/` ChromaDB |

### Running Stage 1

```bash
# 1. Parse CSV
python scripts/graphrag/security/01_load_trivy_csv.py

# 2. Scrape AVD pages (optional but recommended for richer embeddings)
#    Respects a 0.5s delay between requests. HTML is cached in data/scraped_avd/
python scripts/graphrag/security/02_scrape_avd_docs.py

# 3. Build ChromaDB
python scripts/graphrag/security/03_build_security_chromadb.py
```

### Dependencies

```bash
pip install requests beautifulsoup4 langchain-core langchain-huggingface langchain-chroma chromadb sentence-transformers
```

These are additive to the existing CFN pipeline dependencies.

## Chunking Strategy vs CFN Pipeline

| | CFN Pipeline | Security Pipeline |
|---|---|---|
| Chunk unit | Per **property** (fine-grained) | Per **check** (coarse – remediation is holistic) |
| Collection name | `cfn_schema` | `security_checks` |
| Persist dir | `data/chroma_db/` | `data/chroma_security_db/` |
| Bridge key (→ Neo4j) | `property_id` | `check_id` |
| Embedding model | `all-mpnet-base-v2` | `all-mpnet-base-v2` (same) |

## Stage 2 (next PR)

- `04_import_security_to_neo4j.py` – import SecurityCheck, Impact, Remediation, GoodExample, RegoPolicy, AwsService nodes
- `05_execute_security_retrieval.py` – G-Retrieval using `security_checks` ChromaDB + Neo4j traversal
- `06_combined_retrieval.py` – dual-graph retrieval merging CFN schema context + security context into one prompt
