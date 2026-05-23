# GraphRAG Setup: CFN Knowledge Graph

This guide sets up the full CFN GraphRAG pipeline from scratch — spinning up
Neo4j and ChromaDB with **persistent Docker named volumes**, running all five
build scripts in order, and verifying retrieval works.

Data survives container restarts.  Steps 3.1–3.5 only need to be re-run
when the CFN spec changes, not on every `docker compose up`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  GraphRAG Pipeline                  │
│                                                     │
│  01_download_cfn_spec.py  → cfn_resource_spec.json  │
│  02_scrape_cfn_docs.py    → scraped_html/           │
│  03_parse_and_merge.py    → cfn_knowledge_graph.json│
│           │                        │                │
│           ▼                        ▼                │
│  04_import_cfn_to_neo4j.py   05_build_chromadb.py   │
│           │                        │                │
│           ▼                        ▼                │
│     Neo4j :7687              ChromaDB :8000         │
│   (bolt + browser)         (HTTP REST API)          │
│   volume: neo4j_data        volume: chromadb_data   │
│                                                     │
│           └──────── execute_g_retrieval.py ─────────│
└─────────────────────────────────────────────────────┘
```

**Two-store retrieval:**
- **ChromaDB** holds one embedding chunk per CFN property (semantic search).
- **Neo4j** holds the full structural subgraph (graph traversal).
- The `property_id` field (e.g. `AWS::S3::Bucket.BucketEncryption`) is the
  bridge key that links a ChromaDB result back to a Neo4j node.

---

## Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Docker | 24.x | `docker --version` |
| Docker Compose | v2 plugin | `docker compose version` |
| Python | 3.11+ | `python --version` |
| pip | 23+ | `pip --version` |

---

## Step 0 — Install Python Dependencies

From the repo root:

```bash
pip install \
  neo4j \
  chromadb \
  langchain-chroma \
  langchain-huggingface \
  langchain-core \
  langchain-ollama \
  sentence-transformers \
  numpy \
  requests \
  beautifulsoup4
```

Or, if the repo has a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Start Docker Containers

The `docker-compose.yml` at the **repo root** defines both services with
persistent named volumes.  Volumes are managed by Docker (not host
bind-mounts) so they work identically on macOS, Linux, and Windows.

```bash
docker compose up -d
```

Wait for both to pass their health checks (~30 seconds on first boot,
~10 seconds on subsequent starts when volumes already exist):

```bash
docker compose ps
# Both STATUS should show "healthy"
```

### Volume reference

| Volume | Container path | Stores |
|---|---|---|
| `iac-god_neo4j_data` | `/data` | Graph store: nodes, relationships, indexes |
| `iac-god_neo4j_logs` | `/logs` | Server logs |
| `iac-god_chromadb_data` | `/chroma/chroma` | Vector index + collection metadata |

Volumes persist across `docker compose down`.  Use `docker compose down -v`
only when you want to **wipe all data** and start from scratch.

---

## Step 2 — Set Environment Variables

The scripts read connection details from environment variables with
fallback defaults that match `docker-compose.yml`. You only need to export
these if you change the defaults:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"

# Embedding provider (default: huggingface)
export EMBEDDING_PROVIDER=ollama        # or: huggingface
export OLLAMA_BASE_URL=http://localhost:11434

# Optional: tune the cosine distance threshold (default: 0.40)
export CHROMA_DISTANCE_THRESHOLD=0.40
```

> **Embedding model parity:** the `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL`
> used when running `05_build_chromadb.py` must exactly match what the RAG
> tools use at query time.  Mismatches silently return wrong results.  See
> `tools/embedding_provider.py` for full details.

---

## Step 3 — Run the Build Scripts

All scripts must be run **from inside `scripts/graphrag/`** because they
read and write relative paths (`cfn_resource_spec.json`,
`cfn_knowledge_graph.json`, `scraped_html/`).

```bash
cd scripts/graphrag
```

### 3.1 Download CFN Specification

```bash
python 01_download_cfn_spec.py
```

**Output:** `cfn_resource_spec.json`

Expected log:
```
Downloading CloudFormation Resource Specification...
Saved to cfn_resource_spec.json
Total Resources Found: 1105
Total Property Types (Nested Blocks) Found: 3030
```

---

### 3.2 Scrape CFN HTML Documentation

```bash
python 02_scrape_cfn_docs.py
```

**Output:** `scraped_html/` directory (~1100 HTML files)

> This step makes HTTP requests to `docs.aws.amazon.com`. It takes
> **10–20 minutes** depending on your connection.

---

### 3.3 Parse and Merge into Knowledge Graph JSON

```bash
python 03_parse_and_merge.py
```

**Output:** `cfn_knowledge_graph.json`

---

### 3.4 Import Knowledge Graph into Neo4j

```bash
python 04_import_cfn_to_neo4j.py
```

Expected log:
```
Creating indexes...
Clearing existing database...
Importing CloudFormation Knowledge Graph...
Import complete!
```

This creates four node labels and three relationship types:

```
(Resource)-[:HAS_PROPERTY]    → (Property)
(Resource)-[:HAS_NESTED_TYPE] → (NestedType)
(NestedType)-[:REFERENCES_TYPE]→ (NestedType)
(Resource)-[:HAS_EXAMPLE]     → (Example)
```

**Verify in Neo4j Browser** (`http://localhost:7474`, login `neo4j/password`):

```cypher
// Count all nodes
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC;

// Spot-check one resource
MATCH (r:Resource {name: 'AWS::S3::Bucket'})-[:HAS_PROPERTY]->(p)
RETURN r.name, p.name, p.type LIMIT 10;
```

---

### 3.5 Build ChromaDB Vector Index

```bash
# HuggingFace (default)
python 05_build_chromadb.py

# Ollama (mxbai-embed-large)
EMBEDDING_PROVIDER=ollama python 05_build_chromadb.py
```

Expected log:
```
[Build] Embedding provider: Ollama  model: mxbai-embed-large  url: http://localhost:11434
Chunking CloudFormation data...
Created 28000+ document chunks. Ingesting into ChromaDB...
Vector database successfully built!
  Provider   : ollama
  Model      : mxbai-embed-large
  Chunks     : 28312
  Collection : cfn_schema_properties @ localhost:8000
  Distance   : cosine (hnsw:space=cosine)
  Normalised : True
```

> **First run:** HuggingFace downloads the embedding model (~420 MB, 5–15 min).
> Ollama models must be pulled first: `ollama pull mxbai-embed-large`.
> Subsequent runs skip the download and only re-embed (3–5 min).

**Verify the collection:**

```bash
curl -s http://localhost:8000/api/v1/collections | python -m json.tool
# Should list cfn_schema_properties with document count > 0
```

---

## Step 4 — Test G-Retrieval

```bash
# Default test query
python 06_execute_g_retrieval.py

# Custom query
python 06_execute_g_retrieval.py "Create an RDS MySQL instance with Multi-AZ"

# With Ollama
EMBEDDING_PROVIDER=ollama python 06_execute_g_retrieval.py "S3 bucket with KMS encryption"
```

---

## Step 5 — Debug ChromaDB

```bash
python debug_chroma.py
```

This lists all collections, document counts, and runs a sample similarity
query so you can inspect raw cosine distance scores.

---

## Volume Lifecycle

| Scenario                             | Command                                                                                      |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| Normal restart, data preserved       | docker compose down && docker compose up -d                                                  |
| Wipe everything, full rebuild        | docker compose down -v                                                                       |
| Rebuild only ChromaDB (model switch) | docker volume rm iac-god_chromadb_data then re-run 05_build_chromadb.py                      |
| Rebuild only Neo4j (CFN spec update) | docker volume rm iac-god_neo4j_data iac-god_neo4j_logs then re-run 04_import_cfn_to_neo4j.py |

```bash
# Normal restart (data preserved)
docker compose down
docker compose up -d

# Inspect volume sizes on disk
docker system df -v

# Wipe ALL data and start from scratch
docker compose down -v
docker compose up -d
cd scripts/graphrag
python 04_import_cfn_to_neo4j.py
EMBEDDING_PROVIDER=ollama python 05_build_chromadb.py

# Rebuild only ChromaDB (e.g. after switching embedding model)
docker volume rm iac-god_chromadb_data
docker compose up -d chromadb
EMBEDDING_PROVIDER=ollama python 05_build_chromadb.py

# Rebuild only Neo4j (e.g. after CFN spec update)
docker volume rm iac-god_neo4j_data iac-god_neo4j_logs
docker compose up -d neo4j
python 04_import_cfn_to_neo4j.py
```

> **Warning:** `docker compose down -v` and `docker volume rm` are
> irreversible.  The data can be fully rebuilt from the JSON artefacts
> in `scripts/graphrag/`, but the embedding step takes 3–15 minutes.

---

## Environment Variables Reference

| Variable | Default | Used by |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | scripts 04, `neo4j_client.py`, `cfn_graph_neo4j_rag.py` |
| `NEO4J_USER` | `neo4j` | same as above |
| `NEO4J_PASSWORD` | `password` | same as above |
| `CHROMA_HOST` | `localhost` | `cfn_hybrid_rag.py`, `security_hybrid_rag.py` |
| `CHROMA_PORT` | `8000` | same as above |
| `EMBEDDING_PROVIDER` | `huggingface` | `embedding_provider.py`, build scripts |
| `EMBEDDING_MODEL` | provider default | override model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `embedding_provider.py` |
| `CHROMA_DISTANCE_THRESHOLD` | `0.40` | `cfn_hybrid_rag.py`, `security_hybrid_rag.py` |
| `SECURITY_CHAR_BUDGET` | `12000` | `security_hybrid_rag.py` |

---

## Troubleshooting

**`neo4j.exceptions.ServiceUnavailable: Failed to establish connection`**

The container is not yet ready. Wait for `docker compose ps` to show
`healthy`, then retry. Neo4j takes ~15–30 seconds to start.

**`chromadb.errors.InvalidCollectionException` or empty results from ChromaDB**

If the container was previously run without `IS_PERSISTENT=TRUE`, the old
in-memory collection is gone. Rebuild:

```bash
EMBEDDING_PROVIDER=ollama python scripts/graphrag/05_build_chromadb.py
```

**`ValueError: Collection cfn_schema_properties already exists with different metadata`**

The existing collection was created with a different `hnsw:space`. The build
script drops and recreates it automatically, but if you see this at query
time it means the collection was built without the cosine fix. Rebuild:

```bash
docker volume rm iac-god_chromadb_data
docker compose up -d chromadb
EMBEDDING_PROVIDER=ollama python scripts/graphrag/05_build_chromadb.py
```

**`OSError: [Errno 28] No space left on device`**

Ensure Docker Desktop has at least **4 GB RAM** and **10 GB disk** allocated
(Docker Desktop → Settings → Resources). Neo4j data + ChromaDB vectors +
embedding model cache total ~3–4 GB.

**`AuthError: {code: Neo.ClientError.Security.Unauthorized}`**

The `NEO4J_AUTH` env var in `docker-compose.yml` sets `username/password`.
If you changed it, export `NEO4J_PASSWORD` to match.

**Slow scraping in step 3.2**

Add `time.sleep(0.5)` inside the scrape loop in `02_scrape_cfn_docs.py`
if you see 429 errors from the AWS docs CDN. The scrape only needs to run
once — commit `scraped_html/` to avoid repeating it.
