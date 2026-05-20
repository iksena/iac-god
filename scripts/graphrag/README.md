# GraphRAG Setup: CFN Knowledge Graph

This guide sets up the full CFN GraphRAG pipeline from scratch — spinning up
Neo4j and ChromaDB as ephemeral Docker containers (no host volume mounts),
running all five build scripts in order, and verifying retrieval works.

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

> **Note:** The build scripts run on your host machine and connect to the
> containers over `localhost`. No data is persisted to host volumes — all
> graph and vector data lives inside the containers. Restart the containers
> after rebuilding to get a clean state.

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
  sentence-transformers \
  requests \
  beautifulsoup4
```

Or, if the repo has a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Start Docker Containers

Create a `docker-compose.yml` in the **repo root** (not inside `scripts/graphrag/`):

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5.20-community
    container_name: iac-god-neo4j
    ports:
      - "7474:7474"   # Neo4j Browser (HTTP)
      - "7687:7687"   # Bolt protocol (used by scripts)
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'        # optional but useful for debugging
      NEO4J_dbms_memory_heap_max__size: 1G
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 10

  chromadb:
    image: chromadb/chroma:latest
    container_name: iac-god-chromadb
    ports:
      - "8000:8000"
    environment:
      IS_PERSISTENT: "FALSE"           # in-memory only, no volume mount
      ANONYMIZED_TELEMETRY: "FALSE"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 10
```

Start both containers:

```bash
docker compose up -d
```

Wait for both to pass their health checks (~20 seconds):

```bash
docker compose ps
# Both STATUS should show "healthy"
```

> **ChromaDB data is in-memory only.** If the container restarts you must
> re-run `05_build_chromadb.py` to repopulate the collection. This is
> intentional — the collection is rebuilt from the deterministic knowledge
> graph JSON so no volume is needed.

---

## Step 2 — Set Environment Variables

The scripts read connection details from environment variables with
fallback defaults that match the `docker-compose.yml` above. You only
need to export these if you change the defaults:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
# ChromaDB host/port are hardcoded to localhost:8000 in the scripts
```

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
> **10–20 minutes** depending on your connection. AWS rate-limits are
> gentle — the script does not require a delay but you may add one if you
> see 429 errors.

---

### 3.3 Parse and Merge into Knowledge Graph JSON

```bash
python 03_parse_and_merge.py
```

**Output:** `cfn_knowledge_graph.json`

This merges the spec JSON (structural data) with the scraped HTML
(descriptions + YAML examples) into a single flat dictionary keyed by
resource name (e.g. `AWS::S3::Bucket`).

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
(Resource)-[:HAS_PROPERTY]   →(Property)
(Resource)-[:HAS_NESTED_TYPE]→(NestedType)
(NestedType)-[:REFERENCES_TYPE]→(NestedType)
(Resource)-[:HAS_EXAMPLE]    →(Example)
```

And three indexes:
- `resource_name` on `Resource.name`
- `property_id` on `Property.id`
- `nested_type_id` on `NestedType.id`

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
python 05_build_chromadb.py
```

Expected log:
```
Loading embedding model: sentence-transformers/all-mpnet-base-v2...
Chunking CloudFormation data...
Created 28000+ document chunks. Ingesting into ChromaDB Docker container...
Vector database successfully built inside Docker!
```

> **First run downloads the embedding model** (~420 MB) from HuggingFace
> into the local `sentence-transformers` cache. Subsequent runs reuse the
> cache. This step takes **5–15 minutes** on first run (mostly embedding
> inference).

The script connects to `chromadb` at `localhost:8000` and writes all chunks
into the `cfn_schema_properties` collection.

**Verify the collection exists:**

```bash
curl -s http://localhost:8000/api/v1/collections | python -m json.tool
# Should list cfn_schema_properties with a document count > 0
```

---

## Step 4 — Test G-Retrieval

```bash
python execute_g_retrieval.py
```

This runs a sample query end-to-end:
1. Semantic search in ChromaDB (`k=5`).
2. Graph traversal in Neo4j for the matched resource names.
3. Prints the assembled context block that would be injected into the LLM prompt.

For stage-by-stage test coverage:

```bash
python 07_test_g_retrieval_by_stage.py
```

Results are saved to `g_retrieval_test_results/`.

---

## Step 5 — Debug ChromaDB

If retrieval returns no results, run the debug script to inspect the
collection directly:

```bash
python debug_chroma.py
```

This lists all collections, their document counts, and runs a sample
similarity query so you can inspect raw distance scores.

---

## Teardown

Stop and remove both containers (all in-memory data is lost):

```bash
docker compose down
```

To rebuild from scratch after a teardown, restart containers and re-run
steps 3.4 and 3.5 only (the JSON artefacts from steps 3.1–3.3 are cached
locally and do not need to be regenerated unless the CFN spec changes):

```bash
docker compose up -d
cd scripts/graphrag
python 04_import_cfn_to_neo4j.py
python 05_build_chromadb.py
```

---

## Environment Variables Reference

| Variable | Default | Used by |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | scripts 04, `neo4j_client.py`, `cfn_graph_neo4j_rag.py` |
| `NEO4J_USER` | `neo4j` | same as above |
| `NEO4J_PASSWORD` | `password` | same as above |
| `CHROMA_HOST` | `localhost` | `cfn_hybrid_rag.py`, `security_hybrid_rag.py` |
| `CHROMA_PORT` | `8000` | same as above |
| `SECURITY_DISTANCE_THRESHOLD` | `0.55` | `security_hybrid_rag.py` |
| `SECURITY_CHAR_BUDGET` | `12000` | `security_hybrid_rag.py` |

All variables have working defaults that match the `docker-compose.yml` in
this guide. Export overrides only when deploying outside of local Docker.

---

## Troubleshooting

**`neo4j.exceptions.ServiceUnavailable: Failed to establish connection`**

The container is not yet ready. Wait for `docker compose ps` to show
`healthy`, then retry. Neo4j takes ~15 seconds to start.

**`chromadb.errors.InvalidCollectionException` or empty results from ChromaDB**

The `cfn_schema_properties` collection is lost on container restart because
`IS_PERSISTENT=FALSE`. Re-run `05_build_chromadb.py` after every container
restart.

**`OSError: [Errno 28] No space left on device` during ChromaDB ingestion**

The embedding model cache and ~28k vectors require ~2 GB of working memory.
Ensure Docker Desktop has at least **4 GB RAM** allocated
(Docker Desktop → Settings → Resources → Memory).

**`AuthError: {code: Neo.ClientError.Security.Unauthorized}`**

The `NEO4J_AUTH` env var in `docker-compose.yml` sets `username/password`.
If you changed the password, export `NEO4J_PASSWORD` to match before
running the scripts.

**Slow scraping in step 3.2**

The AWS docs CDN occasionally throttles. If you see many timeout errors,
add a `time.sleep(0.5)` inside the scrape loop in `02_scrape_cfn_docs.py`.
Alternatively, the scrape only needs to be done once — commit the
`scraped_html/` output and re-use it across environments.
