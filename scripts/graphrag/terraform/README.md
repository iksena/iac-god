# Terraform GraphRAG Ingestion Pipeline

This folder contains the five-stage ingestion pipeline that builds the Terraform
knowledge graph used by the IaCGOD Terraform generation and remediation agents.

It mirrors the structure of the parent `scripts/graphrag/` CFN pipeline but
targets Terraform's `attributes` / `block_types` schema model instead of
CloudFormation's flat `Properties` dict.

## Architecture

```
flowchart TD
    A[01_download_tf_schema.py] -->|GitHub Releases artifact| B(tf_schema_raw.json)
    C[02_fetch_tf_registry_docs.py] -->|Registry REST API v2| D(tf_registry_docs.json)
    B --> E[03_parse_and_merge_tf.py]
    D --> E
    E -->|Merges into| F(tf_knowledge_graph.json)
    F --> G[04_import_tf_to_neo4j.py]
    F --> H[05_build_tf_chromadb.py]
    G -->|Label-scoped MERGE — CFN graph untouched| I[(Shared Neo4j)]
    H -->|iac_type=terraform metadata| J[(Shared ChromaDB: tf_schema_properties)]
```

## Database Strategy

| Store    | CFN (existing)            | Terraform (new)                  |
|----------|---------------------------|----------------------------------|
| Neo4j    | `:Resource`, `:Property`  | `:TFResource`, `:TFAttribute`, `:TFBlock` |
| ChromaDB | `cfn_schema_properties`   | `tf_schema_properties`           |

Both use the **same Neo4j instance and the same ChromaDB instance**.
Isolation is achieved via node labels (Neo4j) and collection names (ChromaDB).
No new database servers are required.

### Security cross-link

The existing `SecurityCheck` nodes in Neo4j can be extended with a
`tf_resource` property and a `[:APPLIES_TO_TF_RESOURCE]` relationship
during the security ingestion step (`scripts/graphrag/security/04_import_security_to_neo4j.py`).
The same `AVD-AWS-XXXX` rule that targets `AWS::EC2::Instance` in CFN
also targets `aws_instance` in Terraform — no separate mapping CSV needed.

## Scripts

### `01_download_tf_schema.py`

Downloads the pre-built provider schema JSON from the HashiCorp GitHub
release artifacts (no Terraform binary or `terraform init` required).
The artifact is identical to `terraform providers schema -json` output.

```bash
python 01_download_tf_schema.py
# Output: tf_schema_raw.json
```

**Source:** `https://github.com/hashicorp/terraform-provider-aws/releases`

To change the provider version, update `PROVIDER_VERSION` in the script.

### `02_fetch_tf_registry_docs.py`

Paginates the Terraform Registry REST API v2 to fetch human-readable
descriptions and HCL examples for every AWS resource type.

```bash
python 02_fetch_tf_registry_docs.py
# Output: tf_registry_docs.json
```

**Source:** `https://registry.terraform.io/v2/provider-docs`

### `03_parse_and_merge_tf.py`

Merges the raw schema (structural constraints) with the Registry docs
(descriptions and HCL examples) into a single knowledge graph artefact.

```bash
python 03_parse_and_merge_tf.py
# Input:  tf_schema_raw.json + tf_registry_docs.json
# Output: tf_knowledge_graph.json
```

### `04_import_tf_to_neo4j.py`

Ingests `tf_knowledge_graph.json` into Neo4j. Uses label-scoped deletion
so only Terraform nodes (`TFResource`, `TFAttribute`, `TFBlock`, `TFExample`)
are cleared on re-run — the CloudFormation graph is never touched.

```bash
python 04_import_tf_to_neo4j.py
# Input:  tf_knowledge_graph.json
# Env:    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
```

### `05_build_tf_chromadb.py`

Embeds all resource, attribute, and block descriptions into the
`tf_schema_properties` ChromaDB collection.

```bash
python 05_build_tf_chromadb.py
# Input:  tf_knowledge_graph.json
# Env:    CHROMADB_PATH (default: ./chroma_db)
#         EMBED_MODEL   (default: all-MiniLM-L6-v2)
```

## Running the Full Pipeline

```bash
cd scripts/graphrag/terraform

# Step 1 — download schema
python 01_download_tf_schema.py

# Step 2 — fetch registry docs (takes a few minutes due to pagination)
python 02_fetch_tf_registry_docs.py

# Step 3 — merge
python 03_parse_and_merge_tf.py

# Step 4 — Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
python 04_import_tf_to_neo4j.py

# Step 5 — ChromaDB
export CHROMADB_PATH=../../chroma_db   # same dir as CFN ChromaDB
python 05_build_tf_chromadb.py
```

## Intermediate Files

| File                      | Producer | Consumer(s)        |
|---------------------------|----------|--------------------|
| `tf_schema_raw.json`      | 01       | 03                 |
| `tf_registry_docs.json`   | 02       | 03                 |
| `tf_knowledge_graph.json` | 03       | 04, 05             |

Intermediate JSON files are not committed to the repository (add to `.gitignore`).

## Dependencies

```
httpx>=0.27
neo4j>=5.0
chromadb>=0.5
sentence-transformers>=3.0
beautifulsoup4  # already required by CFN pipeline
```
