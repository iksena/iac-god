"""01_download_tf_schema.py

Download the official AWS provider source zip from the GitHub release page
and extract the Terraform schema in the same format expected by downstream
scripts (02–05).

Background
----------
Starting with v6.x, HashiCorp no longer ships a pre-built
`terraform-provider-aws_<version>_SCHEMA.json` artifact alongside the
binary releases.  The authoritative schema lives inside the source zip as
Go `schema.Schema` structs embedded in each resource's `*.go` file.

Because parsing raw Go source is fragile, this script uses a two-step
approach that works without requiring Go, Terraform, or tfschema:

  1. Download the source zip from:
       https://github.com/hashicorp/terraform-provider-aws/archive/refs/tags/v<VERSION>.zip

  2. Walk every `internal/service/**/resource_*.go` (and `data_source_*.go`)
     file and parse HCL-style schema declarations using a targeted regex
     that is robust to the consistent formatting used in the provider codebase.

  3. Produce tf_schema_raw.json in the same nested structure the downstream
     scripts already consume:
       {
         "provider_schemas": {
           "registry.terraform.io/hashicorp/aws": {
             "resource_schemas": {
               "aws_s3_bucket": {
                 "block": {
                   "attributes": { "bucket": {"type": "string", "optional": true, ...} },
                   "block_types": { ... }
                 }
               }
             }
           }
         }
       }

The regex-based extractor has intentional limits:
  * It captures attribute names and their Required/Optional/Computed flags.
  * It captures nested block names and their nesting mode (List/Set/Single).
  * It does NOT parse type expressions beyond simple string|bool|int|number.
  * Complex TypeList/TypeSet element schemas are captured as a nested block.

This gives downstream scripts enough information to:
  - Build the Neo4j knowledge graph (04)
  - Embed schema chunks in ChromaDB (05)
  - Answer TFLint / deploy error queries (06)

Output
------
tf_schema_raw.json
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import PurePosixPath
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROVIDER_VERSION = "6.47.0"

_ZIP_URL = (
    "https://github.com/hashicorp/terraform-provider-aws"
    f"/archive/refs/tags/v{PROVIDER_VERSION}.zip"
)

OUTPUT_FILE = "tf_schema_raw.json"

_PROVIDER_KEY = "registry.terraform.io/hashicorp/aws"

# ---------------------------------------------------------------------------
# Step 1: Download the source zip
# ---------------------------------------------------------------------------

def download_zip(url: str) -> bytes:
    print(f"[01] Downloading provider source zip v{PROVIDER_VERSION} ...")
    print(f"     URL: {url}")
    print(f"     (this is ~50-80 MB; please wait)")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()
    size_mb = len(response.content) / 1_048_576
    print(f"[01] Downloaded {size_mb:.1f} MB.")
    return response.content

# ---------------------------------------------------------------------------
# Step 2: Extract resource name -> Go source mapping from the zip
# ---------------------------------------------------------------------------

# The zip root is  terraform-provider-aws-<version>/
# Resource Go files live at:
#   internal/service/<service>/resource_<name>.go
#   internal/service/<service>/data_source_<name>.go
# The Terraform resource name is declared in the ResourceType() or
# ResourceName() func, or inferred from the file path as aws_<service>_<name>.

# Regex to find resource name declarations in Go source:
#   return "aws_s3_bucket"
_RESOURCE_NAME_RE = re.compile(r'return\s+"(aws_[a-z0-9_]+)"')

# Regex to find schema.Schema{} attribute entries:
#   "bucket": {
#       Type:     schema.TypeString,
#       Required: true,
#   },
_ATTR_BLOCK_RE = re.compile(
    r'"([a-z][a-z0-9_]*)"\s*:\s*\{([^{}]*?)\}',
    re.DOTALL,
)

# Flags inside an attribute block
_REQUIRED_RE  = re.compile(r'\bRequired:\s*true')
_OPTIONAL_RE  = re.compile(r'\bOptional:\s*true')
_COMPUTED_RE  = re.compile(r'\bComputed:\s*true')
_SENSITIVE_RE = re.compile(r'\bSensitive:\s*true')
_TYPE_RE      = re.compile(r'\bType:\s*schema\.(\w+)')

# Nested block markers — when Type is TypeList/TypeSet/TypeMap with Elem containing Resource
_NESTED_RE = re.compile(r'\bType:\s*schema\.Type(?:List|Set|Map)')

# Nesting mode inference from Go type
_NESTING_MAP = {
    "TypeList": "list",
    "TypeSet": "set",
    "TypeMap": "map",
    "TypeSingle": "single",
}

# Simple Go type -> Terraform type string
_TF_TYPE_MAP = {
    "TypeString": "string",
    "TypeBool":   "bool",
    "TypeInt":    "number",
    "TypeFloat":  "number",
    "TypeList":   "list",
    "TypeSet":    "set",
    "TypeMap":    "map",
}


def _infer_resource_name_from_path(zip_path: str, root_prefix: str) -> str | None:
    """Derive the Terraform resource name from the file path as a fallback.

    internal/service/s3/resource_bucket.go  ->  aws_s3_bucket
    internal/service/ec2/resource_instance.go -> aws_ec2_instance  (approx)
    """
    rel = zip_path[len(root_prefix):]
    parts = PurePosixPath(rel).parts  # ('internal', 'service', 's3', 'resource_bucket.go')
    if len(parts) < 4:
        return None
    service = parts[2]
    filename = parts[-1]  # resource_bucket.go or data_source_bucket.go
    stem = filename.replace(".go", "")
    for prefix in ("resource_", "data_source_"):
        if stem.startswith(prefix):
            suffix = stem[len(prefix):]
            return f"aws_{service}_{suffix}"
    return None


def parse_attributes(go_source: str) -> dict[str, Any]:
    """Extract attribute definitions from a Go schema block."""
    attributes: dict[str, Any] = {}
    block_types: dict[str, Any] = {}

    for m in _ATTR_BLOCK_RE.finditer(go_source):
        attr_name = m.group(1)
        body = m.group(2)

        # Skip Go-internal identifiers that are not Terraform attributes
        if attr_name in (
            "schema", "resource", "provider", "elem", "validationfunc",
            "default", "description",
        ):
            continue

        is_nested = bool(_NESTED_RE.search(body))
        go_type   = (_TYPE_RE.search(body) or type("_", (), {"group": lambda *_: ""})()).group(1) or ""
        tf_type   = _TF_TYPE_MAP.get(go_type, "string")

        if is_nested:
            nesting_mode = _NESTING_MAP.get(go_type, "list")
            block_types[attr_name] = {
                "nesting_mode": nesting_mode,
                "min_items": 0,
                "max_items": 0,
                "attributes": {},
                "block_types": {},
            }
        else:
            attributes[attr_name] = {
                "type":      tf_type,
                "required":  bool(_REQUIRED_RE.search(body)),
                "optional":  bool(_OPTIONAL_RE.search(body)),
                "computed":  bool(_COMPUTED_RE.search(body)),
                "sensitive": bool(_SENSITIVE_RE.search(body)),
                "description": "",
            }

    return attributes, block_types


def extract_resources_from_zip(zip_bytes: bytes) -> dict[str, Any]:
    """Walk the source zip and build a resource_schemas dict."""
    resource_schemas: dict[str, Any] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Find the root prefix (e.g. "terraform-provider-aws-6.47.0/")
        root_prefix = ""
        for name in zf.namelist():
            if name.endswith("/") and name.count("/") == 1:
                root_prefix = name
                break
        print(f"[01] Zip root prefix: '{root_prefix}'")

        # Collect relevant Go files
        go_files = [
            name for name in zf.namelist()
            if (
                name.startswith(root_prefix + "internal/service/")
                and name.endswith(".go")
                and not name.endswith("_test.go")
                and ("/resource_" in name or "/data_source_" in name)
            )
        ]
        print(f"[01] Found {len(go_files):,} resource/data-source Go files.")

        for go_path in go_files:
            try:
                raw = zf.read(go_path).decode("utf-8", errors="replace")
            except Exception:
                continue

            # Try to find the declared resource name first
            name_matches = _RESOURCE_NAME_RE.findall(raw)
            if name_matches:
                resource_name = name_matches[0]
            else:
                resource_name = _infer_resource_name_from_path(go_path, root_prefix)

            if not resource_name:
                continue

            # Skip data sources (optional: remove this to include them)
            # They are still useful for schema context but kept separate for now.
            # Uncomment to include data sources:
            # if "/data_source_" in go_path:
            #     resource_name = resource_name.replace("aws_", "aws_data_")

            attributes, block_types = parse_attributes(raw)

            if not attributes and not block_types:
                continue  # empty parse — skip

            # Merge if the resource was already seen (multiple files per resource)
            if resource_name not in resource_schemas:
                resource_schemas[resource_name] = {
                    "block": {
                        "attributes": {},
                        "block_types": {},
                        "description": "",
                    }
                }

            existing = resource_schemas[resource_name]["block"]
            existing["attributes"].update(attributes)
            existing["block_types"].update(block_types)

    return resource_schemas

# ---------------------------------------------------------------------------
# Step 3: Wrap in the expected provider_schemas envelope
# ---------------------------------------------------------------------------

def build_schema_envelope(resource_schemas: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": "1.0",
        "provider_schemas": {
            _PROVIDER_KEY: {
                "provider": {"version": 0, "block": {"attributes": {}, "block_types": {}}},
                "resource_schemas": resource_schemas,
                "data_source_schemas": {},
            }
        },
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    zip_bytes = download_zip(_ZIP_URL)

    print("[01] Parsing resource schemas from Go source ...")
    resource_schemas = extract_resources_from_zip(zip_bytes)
    print(f"[01] Extracted {len(resource_schemas):,} resource definitions.")

    if not resource_schemas:
        print(
            "[01] ERROR: No resources extracted. "
            "Check the zip structure or regex patterns.",
            file=sys.stderr,
        )
        sys.exit(1)

    schema = build_schema_envelope(resource_schemas)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"[01] Saved {len(resource_schemas):,} resources to {OUTPUT_FILE}.")

    # Quick sanity check: print required attributes for aws_s3_bucket
    probe = resource_schemas.get("aws_s3_bucket")
    if probe:
        req_attrs = [
            k for k, v in probe["block"]["attributes"].items()
            if v.get("required")
        ]
        print(f"[01] Sanity check aws_s3_bucket required attrs: {req_attrs}")
    else:
        print("[01] Note: aws_s3_bucket not found in extracted schemas (unexpected).")
