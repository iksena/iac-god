"""
tools/template_annotator.py

Parses IaC templates (CloudFormation YAML/JSON, Terraform HCL) and annotates
them with metadata needed by the engineer and remediation agents:
  - file path, template type, resource blocks
  - detected security smells (via static analysis hook)
  - line numbers for each resource block (for targeted patching)
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe CloudFormation YAML Loader
# ---------------------------------------------------------------------------

class _CFNTag:
    """Represents a CloudFormation intrinsic function tag (e.g. !Ref, !Sub)."""

    def __init__(self, tag: str, value: Any) -> None:
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover
        return f"CFNTag({self.tag!r}, {self.value!r})"

    def to_dict(self) -> dict:
        """Serialise back to a CFN-style dict for downstream processing."""
        # e.g. !Ref Foo  →  {"Ref": "Foo"}
        short = self.tag.lstrip("!")
        return {short: self.value}


def _cfn_constructor(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> _CFNTag:
    """Generic multi-type constructor for any CloudFormation !Tag."""
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:
        value = loader.construct_scalar(node)
    return _CFNTag(tag=tag_suffix, value=value)


def _build_cfn_loader() -> type[yaml.Loader]:
    """
    Returns a YAML Loader subclass that safely handles all CloudFormation
    intrinsic function tags without raising ConstructorError.

    Uses yaml.add_multi_constructor so any unknown !Tag is caught, not just
    a hardcoded list — future CFN tags won't break the parser.
    """
    class CFNLoader(yaml.SafeLoader):
        pass

    # Catch ALL custom tags (any string starting with "!")
    yaml.add_multi_constructor("!", _cfn_constructor, Loader=CFNLoader)

    # Also explicitly register the most common ones for robustness
    _known_cfn_tags = [
        "!Ref", "!Sub", "!GetAtt", "!If", "!Not", "!And", "!Or",
        "!Equals", "!Select", "!Join", "!Split", "!FindInMap",
        "!Base64", "!Cidr", "!ImportValue", "!Transform",
        "!Condition", "!GetAZs",
    ]
    for tag in _known_cfn_tags:
        yaml.add_constructor(tag, lambda l, n, t=tag: _cfn_constructor(l, t, n), Loader=CFNLoader)

    return CFNLoader


CFN_LOADER = _build_cfn_loader()


def load_cfn_yaml(content: str) -> dict:
    """
    Parse a CloudFormation YAML template safely.
    All !Ref / !GetAtt / etc. are preserved as _CFNTag objects.
    """
    return yaml.load(content, Loader=CFN_LOADER)  # noqa: S506 — custom loader, not bare load


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResourceAnnotation:
    """Metadata for a single IaC resource block."""
    resource_id: str                     # Logical ID (CFN) or address (TF)
    resource_type: str                   # e.g. "AWS::S3::Bucket", "aws_s3_bucket"
    start_line: int                      # 1-indexed
    end_line: int | None                 # None if unknown (JSON/HCL)
    raw: dict                            # Parsed properties
    smells: list[dict] = field(default_factory=list)   # Populated by smell detector


@dataclass
class TemplateAnnotation:
    """Full annotation for a single IaC template file."""
    file_path: str
    template_type: str                   # "cloudformation" | "terraform" | "unknown"
    raw: dict                            # Full parsed template
    resources: list[ResourceAnnotation] = field(default_factory=list)
    parse_error: str | None = None       # Set if parsing failed


# ---------------------------------------------------------------------------
# Line-number tracker for YAML nodes
# ---------------------------------------------------------------------------

def _yaml_with_line_numbers(content: str) -> dict[str, int]:
    """
    Returns a map of {resource_logical_id: start_line} by scanning the raw
    YAML text. Used to annotate resources with their source line numbers.

    We do a lightweight regex scan rather than a full AST walk because
    PyYAML's composed nodes expose line info only during construction.
    """
    line_map: dict[str, int] = {}
    # Match top-level keys under Resources: section
    in_resources = False
    resource_indent: int | None = None

    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if re.match(r"^Resources\s*:", line):
            in_resources = True
            resource_indent = None
            continue

        if in_resources:
            # First non-empty child of Resources sets the indent level
            if stripped and not stripped.startswith("#"):
                if resource_indent is None:
                    resource_indent = indent
                if indent == resource_indent and stripped.endswith(":"):
                    resource_id = stripped.rstrip(":")
                    line_map[resource_id] = lineno
                elif indent < resource_indent:
                    # Left the Resources block
                    in_resources = False

    return line_map


# ---------------------------------------------------------------------------
# Template parsers
# ---------------------------------------------------------------------------

def _parse_cloudformation(content: str, file_path: str) -> TemplateAnnotation:
    """Parse a CloudFormation template (YAML or JSON)."""
    path = Path(file_path)
    raw: dict = {}

    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            raw = load_cfn_yaml(content)
            line_map = _yaml_with_line_numbers(content)
        else:
            # JSON CloudFormation
            raw = json.loads(content)
            line_map = {}
    except Exception as exc:
        logger.warning("Failed to parse CloudFormation template %s: %s", file_path, exc)
        return TemplateAnnotation(
            file_path=file_path,
            template_type="cloudformation",
            raw={},
            parse_error=str(exc),
        )

    resources: list[ResourceAnnotation] = []
    for logical_id, resource_body in (raw.get("Resources") or {}).items():
        if not isinstance(resource_body, dict):
            continue

        # Resolve _CFNTag wrappers in Type field (shouldn't happen but be safe)
        rtype = resource_body.get("Type", "")
        if isinstance(rtype, _CFNTag):
            rtype = str(rtype.value)

        start_line = line_map.get(logical_id, 0)
        resources.append(ResourceAnnotation(
            resource_id=logical_id,
            resource_type=rtype,
            start_line=start_line,
            end_line=None,          # CFN YAML end lines need AST walk; deferred
            raw=resource_body,
        ))

    return TemplateAnnotation(
        file_path=file_path,
        template_type="cloudformation",
        raw=raw,
        resources=resources,
    )


def _parse_terraform(content: str, file_path: str) -> TemplateAnnotation:
    """
    Minimal Terraform HCL parser using regex for resource block detection.
    For production use, swap with python-hcl2 or pyhcl.
    """
    resources: list[ResourceAnnotation] = []

    try:
        # Try python-hcl2 if available
        import hcl2  # type: ignore
        import io
        raw = hcl2.load(io.StringIO(content))
    except ImportError:
        logger.debug("python-hcl2 not installed; using regex fallback for %s", file_path)
        raw = {}

    # Regex fallback: detect resource blocks and line numbers
    pattern = re.compile(
        r'^resource\s+"(?P<type>[^"]+)"\s+"(?P<name>[^"]+)"\s*\{',
        re.MULTILINE,
    )
    lines = content.splitlines()
    for match in pattern.finditer(content):
        rtype = match.group("type")
        rname = match.group("name")
        start_line = content[: match.start()].count("\n") + 1
        resources.append(ResourceAnnotation(
            resource_id=f"{rtype}.{rname}",
            resource_type=rtype,
            start_line=start_line,
            end_line=None,
            raw=raw.get("resource", {}).get(rtype, {}).get(rname, {}),
        ))

    return TemplateAnnotation(
        file_path=file_path,
        template_type="terraform",
        raw=raw,
        resources=resources,
    )


# ---------------------------------------------------------------------------
# Template type detection
# ---------------------------------------------------------------------------

_CFN_MARKERS = {"AWSTemplateFormatVersion", "Resources", "Outputs", "Parameters"}


def _detect_template_type(content: str, file_path: str) -> str:
    """Heuristically detect whether a file is CloudFormation or Terraform."""
    path = Path(file_path)

    if path.suffix.lower() in (".tf",):
        return "terraform"

    # Try to detect CFN from content markers
    if any(marker in content for marker in _CFN_MARKERS):
        return "cloudformation"

    if path.suffix.lower() in (".json",):
        try:
            doc = json.loads(content)
            if isinstance(doc, dict) and "Resources" in doc:
                return "cloudformation"
        except json.JSONDecodeError:
            pass

    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_template(file_path: str, content: str | None = None) -> TemplateAnnotation:
    """
    Parse and annotate an IaC template file.

    Args:
        file_path: Absolute or relative path to the template file.
        content:   Raw file content. If None, the file is read from disk.

    Returns:
        TemplateAnnotation with resource-level metadata and line numbers.
        Smell detection is NOT performed here — call attach_smells() after
        running your static analysis tool (Checkov, cfn-lint, tfsec, etc.).
    """
    if content is None:
        content = Path(file_path).read_text(encoding="utf-8")

    template_type = _detect_template_type(content, file_path)
    logger.debug("Detected template type '%s' for %s", template_type, file_path)

    if template_type == "cloudformation":
        return _parse_cloudformation(content, file_path)
    elif template_type == "terraform":
        return _parse_terraform(content, file_path)
    else:
        return TemplateAnnotation(
            file_path=file_path,
            template_type="unknown",
            raw={},
            parse_error=f"Unrecognised template format for file: {file_path}",
        )


def attach_smells(
    annotation: TemplateAnnotation,
    smell_report: list[dict],
) -> TemplateAnnotation:
    """
    Attach static-analysis smell findings to the relevant ResourceAnnotation.

    Args:
        annotation:   Output of annotate_template().
        smell_report: List of smell dicts, each containing at minimum:
                      {"resource_id": str, "rule_id": str, "severity": str,
                       "description": str, "line": int}

    Returns:
        The same TemplateAnnotation with smells attached to resources.
    """
    resource_index = {r.resource_id: r for r in annotation.resources}

    for smell in smell_report:
        rid = smell.get("resource_id", "")
        if rid in resource_index:
            resource_index[rid].smells.append(smell)
        else:
            # Smell is file-level or couldn't be mapped to a resource
            logger.debug(
                "Smell %s could not be mapped to a resource (resource_id=%r)",
                smell.get("rule_id"),
                rid,
            )

    return annotation


def annotation_to_history_entry(annotation: TemplateAnnotation) -> dict:
    """
    Convert a TemplateAnnotation into a remediation history entry dict
    suitable for injection into the agent prompt context.

    This is the bridge between template_annotator and the prompt builder.
    """
    return {
        "file_path": annotation.file_path,
        "template_type": annotation.template_type,
        "parse_error": annotation.parse_error,
        "resources": [
            {
                "resource_id": r.resource_id,
                "resource_type": r.resource_type,
                "start_line": r.start_line,
                "smells": r.smells,
            }
            for r in annotation.resources
        ],
    }