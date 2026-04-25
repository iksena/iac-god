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

    Uses yaml.add_multi_constructor so any unknown !Tag is caught — future
    CFN tags won't break the parser.
    """
    class CFNLoader(yaml.SafeLoader):
        pass

    yaml.add_multi_constructor("!", _cfn_constructor, Loader=CFNLoader)

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
    """Parse a CloudFormation YAML template safely.

    All !Ref / !GetAtt / etc. are preserved as _CFNTag objects.
    """
    return yaml.load(content, Loader=CFN_LOADER)  # noqa: S506 — custom loader, not bare load


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResourceAnnotation:
    """Metadata for a single IaC resource block."""
    resource_id: str
    resource_type: str
    start_line: int
    end_line: int | None
    raw: dict
    smells: list[dict] = field(default_factory=list)


@dataclass
class TemplateAnnotation:
    """Full annotation for a single IaC template file."""
    file_path: str
    template_type: str
    raw: dict
    resources: list[ResourceAnnotation] = field(default_factory=list)
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Line-number tracker for YAML nodes
# ---------------------------------------------------------------------------

def _yaml_with_line_numbers(content: str) -> dict[str, int]:
    """Return a map of {resource_logical_id: start_line} by scanning raw YAML text."""
    line_map: dict[str, int] = {}
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
            if stripped and not stripped.startswith("#"):
                if resource_indent is None:
                    resource_indent = indent
                if indent == resource_indent and stripped.endswith(":"):
                    resource_id = stripped.rstrip(":")
                    line_map[resource_id] = lineno
                elif indent < resource_indent:
                    in_resources = False

    return line_map


# ---------------------------------------------------------------------------
# Template parsers
# ---------------------------------------------------------------------------

def _parse_cloudformation(content: str, file_path: str) -> TemplateAnnotation:
    """Parse a CloudFormation template (YAML or JSON)."""
    try:
        raw = load_cfn_yaml(content)
        line_map = _yaml_with_line_numbers(content)
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

        rtype = resource_body.get("Type", "")
        if isinstance(rtype, _CFNTag):
            rtype = str(rtype.value)

        resources.append(ResourceAnnotation(
            resource_id=logical_id,
            resource_type=rtype,
            start_line=line_map.get(logical_id, 0),
            end_line=None,
            raw=resource_body,
        ))

    return TemplateAnnotation(
        file_path=file_path,
        template_type="cloudformation",
        raw=raw,
        resources=resources,
    )


def _parse_terraform(content: str, file_path: str) -> TemplateAnnotation:
    """Minimal Terraform HCL parser using regex for resource block detection.

    For production use, swap with python-hcl2 or pyhcl.
    """
    resources: list[ResourceAnnotation] = []

    try:
        import hcl2  # type: ignore
        import io
        raw = hcl2.load(io.StringIO(content))
    except ImportError:
        logger.debug("python-hcl2 not installed; using regex fallback for %s", file_path)
        raw = {}

    pattern = re.compile(
        r'^resource\s+"(?P<type>[^"]+)"\s+"(?P<name>[^"]+)"\s*\{',
        re.MULTILINE,
    )
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
    if Path(file_path).suffix.lower() in (".tf",):
        return "terraform"

    if any(marker in content for marker in _CFN_MARKERS):
        return "cloudformation"

    if Path(file_path).suffix.lower() in (".json",):
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
    """Parse and annotate an IaC template file.

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
    """Attach static-analysis smell findings to the relevant ResourceAnnotation.

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
            logger.debug(
                "Smell %s could not be mapped to a resource (resource_id=%r)",
                smell.get("rule_id"),
                rid,
            )

    return annotation


# ---------------------------------------------------------------------------
# Annotated YAML renderer for retriever prompt
# ---------------------------------------------------------------------------

def render_annotated_template(
    annotation: TemplateAnnotation,
    errors: list[str],
    include_security_smells: bool = False,
) -> str:
    """Re-serialise the CFN template as YAML with inline comments anchoring
    each validation/deployment error to the resource block it belongs to.

    Produces output like:

        Resources:
          MyBucket:  # AWS::S3::Bucket | line 12
            # ERROR: cfn-lint: E3001 Invalid resource type
            # ERROR: deploy: CREATE_FAILED - BucketName must be globally unique
            Type: AWS::S3::Bucket
            Properties:
              ...

    Args:
        annotation:              Output of annotate_template() + attach_smells().
        errors:                  Flat list of error strings from _extract_errors().
                                 Must already have security stages filtered out.
        include_security_smells: When True, append # SMELL: comments for each
                                 smell attached to the resource. Default False so
                                 the retriever stays focused on structural errors.

    Returns:
        A YAML string of the Resources section with inline error comments.
        Falls back to a plain error list when the annotation has no resources.
    """
    if not annotation or not annotation.resources:
        if errors:
            return "# No parseable template — errors:\n" + "\n".join(
                f"# ERROR: {e}" for e in errors
            )
        return "# No template or errors available."

    resource_errors: dict[str, list[str]] = {r.resource_id: [] for r in annotation.resources}
    template_level_errors: list[str] = []

    for err in errors:
        matched = False
        for r in annotation.resources:
            if r.resource_id in err or (r.resource_type and r.resource_type in err):
                resource_errors[r.resource_id].append(err)
                matched = True
                break
        if not matched:
            template_level_errors.append(err)

    lines: list[str] = []

    if template_level_errors:
        lines.append("# --- Template-level errors ---")
        for err in template_level_errors:
            lines.append(f"# ERROR: {err}")
        lines.append("")

    lines.append("Resources:")

    for r in annotation.resources:
        line_hint = f" | line {r.start_line}" if r.start_line else ""
        lines.append(f"  {r.resource_id}:  # {r.resource_type}{line_hint}")

        for err in resource_errors.get(r.resource_id, []):
            lines.append(f"    # ERROR: {err}")

        if include_security_smells:
            for smell in r.smells:
                rule_id = smell.get("rule_id", "?")
                desc = smell.get("description", "")
                lines.append(f"    # SMELL: {rule_id} — {desc}")

        rtype = r.resource_type or "Unknown"
        lines.append(f"    Type: {rtype}")

        props = r.raw.get("Properties", {}) if r.raw else {}
        if props:
            lines.append("    Properties:")
            for key in sorted(props.keys()):
                val = props[key]
                if isinstance(val, (str, int, float, bool)) or val is None:
                    lines.append(f"      {key}: {val}")
                elif isinstance(val, dict):
                    lines.append(f"      {key}: {{...}}  # {len(val)} keys")
                elif isinstance(val, list):
                    lines.append(f"      {key}: [...]  # {len(val)} items")
                else:
                    lines.append(f"      {key}: <{type(val).__name__}>")
        lines.append("")

    return "\n".join(lines).rstrip()
