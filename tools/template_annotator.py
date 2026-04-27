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
from collections import defaultdict
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
    """Parse a CloudFormation YAML template, preserving intrinsic tags."""
    return yaml.load(content, Loader=CFN_LOADER)  # noqa: S506


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
    """Return {resource_logical_id: start_line} by scanning raw YAML text."""
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
    """Minimal Terraform HCL parser using regex for resource block detection."""
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
    """Parse and annotate an IaC template file."""
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
    """Attach static-analysis smell findings to the relevant ResourceAnnotation."""
    resource_index = {r.resource_id: r for r in annotation.resources}
    for smell in smell_report:
        rid = smell.get("resource_id", "")
        if rid in resource_index:
            resource_index[rid].smells.append(smell)
        else:
            logger.debug(
                "Smell %s could not be mapped to a resource (resource_id=%r)",
                smell.get("rule_id"), rid,
            )
    return annotation


def extract_resource_types(annotation: TemplateAnnotation | None) -> set[str]:
    """Return the set of AWS resource type strings present in an annotation."""
    if not annotation or not annotation.resources:
        return set()
    return {r.resource_type for r in annotation.resources if r.resource_type}


# ---------------------------------------------------------------------------
# Annotated YAML renderer for retriever / remediator / engineer prompts
# ---------------------------------------------------------------------------

# Priority 1 — cfn-lint dict-repr:  {'LineNumber': 115, 'ColumnNumber': 7}
_DICT_LINENO_RE = re.compile(r"'LineNumber'\s*:\s*(\d+)")

# Priority 2 — deploy colon-ref:    :115  or  :115:7
_COLON_LINENO_RE = re.compile(r":(\d+)(?::\d+)?")

# Priority 3 — YAML parser prose:   line 24  or  line 24, column 21
#   Matches "line <N>" that is NOT preceded by another digit or word char so
#   we don't accidentally match things like "baseline 24" or "outline 24".
_PROSE_LINENO_RE = re.compile(r"(?<!\w)line\s+(\d+)", re.IGNORECASE)


def _extract_line_number(error: str) -> int | None:
    """Return the line number embedded in an error string.

    Tries three formats in priority order:
      1. cfn-lint dict-repr  — {'LineNumber': N}
      2. colon-separated     — :N  or  :N:M
      3. YAML parser prose   — "line N"  or  "line N, column M"

    Returns None when no line reference is found (error goes to header block).
    """
    m = _DICT_LINENO_RE.search(error)
    if m:
        return int(m.group(1))
    m = _COLON_LINENO_RE.search(error)
    if m:
        return int(m.group(1))
    m = _PROSE_LINENO_RE.search(error)
    if m:
        return int(m.group(1))
    return None


def render_annotated_template(
    template_yaml: str,
    errors: list[str],
) -> str:
    """Inject error comments into the raw template at the reported line numbers.

    For each error that carries a LineNumber, a '# ERROR: ...' comment is
    inserted immediately BEFORE that line in the original template text.
    The template itself is never re-serialised, so all nested properties,
    intrinsic tags (!Ref, !Sub, ...) and formatting are preserved exactly.

    Errors without a detectable line number (e.g. deploy failures that only
    report a resource name) are collected into a header comment block above
    the template.

    Line number extraction handles three formats:
      - cfn-lint dict-repr:  {'LineNumber': N}
      - deploy colon-ref:    :N  or  :N:M
      - YAML parser prose:   "line N"  or  "line N, column M"

    Args:
        template_yaml: Raw CloudFormation template string.
        errors:        Flat list of error strings (security stages excluded).

    Returns:
        The full template string with inline # ERROR: comment annotations.
    """
    if not template_yaml:
        return "# No template available.\n" + "\n".join(f"# ERROR: {e}" for e in errors)

    # Group errors by their target line number.
    # Errors with no line number go into the header (line 0).
    errors_by_line: dict[int, list[str]] = defaultdict(list)
    for err in errors:
        lineno = _extract_line_number(err)
        errors_by_line[lineno if lineno is not None else 0].append(err)

    source_lines = template_yaml.splitlines()
    output: list[str] = []

    # Header block for errors with no detectable line number.
    if errors_by_line.get(0):
        output.append("# --- Errors without line numbers (deploy failures) ---")
        for err in errors_by_line[0]:
            output.append(f"# ERROR: {err}")
        output.append("")

    # Walk source lines, injecting error comments before the target line.
    for lineno, line in enumerate(source_lines, start=1):
        for err in errors_by_line.get(lineno, []):
            # Preserve the indentation of the target line so the comment
            # sits flush with the YAML key it annotates.
            indent = len(line) - len(line.lstrip())
            output.append(" " * indent + f"# ERROR: {err}")
        output.append(line)

    return "\n".join(output)
