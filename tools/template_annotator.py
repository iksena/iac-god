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
# hcl2 structure helpers
# ---------------------------------------------------------------------------

def _hcl2_unwrap(value: Any) -> Any:
    """Unwrap a single-element list produced by hcl2.

    hcl2.load() wraps every dict value in a list::

        {"resource": [{"aws_vpc": [{"main": {"cidr_block": ["10.0.0.0/16"]}}]}]}

    This helper peels off one list layer when the list has exactly one
    element, leaving plain dicts and other types untouched.  The result is
    never further unwrapped here; callers chain calls as needed.
    """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _hcl2_lookup(raw: dict, rtype: str, rname: str) -> dict:
    """Safely drill into the hcl2 resource structure to retrieve the raw block.

    hcl2 nests resources as::

        raw["resource"] -> list
          -> dict keyed by resource type
            -> list
              -> dict keyed by resource name
                -> the raw attribute dict

    Each level is wrapped in a list, so we unwrap at each step.
    Returns an empty dict on any shape mismatch so _parse_terraform
    never raises.
    """
    try:
        resources_outer = _hcl2_unwrap(raw.get("resource", []))
        if not isinstance(resources_outer, dict):
            return {}
        type_outer = _hcl2_unwrap(resources_outer.get(rtype, []))
        if not isinstance(type_outer, dict):
            return {}
        name_val = _hcl2_unwrap(type_outer.get(rname, {}))
        return name_val if isinstance(name_val, dict) else {}
    except Exception:
        return {}


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

    Uses python-hcl2 when available to extract the raw attribute dict for
    each resource.  hcl2.load() wraps every nesting level in a list, e.g.::

        {"resource": [{"aws_vpc": [{"main": {"cidr_block": ["10.0.0.0/16"]}}]}]}

    _hcl2_lookup() unwraps those list layers safely so ResourceAnnotation.raw
    receives a plain dict.  Falls back to an empty dict when hcl2 is absent
    or the structure does not match expectations.
    """
    resources: list[ResourceAnnotation] = []

    raw: dict = {}
    try:
        import hcl2  # type: ignore
        import io
        raw = hcl2.load(io.StringIO(content))
    except ImportError:
        logger.debug("python-hcl2 not installed; using regex fallback for %s", file_path)
    except Exception as exc:
        logger.warning("hcl2 parse error for %s: %s", file_path, exc)

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
            raw=_hcl2_lookup(raw, rtype, rname),
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
# CloudFormation annotated YAML renderer
# ---------------------------------------------------------------------------

# Priority 1 — cfn-lint dict-repr:  {'LineNumber': 115, 'ColumnNumber': 7}
_DICT_LINENO_RE = re.compile(r"'LineNumber'\s*:\s*(\d+)")

# Priority 2 — deploy colon-ref:    :115  or  :115:7
_COLON_LINENO_RE = re.compile(r":(\d+)(?::\d+)?")

# Priority 3 — YAML parser prose:   line 24  or  line 24, column 21
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
    """Inject error comments into the raw CloudFormation template at the
    reported line numbers.

    For each error that carries a LineNumber, a '# ERROR: ...' comment is
    inserted immediately BEFORE that line in the original template text.
    The template itself is never re-serialised, so all nested properties,
    intrinsic tags (!Ref, !Sub, ...) and formatting are preserved exactly.

    Errors without a detectable line number (e.g. deploy failures that only
    report a resource name) are collected into a header comment block above
    the template.

    Args:
        template_yaml: Raw CloudFormation template string.
        errors:        Flat list of error strings (security stages excluded).

    Returns:
        The full template string with inline # ERROR: comment annotations.
    """
    if not template_yaml:
        return "# No template available.\n" + "\n".join(f"# ERROR: {e}" for e in errors)

    errors_by_line: dict[int, list[str]] = defaultdict(list)
    for err in errors:
        lineno = _extract_line_number(err)
        errors_by_line[lineno if lineno is not None else 0].append(err)

    source_lines = template_yaml.splitlines()
    output: list[str] = []

    if errors_by_line.get(0):
        output.append("# --- Errors without line numbers (deploy failures) ---")
        for err in errors_by_line[0]:
            output.append(f"# ERROR: {err}")
        output.append("")

    for lineno, line in enumerate(source_lines, start=1):
        for err in errors_by_line.get(lineno, []):
            indent = len(line) - len(line.lstrip())
            output.append(" " * indent + f"# ERROR: {err}")
        output.append(line)

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Terraform annotated HCL renderer
# ---------------------------------------------------------------------------

# Stage name substrings that carry line-referenced errors worth annotating.
# Security (checkov/trivy) and deploy stages report resource-level or
# policy-level findings that have no HCL line number, so they are excluded.
_TF_ANNOTATABLE_STAGES = ("tflint", "terraform-validate", "terraform validate")

# Line-number extraction patterns for Terraform tool output.
# Priority 1 — terraform validate / tflint prose:  "line N, column M"
_TF_PROSE_LINE_RE = re.compile(r"(?<!\w)line\s+(\d+)", re.IGNORECASE)

# Priority 2 — tflint --format compact:  main.tf:N:M:
_TF_COMPACT_RE = re.compile(r"(?:main\.tf|\S+\.tf):(\d+):\d+")

# Priority 3 — bare colon-ref of last resort:  :N:M  or  :N
_TF_COLON_RE = re.compile(r":(\d+)(?::\d+)?")


def _extract_tf_line_number(error: str) -> int | None:
    """Extract a line number from a tflint or terraform validate error string.

    Returns None when no line reference is found; such errors are silently
    dropped by render_annotated_terraform (matching the CFN behaviour where
    deploy errors without line numbers are not annotated inline).
    """
    m = _TF_PROSE_LINE_RE.search(error)
    if m:
        return int(m.group(1))
    m = _TF_COMPACT_RE.search(error)
    if m:
        return int(m.group(1))
    m = _TF_COLON_RE.search(error)
    if m:
        return int(m.group(1))
    return None


def _collapse_to_one_line(error: str) -> str:
    """Flatten a multi-line error message to a single line.

    Consecutive whitespace (including newlines) is collapsed to a single
    space and the result is stripped so the # ERROR: comment stays on one
    line inside the HCL file.
    """
    return re.sub(r"\s+", " ", error).strip()


def render_annotated_terraform(
    template_hcl: str,
    stage_errors: dict[str, list[str]],
) -> str:
    """Inject single-line error comments into a Terraform HCL template.

    Only errors from stages whose names contain 'tflint' or
    'terraform-validate' / 'terraform validate' are considered — these are
    the structural and syntax stages that report line numbers.  Security
    (checkov/trivy) and deploy stage errors are excluded entirely because
    they carry no HCL line reference.

    Each qualifying error is collapsed to one line and injected as:
        # ERROR: <message>
    immediately before the source line it references.  Errors without a
    detectable line number are silently dropped (consistent with the CFN
    renderer not annotating deploy failures inline).

    Args:
        template_hcl:  Raw HCL string (the LLM-generated main.tf content).
        stage_errors:  Mapping of stage_name -> list[error_string], as
                       produced by the validator pipeline.

    Returns:
        The full HCL string with inline # ERROR: comment annotations.
    """
    if not template_hcl:
        return "# No template available.\n"

    # Collect errors only from annotatable stages.
    qualifying_errors: list[str] = []
    for stage_name, errs in stage_errors.items():
        stage_lower = stage_name.lower()
        if any(marker in stage_lower for marker in _TF_ANNOTATABLE_STAGES):
            qualifying_errors.extend(errs)

    if not qualifying_errors:
        return template_hcl

    # Group by target line number; drop errors with no detectable line.
    errors_by_line: dict[int, list[str]] = defaultdict(list)
    for err in qualifying_errors:
        lineno = _extract_tf_line_number(err)
        if lineno is not None:
            errors_by_line[lineno].append(_collapse_to_one_line(err))

    if not errors_by_line:
        return template_hcl

    source_lines = template_hcl.splitlines()
    output: list[str] = []

    for lineno, line in enumerate(source_lines, start=1):
        for err in errors_by_line.get(lineno, []):
            # Preserve the indentation of the target line so the comment
            # sits flush with the HCL attribute it annotates.
            indent = len(line) - len(line.lstrip())
            output.append(" " * indent + f"# ERROR: {err}")
        output.append(line)

    return "\n".join(output)
