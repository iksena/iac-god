# tools/validators.py
import subprocess
import tempfile
import json
from pathlib import Path
from state import ValidationResult
from yamllint import linter
from yamllint.config import YamlLintConfig
from config import DeployConfig, DeployTarget, DEFAULT_DEPLOY_CONFIG
from tools.deploy_validator import validate_deployment
from state import ValidationResult, DeployValidationResult


def _derive_policy_rates(total_policies: int, passed_policies: int, filtered_failed_policies: int) -> tuple[float, float]:
    if total_policies <= 0:
        return 1.0, 1.0
    ppr = passed_policies / total_policies
    fcr = (total_policies - filtered_failed_policies) / total_policies
    return ppr, fcr

_YAMLLINT_CONFIG = YamlLintConfig(
    """
extends: default
rules:
    document-start: disable
    line-length: disable
    trailing-spaces: disable
    new-line-at-end-of-file: disable
    indentation:
        spaces: consistent
        indent-sequences: consistent
    truthy:
        allowed-values: ['true', 'false', 'yes', 'no']
"""
)

def validate_yaml(template: str) -> ValidationResult:
    """Stage 1: Basic YAML syntax and style check via yamllint."""
    problems = list(linter.run(template, _YAMLLINT_CONFIG))
    errors = [
        f"[{problem.rule}] line {problem.line}, column {problem.column}: {problem.desc}"
        for problem in problems
    ]
    raw_output = "YAML syntax OK" if not errors else "\n".join(errors)
    return ValidationResult(
        stage="yaml",
        passed=not errors,
        errors=errors,
        raw_output=raw_output,
    )


def _format_cfn_lint_finding(finding: dict) -> str:
    """Format a single cfn-lint JSON finding into a structured error string.

    Output format (all fields that are present):
        [W3005] line 42 | Resource: MyBucket | <message> | <rule description> | See: <url>

    The 'line NN' token is intentionally kept as a plain word-number pair so
    that the regex in retriever.py (_WORD_LINE_RE) and the line-extraction
    logic in template_annotator.py can both find it without needing to parse
    a raw Python dict repr.
    """
    rule    = finding.get("Rule") or {}
    rule_id = rule.get("Id") or "?"

    location = finding.get("Location") or {}
    start    = location.get("Start") or {}
    line_num = start.get("LineNumber")          # int or None

    # Resource logical ID sits at Path[1] when present.
    path = location.get("Path") or []
    resource = path[1] if len(path) > 1 else None

    message     = (finding.get("Message") or "").strip()
    description = (rule.get("Description") or "").strip()
    # source_url  = (rule.get("Source") or "").strip()

    parts: list[str] = [f"[{rule_id}]"]
    if line_num is not None:
        parts.append(f"line {line_num}")
    if resource:
        parts.append(f"Resource: {resource}")
    if message:
        parts.append(message)
    if description and description.lower() != message.lower():
        parts.append(description)
    # if source_url:
    #     parts.append(f"See: {source_url}")

    return " | ".join(parts)


def validate_cfn_lint(template: str) -> ValidationResult:
    """Stage 2: AWS CloudFormation linting via cfn-lint."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(template)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["cfn-lint", tmp_path, "--format", "json"],
            capture_output=True, text=True, timeout=60,
        )
        errors = []
        raw = result.stdout or result.stderr
        if result.returncode != 0:
            try:
                findings = json.loads(raw)
                errors = [_format_cfn_lint_finding(f) for f in findings]
            except json.JSONDecodeError:
                errors = [raw]
        return ValidationResult(
            stage="cfn-lint",
            passed=result.returncode == 0,
            errors=errors,
            raw_output=raw,
        )
    except FileNotFoundError:
        return ValidationResult(
            stage="cfn-lint", passed=False,
            errors=["cfn-lint not installed. Run: pip install cfn-lint"],
            raw_output="TOOL_NOT_FOUND",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def validate_checkov(template: str) -> ValidationResult:
    """Stage 3: Security policy check via Checkov."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(template)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [
                "checkov", "-f", tmp_path,
                "--framework", "cloudformation",
                "--output", "json",
                "--quiet",
            ],
            capture_output=True, text=True, timeout=120,
        )
        raw = result.stdout or result.stderr
        errors = []
        total_policies = 0
        passed_policies = 0
        failed_policies = 0
        filtered_failed_policies = 0
        try:
            data = json.loads(raw)
            failed = data.get("results", {}).get("failed_checks", [])
            passed = data.get("results", {}).get("passed_checks", [])
            failed_policies = len(failed)
            passed_policies = len(passed)
            total_policies = failed_policies + passed_policies

            for check in failed:
                severity = str(check.get("severity", "")).lower()
                if severity in ("high", "critical"):
                    filtered_failed_policies += 1

            errors = [
                f"[{c['check_id']}] {c['check_result']['result']}: "
                f"{c['resource']} \u2014 {c['check'].get('name','')}"
                for c in failed
            ]
        except (json.JSONDecodeError, KeyError):
            if result.returncode not in (0, 1):
                errors = [raw]

        ppr, fcr = _derive_policy_rates(total_policies, passed_policies, filtered_failed_policies)

        return ValidationResult(
            stage="checkov",
            passed=len(errors) == 0,
            errors=errors,
            raw_output=raw,
            policy_stats={
                "total_policies": total_policies,
                "passed_policies": passed_policies,
                "failed_policies": failed_policies,
                "filtered_failed_policies": filtered_failed_policies,
            },
            scenario_policy_pass_rate=ppr,
            filtered_compliance_rate=fcr,
        )
    except FileNotFoundError:
        return ValidationResult(
            stage="checkov", passed=False,
            errors=["checkov not installed. Run: pip install checkov"],
            raw_output="TOOL_NOT_FOUND",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def validate_trivy(template: str) -> ValidationResult:
    """Stage 4: Misconfiguration scan via Trivy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfn_path = Path(tmpdir) / "template.yaml"
        cfn_path.write_text(template)
        try:
            result = subprocess.run(
                [
                    "trivy", "config",
                    "--format", "json",
                    "--exit-code", "1",
                    str(tmpdir),
                ],
                capture_output=True, text=True, timeout=120,
            )
            raw = result.stdout or result.stderr
            errors = []
            total_policies = 0
            passed_policies = 0
            failed_policies = 0
            filtered_failed_policies = 0
            try:
                data = json.loads(raw)
                for r in data.get("Results", []):
                    summary = r.get("MisconfSummary") or {}
                    passed_policies += int(summary.get("Successes", 0) or 0)
                    failed_policies += int(summary.get("Failures", 0) or 0)

                    for m in r.get("Misconfigurations", []):
                        if m.get("Severity", "").lower() in ("high", "critical"):
                            filtered_failed_policies += 1
                            errors.append(
                                f"[{m['ID']}] {m['Severity']}: {m['Title']} \u2014 {m['Message']}"
                            )

                total_policies = passed_policies + failed_policies
            except (json.JSONDecodeError, KeyError):
                if result.returncode not in (0, 1):
                    errors = [raw]

            ppr, fcr = _derive_policy_rates(total_policies, passed_policies, filtered_failed_policies)

            return ValidationResult(
                stage="trivy",
                passed=len(errors) == 0,
                errors=errors,
                raw_output=raw,
                policy_stats={
                    "total_policies": total_policies,
                    "passed_policies": passed_policies,
                    "failed_policies": failed_policies,
                    "filtered_failed_policies": filtered_failed_policies,
                },
                scenario_policy_pass_rate=ppr,
                filtered_compliance_rate=fcr,
            )
        except FileNotFoundError:
            return ValidationResult(
                stage="trivy", passed=False,
                errors=["trivy not installed. See: https://aquasecurity.github.io/trivy"],
                raw_output="TOOL_NOT_FOUND",
            )


def run_all_validators(
    template: str,
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> tuple[list[ValidationResult], bool, DeployValidationResult]:
    """
    Run all validation stages. Returns (static_results, all_passed, deploy_result).
    Deploy stage only runs if all static validators pass.
    """
    yaml_result = validate_yaml(template)
    cfn_lint_result = validate_cfn_lint(template)

    results = [
        yaml_result,
        cfn_lint_result,
    ]
    
    # Trivy runs only after YAML and cfn-lint succeed.
    if yaml_result["passed"] and cfn_lint_result["passed"]:
        trivy_result = validate_trivy(template)
    else:
        trivy_result = ValidationResult(
            stage="trivy",
            passed=False,
            errors=[],
            raw_output="Skipped: yaml/cfn-lint prerequisite validation failed",
        )

    results.append(trivy_result)
    static_passed = all(r["passed"] for r in results)

    if static_passed and deploy_config.target != DeployTarget.NONE:
        deploy_result = validate_deployment(template, deploy_config)
    else:
        deploy_result = DeployValidationResult(
            target="skipped",
            passed=True,
            stack_id=None,
            completed_resources=[],
            failed_resources=[],
            error_message=None if static_passed else "Skipped: static validation failed",
            duration_seconds=0.0,
            deployment_logs=[],
        )

    all_passed = static_passed and deploy_result["passed"]
    return results, all_passed, deploy_result
