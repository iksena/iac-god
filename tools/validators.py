# tools/validators.py
import subprocess
import tempfile
import json
import yaml
from pathlib import Path
from state import ValidationResult

def validate_yaml(template: str) -> ValidationResult:
    """Stage 1: Basic YAML syntax check."""
    try:
        yaml.safe_load(template)
        return ValidationResult(
            stage="yaml", passed=True, errors=[], raw_output="YAML syntax OK"
        )
    except yaml.YAMLError as e:
        return ValidationResult(
            stage="yaml", passed=False, errors=[str(e)], raw_output=str(e)
        )

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
                errors = [
                    f"[{f.get('Rule',{}).get('Id','?')}] "
                    f"{f.get('Location',{}).get('Start',{})}: "
                    f"{f.get('Message','')}"
                    for f in findings
                ]
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
        try:
            data = json.loads(raw)
            failed = data.get("results", {}).get("failed_checks", [])
            errors = [
                f"[{c['check_id']}] {c['check_result']['result']}: "
                f"{c['resource']} — {c['check'].get('name','')}"
                for c in failed
            ]
        except (json.JSONDecodeError, KeyError):
            if result.returncode not in (0, 1):
                errors = [raw]

        return ValidationResult(
            stage="checkov",
            passed=len(errors) == 0,
            errors=errors,
            raw_output=raw,
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
            try:
                data = json.loads(raw)
                for r in data.get("Results", []):
                    for m in r.get("Misconfigurations", []):
                        errors.append(
                            f"[{m['ID']}] {m['Severity']}: {m['Title']} — {m['Message']}"
                        )
            except (json.JSONDecodeError, KeyError):
                if result.returncode not in (0, 1):
                    errors = [raw]

            return ValidationResult(
                stage="trivy",
                passed=len(errors) == 0,
                errors=errors,
                raw_output=raw,
            )
        except FileNotFoundError:
            return ValidationResult(
                stage="trivy", passed=False,
                errors=["trivy not installed. See: https://aquasecurity.github.io/trivy"],
                raw_output="TOOL_NOT_FOUND",
            )

def run_all_validators(template: str) -> tuple[list[ValidationResult], bool]:
    """Run all 4 validation stages. Returns (results, all_passed)."""
    results = [
        validate_yaml(template),
        validate_cfn_lint(template),
        validate_checkov(template),
        validate_trivy(template),
    ]
    all_passed = all(r["passed"] for r in results)
    return results, all_passed