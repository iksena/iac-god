import subprocess
import tempfile
import json
from pathlib import Path
from state import ValidationResult, DeployValidationResult
from yamllint import linter
from yamllint.config import YamlLintConfig
from config import DeployConfig, DeployTarget, DEFAULT_DEPLOY_CONFIG
from tools.deploy_validator import validate_deployment


def _derive_policy_rates(
    total_policies: int, passed_policies: int, filtered_failed_policies: int
) -> tuple[float, float]:
    if total_policies <= 0:
        return 1.0, 1.0
    ppr = passed_policies / total_policies
    fcr = (total_policies - filtered_failed_policies) / total_policies
    return ppr, fcr


# ---------------------------------------------------------------------------
# YAML lint config (CloudFormation only)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CloudFormation validators
# ---------------------------------------------------------------------------

def validate_yaml(template: str) -> ValidationResult:
    """Stage 1 (CFN): Basic YAML syntax and style check via yamllint."""
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

    Output format:
        [W3005] line 42 | Resource: MyBucket | <message> | <rule description>
    """
    rule    = finding.get("Rule") or {}
    rule_id = rule.get("Id") or "?"

    location = finding.get("Location") or {}
    start    = location.get("Start") or {}
    line_num = start.get("LineNumber")

    path = location.get("Path") or []
    resource = path[1] if len(path) > 1 else None

    message     = (finding.get("Message") or "").strip()
    description = (rule.get("Description") or "").strip()

    parts: list[str] = [f"[{rule_id}]"]
    if line_num is not None:
        parts.append(f"line {line_num}")
    if resource:
        parts.append(f"Resource: {resource}")
    if message:
        parts.append(message)
    if description and description.lower() != message.lower():
        parts.append(description)

    return " | ".join(parts)


def validate_cfn_lint(template: str) -> ValidationResult:
    """Stage 2 (CFN): AWS CloudFormation linting via cfn-lint."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(template)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["cfn-lint", tmp_path, "--format", "json"],
            capture_output=True, text=True, timeout=60,
        )
        raw = result.stdout or result.stderr

        try:
            findings = json.loads(raw)
            errors = [_format_cfn_lint_finding(f) for f in findings]
            # cfn-lint's own exit code defaults to --non-zero-exit-code=
            # informational, so it goes non-zero for ANY finding, including
            # plain Warning/Informational ones (e.g. W3002). Only Error-level
            # findings should actually fail the stage; Warning/Informational
            # are surfaced to the LLM but non-blocking, mirroring
            # validate_tflint's severity handling below.
            passed = not any(
                (finding.get("Level") or "").lower() == "error" for finding in findings
            )
        except json.JSONDecodeError:
            # Fallback: not valid JSON (e.g. a crash), trust the exit code.
            passed = result.returncode == 0
            errors = [] if passed else [raw]

        return ValidationResult(
            stage="cfn-lint",
            passed=passed,
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


# ---------------------------------------------------------------------------
# Terraform validators
# ---------------------------------------------------------------------------

def validate_tflint(template: str) -> ValidationResult:
    """Stage 1 (Terraform): HCL style/best-practice linting via tflint.

    Mirrors validate_yaml() (CFN Stage 1) in contract and error format.

    Writes main.tf to a temp directory, then runs:
        tflint --init --no-color          (downloads provider ruleset plugins)
        tflint --format json --no-color   (emits structured JSON findings)

    tflint JSON output shape:
        {
          "issues": [
            {
              "rule":    { "name": str, "severity": "error"|"warning"|"notice" },
              "message": str,
              "range":   { "start": { "line": int } }
            }
          ]
        }

    Severity policy (mirrors cfn-lint WARNING vs ERROR handling):
      - ERROR-severity issues  -> cause the stage to fail.
      - WARNING/NOTICE issues  -> reported in errors list but stage still passes
                                  (non-blocking), allowing terraform-validate to
                                  run and give the LLM a full picture.

    Returns a ValidationResult with stage="tflint".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"
        tf_path.write_text(template, encoding="utf-8")
        try:
            # Step 1: init — downloads the AWS provider ruleset plugin.
            # Non-fatal: plugin may already be cached or we may be offline.
            init_result = subprocess.run(
                ["tflint", "--init", "--no-color"],
                cwd=tmpdir,
                capture_output=True, text=True, timeout=120,
            )
            if init_result.returncode != 0:
                print(f"[tflint] init warning: {init_result.stderr.strip()}")

            # Step 2: lint — emit structured JSON findings.
            lint_result = subprocess.run(
                ["tflint", "--format", "json", "--no-color"],
                cwd=tmpdir,
                capture_output=True, text=True, timeout=60,
            )
            raw = lint_result.stdout or lint_result.stderr
            errors: list[str] = []
            error_severity_count = 0

            try:
                data = json.loads(raw)
                for issue in data.get("issues", []):
                    rule      = issue.get("rule", {})
                    rule_id   = rule.get("name", "?")
                    severity  = rule.get("severity", "error").lower()
                    message   = (issue.get("message") or "").strip()
                    line_num  = (
                        issue.get("range", {})
                             .get("start", {})
                             .get("line")
                    )

                    # Build structured error string mirroring cfn-lint format:
                    #   [rule_id] [SEVERITY] line N | message
                    parts: list[str] = [f"[{rule_id}]", f"[{severity.upper()}]"]
                    if line_num:
                        parts.append(f"line {line_num}")
                    if message:
                        parts.append(message)
                    errors.append(" | ".join(parts))

                    if severity == "error":
                        error_severity_count += 1

                # Stage fails only when there are ERROR-severity issues.
                # WARNING/NOTICE are surfaced to the LLM but are non-blocking,
                # consistent with cfn-lint's behaviour for W-prefixed rules.
                passed = error_severity_count == 0

            except (json.JSONDecodeError, KeyError):
                # Fallback: treat non-zero exit as failure with raw output.
                passed = lint_result.returncode == 0
                if not passed:
                    errors = [raw]

            return ValidationResult(
                stage="tflint",
                passed=passed,
                errors=errors,
                raw_output=raw,
            )
        except FileNotFoundError:
            return ValidationResult(
                stage="tflint", passed=False,
                errors=["tflint not installed. See: https://github.com/terraform-linters/tflint"],
                raw_output="TOOL_NOT_FOUND",
            )


def validate_terraform(template: str) -> ValidationResult:
    """Stage 2 (Terraform): structural/syntax validation via `terraform validate`.

    Writes the HCL to a temp directory as main.tf, runs:
        terraform init -backend=false -input=false
        terraform validate -json

    Returns a ValidationResult with stage="terraform-validate".
    The JSON output from `terraform validate` is parsed and each diagnostic
    is formatted as:
        [severity] line N | <summary> | <detail>
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"
        tf_path.write_text(template, encoding="utf-8")
        try:
            # Step 1: init (no backend, no input prompts)
            # Timeout increased to 600s (10 min) to allow provider plugin downloads
            init_result = subprocess.run(
                ["terraform", "init", "-backend=false", "-input=false", "-no-color"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if init_result.returncode != 0:
                err_text = (init_result.stderr or init_result.stdout).strip()
                return ValidationResult(
                    stage="terraform-validate",
                    passed=False,
                    errors=[f"terraform init failed: {err_text}"],
                    raw_output=err_text,
                )

            # Step 2: validate
            val_result = subprocess.run(
                ["terraform", "validate", "-json", "-no-color"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            raw = val_result.stdout or val_result.stderr
            errors: list[str] = []
            try:
                data = json.loads(raw)
                passed = bool(data.get("valid", False))
                for diag in data.get("diagnostics", []):
                    severity = diag.get("severity", "error").upper()
                    summary  = (diag.get("summary") or "").strip()
                    detail   = (diag.get("detail") or "").strip()
                    # Extract line number from range if present
                    rng      = (diag.get("range") or {}).get("start") or {}
                    line_num = rng.get("line")
                    parts: list[str] = [f"[{severity}]"]
                    if line_num:
                        parts.append(f"line {line_num}")
                    if summary:
                        parts.append(summary)
                    if detail and detail.lower() != summary.lower():
                        parts.append(detail)
                    errors.append(" | ".join(parts))
            except (json.JSONDecodeError, KeyError):
                passed = val_result.returncode == 0
                if not passed:
                    errors = [raw]

            return ValidationResult(
                stage="terraform-validate",
                passed=passed and not errors,
                errors=errors,
                raw_output=raw,
            )
        except FileNotFoundError:
            return ValidationResult(
                stage="terraform-validate",
                passed=False,
                errors=["terraform not installed. See: https://developer.hashicorp.com/terraform/install"],
                raw_output="TOOL_NOT_FOUND",
            )


# ---------------------------------------------------------------------------
# Shared security validators (both CFN and Terraform)
# ---------------------------------------------------------------------------

def validate_checkov(template: str, iac_type: str = "cloudformation") -> ValidationResult:
    """Security policy check via Checkov.

    Branches on iac_type:
      - cloudformation: saves as .yaml, runs --framework cloudformation
      - terraform:      saves as main.tf inside a temp dir, runs --framework terraform
    """
    if iac_type == "terraform":
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()
        tmp_path = str(Path(tmpdir) / "main.tf")
        Path(tmp_path).write_text(template, encoding="utf-8")
        scan_target = tmpdir
        framework = "terraform"
        cleanup = lambda: tmpdir_ctx.__exit__(None, None, None)  # noqa: E731
    else:
        f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
        f.write(template)
        f.close()
        tmp_path = f.name
        scan_target = tmp_path
        framework = "cloudformation"
        cleanup = lambda: Path(tmp_path).unlink(missing_ok=True)  # noqa: E731

    try:
        result = subprocess.run(
            [
                "checkov", "-f" if iac_type == "cloudformation" else "-d",
                scan_target,
                "--framework", framework,
                "--output", "json",
                "--quiet",
            ],
            capture_output=True, text=True, timeout=120,
        )
        raw = result.stdout or result.stderr
        errors: list[str] = []
        total_policies = passed_policies = failed_policies = filtered_failed_policies = 0
        try:
            data = json.loads(raw)
            # Checkov may return a list (one entry per framework) or a single dict.
            results_node = data if isinstance(data, dict) else (data[0] if data else {})
            failed = results_node.get("results", {}).get("failed_checks", [])
            passed = results_node.get("results", {}).get("passed_checks", [])
            failed_policies = len(failed)
            passed_policies = len(passed)
            total_policies = failed_policies + passed_policies

            for check in failed:
                severity = str(check.get("severity", "")).lower()
                if severity in ("high", "critical"):
                    filtered_failed_policies += 1

            errors = [
                f"[{c['check_id']}] {c['check_result']['result']}: "
                f"{c['resource']} \u2014 {c['check'].get('name', '')}"
                for c in failed
            ]
        except (json.JSONDecodeError, KeyError, IndexError):
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
        cleanup()


def validate_trivy(template: str, iac_type: str = "cloudformation") -> ValidationResult:
    """Misconfiguration scan via Trivy.

    Branches on iac_type:
      - cloudformation: saves as template.yaml
      - terraform:      saves as main.tf
    Both use `trivy config` against the temp directory.
    """
    suffix = ".tf" if iac_type == "terraform" else ".yaml"
    filename = "main.tf" if iac_type == "terraform" else "template.yaml"

    with tempfile.TemporaryDirectory() as tmpdir:
        scan_path = Path(tmpdir) / filename
        scan_path.write_text(template, encoding="utf-8")
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
            errors: list[str] = []
            total_policies = passed_policies = failed_policies = filtered_failed_policies = 0
            try:
                data = json.loads(raw)
                for r in data.get("Results", []):
                    summary = r.get("MisconfSummary") or {}
                    passed_policies  += int(summary.get("Successes", 0) or 0)
                    failed_policies  += int(summary.get("Failures",  0) or 0)

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


# ---------------------------------------------------------------------------
# Orchestrator: run the correct pipeline based on iac_type
# ---------------------------------------------------------------------------

def run_all_validators(
    template: str,
    iac_type: str = "cloudformation",
    deploy_config: DeployConfig = DEFAULT_DEPLOY_CONFIG,
) -> tuple[list[ValidationResult], bool, DeployValidationResult]:
    """
    Run the correct validation pipeline for the given IaC type.

    CloudFormation pipeline (two structural stages):
        yaml  ->  cfn-lint  ->  trivy  ->  (deploy)

    Terraform pipeline (two structural stages, symmetric with CFN):
        tflint  ->  terraform-validate  ->  trivy  ->  (deploy)

    Stage symmetry is intentional: both pipelines have an identical number of
    structural stages so that the multi-agent repair loop (graph.py, state.py
    routing, retriever error-type detection) behaves identically for both IaC
    types. This structural parity is a requirement of the generalisation
    research hypothesis.

    Skipped stages (trivy when structural stages fail, deploy when static
    validation fails) are represented with passed=True and an empty errors
    list so that classify_failing_stages() does not count them as failures.
    A skipped stage is not a failed stage — it simply did not run.

    Checkov is wired but currently skipped in both pipelines (kept for future
    re-enablement); trivy covers the security stage.

    Returns (static_results, all_passed, deploy_result).
    """
    if iac_type == "terraform":
        # ----------------------------------------------------------------
        # Terraform pipeline
        # Stage 1: tflint        (HCL style / best-practice linting)
        # Stage 2: terraform-validate  (structural / provider schema check)
        # Stage 3: trivy         (security misconfigurations)
        # Stage 4: deploy        (terraform apply against LocalStack)
        #
        # trivy runs only when BOTH structural stages pass, mirroring the
        # CFN behaviour where trivy is skipped until yaml+cfn-lint succeed.
        # ----------------------------------------------------------------
        tflint_result      = validate_tflint(template)
        tf_validate_result = validate_terraform(template)
        results: list[ValidationResult] = [tflint_result, tf_validate_result]

        if tflint_result["passed"] and tf_validate_result["passed"]:
            trivy_result = validate_trivy(template, iac_type="terraform")
        else:
            # passed=True: a skipped stage is not a failed stage.
            # The empty errors list means classify_failing_stages() will not
            # add "security" to failing_stages, preventing spurious routing.
            trivy_result = ValidationResult(
                stage="trivy",
                passed=True,
                errors=[],
                raw_output="Skipped: tflint/terraform-validate prerequisite failed",
            )
        results.append(trivy_result)

        static_passed = all(r["passed"] for r in results)

        if static_passed and deploy_config.target != DeployTarget.NONE:
            deploy_result = validate_deployment(
                template,
                deploy_config=deploy_config,
                iac_type="terraform",
            )
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

    else:
        # ----------------------------------------------------------------
        # CloudFormation pipeline (original behaviour)
        # ----------------------------------------------------------------
        yaml_result = validate_yaml(template)
        cfn_lint_result = validate_cfn_lint(template)
        results = [yaml_result, cfn_lint_result]

        # Trivy runs only after YAML and cfn-lint succeed
        if yaml_result["passed"] and cfn_lint_result["passed"]:
            trivy_result = validate_trivy(template, iac_type="cloudformation")
        else:
            # passed=True: a skipped stage is not a failed stage.
            trivy_result = ValidationResult(
                stage="trivy",
                passed=True,
                errors=[],
                raw_output="Skipped: yaml/cfn-lint prerequisite validation failed",
            )
        results.append(trivy_result)

        static_passed = all(r["passed"] for r in results)

        if static_passed and deploy_config.target != DeployTarget.NONE:
            deploy_result = validate_deployment(
                template,
                deploy_config=deploy_config,
                iac_type="cloudformation",
            )
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
