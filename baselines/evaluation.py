"""Scenario evaluation state machine shared by the MCP server and the runner.

This module is the *instrumentation* half of the harness baseline. The harness
(Claude Code) owns the generate -> validate -> repair loop; everything that
makes a run measurable lives here:

  - the validators           (tools.validators.run_all_validators, unmodified)
  - the deploy stage         (tools.deploy_validator.validate_deployment)
  - the error text the model sees
  - the iteration counter and its cap
  - the per-iteration research artifacts (tracking.recorder.ResearchRecorder)

Comparability requirements, and how they are met:

  * Same validators. run_all_validators() is the exact function
    agents/validator.py calls, so stage order, skip semantics and policy_stats
    are identical between the two systems.

  * Same error text. build_validation_error_text is agents.engineer's own
    formatter, imported rather than reimplemented, so the harness reads
    character-identical errors to what IaCGOD's Engineer reads in simple mode.
    This is a deliberate coupling to a private name: a divergence here would
    silently confound the comparison.

  * Same iteration semantics. One validate_iac call == one iteration, matching
    agents/validator.py, which increments once per generate -> validate cycle.
    So iterations_used == 1 means the same thing in both systems and pass@1
    stays comparable. deploy_iac attaches to the current iteration instead of
    opening a new one, so it cannot inflate the count.

  * Same artifacts. ResearchRecorder writes runs/<run_id>/iteration_NNN.json
    and deployment_log_NNN.txt exactly as a multi-agent run does, so the
    existing aggregation scripts read baseline runs without modification.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DeployConfig, DeployTarget
from state import DeployValidationResult
from tools.deploy_validator import validate_deployment
from tools.validators import run_all_validators
from tracking.recorder import ResearchRecorder

# Imported, not reimplemented: the harness must see byte-identical error text
# to IaCGOD's Engineer. See module docstring.
from agents.engineer import _build_simple_fix_errors as build_validation_error_text


# ---------------------------------------------------------------------------
# Environment contract between the runner and the MCP server subprocess
# ---------------------------------------------------------------------------
# The MCP server is spawned by the harness, not by the runner, so scenario
# context has to travel through the generated .mcp.json env block.

ENV_RUN_ID = "IACGOD_RUN_ID"
ENV_IAC_TYPE = "IACGOD_IAC_TYPE"
ENV_WORKDIR = "IACGOD_WORKDIR"
ENV_RUNS_DIR = "IACGOD_RUNS_DIR"
ENV_MAX_ITERATIONS = "IACGOD_MAX_ITERATIONS"
ENV_DEPLOY_TARGET = "IACGOD_DEPLOY_TARGET"
ENV_USER_REQUEST = "IACGOD_USER_REQUEST"

STATE_FILENAME = "baseline_state.json"

TEMPLATE_FILENAME = {
    "cloudformation": "template.yaml",
    "terraform": "main.tf",
}


def template_filename(iac_type: str) -> str:
    return TEMPLATE_FILENAME.get(iac_type, "template.yaml")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tool outcomes
# ---------------------------------------------------------------------------


@dataclass
class ToolOutcome:
    """What a tool call returns to the harness, plus what we record about it."""

    text: str
    passed: bool
    counted_iteration: int | None = None


@dataclass
class ScenarioConfig:
    run_id: str
    iac_type: str
    workdir: Path
    runs_dir: Path
    max_iterations: int
    deploy_target: str
    user_request: str = ""

    @classmethod
    def from_env(cls) -> "ScenarioConfig":
        missing = [
            k
            for k in (ENV_RUN_ID, ENV_IAC_TYPE, ENV_WORKDIR, ENV_MAX_ITERATIONS)
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(
                "MCP server started without scenario context; missing env: "
                + ", ".join(missing)
            )
        return cls(
            run_id=os.environ[ENV_RUN_ID],
            iac_type=os.environ[ENV_IAC_TYPE],
            workdir=Path(os.environ[ENV_WORKDIR]),
            runs_dir=Path(os.environ.get(ENV_RUNS_DIR, "runs")),
            max_iterations=int(os.environ[ENV_MAX_ITERATIONS]),
            deploy_target=os.environ.get(ENV_DEPLOY_TARGET, "none"),
            user_request=os.environ.get(ENV_USER_REQUEST, ""),
        )

    def to_env(self) -> dict[str, str]:
        return {
            ENV_RUN_ID: self.run_id,
            ENV_IAC_TYPE: self.iac_type,
            ENV_WORKDIR: str(self.workdir),
            ENV_RUNS_DIR: str(self.runs_dir),
            ENV_MAX_ITERATIONS: str(self.max_iterations),
            ENV_DEPLOY_TARGET: self.deploy_target,
            ENV_USER_REQUEST: self.user_request,
        }


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass
class ScenarioLedger:
    """Owns the iteration counter, the validators and the run artifacts.

    Lives inside the MCP server process for the duration of one scenario. Its
    persisted state (runs/<run_id>/baseline_state.json) is the handoff back to
    the runner once the harness process exits.
    """

    config: ScenarioConfig
    recorder: ResearchRecorder = field(init=False)

    iteration: int = 0
    iterations: list[dict[str, Any]] = field(default_factory=list)
    cap_reached: bool = False

    # Content hash of the template at its last validate_iac call, with that
    # call's results. deploy_iac is gated on this so the harness cannot deploy
    # a template that has not passed static validation - mirroring
    # run_all_validators, which skips deploy when static fails.
    last_validated_sha: str | None = None
    last_static_passed: bool = False
    last_validation_results: list[dict[str, Any]] = field(default_factory=list)
    last_deploy_result: dict[str, Any] | None = None

    submitted: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.recorder = ResearchRecorder(
            run_id=self.config.run_id,
            output_dir=str(self.config.runs_dir),
        )
        self._persist()

    # -- helpers ---------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.recorder.output_dir / STATE_FILENAME

    def _read_template(self, file_path: str) -> tuple[str | None, str]:
        """Return (content, error_text). content is None when unreadable."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.config.workdir / path
        if not path.exists():
            return None, (
                f"No file at {path}. Write the "
                f"{template_filename(self.config.iac_type)} first, then validate it."
            )
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"Could not read {path}: {exc}"
        if not content.strip():
            return None, f"{path} is empty. Write the template before validating."
        return content, ""

    def _snapshot(self, template: str) -> None:
        """Write iteration_NNN.json in the same shape a multi-agent run does.

        objectives and remediation_history are empty by construction: the
        baseline has no Planner and no Remediator. That absence is the
        experimental condition, not missing data.

        validation_passed means fully validated, exactly as it does in a
        multi-agent snapshot: static clean AND deployed, whenever deployment is
        part of the run. A snapshot written between a passing validate_iac and
        its deploy_iac must not claim success the deploy has not conferred —
        deploy_iac re-snapshots the same iteration once it has a verdict.
        """
        if self.config.deploy_target == "none":
            fully_validated = self.last_static_passed
        else:
            fully_validated = self.last_static_passed and bool(
                self.last_deploy_result and self.last_deploy_result.get("passed")
            )

        self.recorder.save_iteration_snapshot(
            {
                "current_iteration": self.iteration,
                "objectives": [],
                "iac_template": template,
                "validation_results": self.last_validation_results,
                "validation_passed": bool(fully_validated),
                "deploy_validation_result": self.last_deploy_result,
                "remediation_history": [],
            }
        )

    def _persist(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "run_id": self.config.run_id,
                    "iac_type": self.config.iac_type,
                    "workdir": str(self.config.workdir),
                    "deploy_target": self.config.deploy_target,
                    "max_iterations": self.config.max_iterations,
                    "iteration_count": self.iteration,
                    "cap_reached": self.cap_reached,
                    "iterations": self.iterations,
                    "last_validated_sha": self.last_validated_sha,
                    "last_static_passed": self.last_static_passed,
                    "last_validation_results": self.last_validation_results,
                    "last_deploy_result": self.last_deploy_result,
                    "submitted": self.submitted,
                    "updated_at": _now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- tools -----------------------------------------------------------

    def validate(self, file_path: str) -> ToolOutcome:
        """Static validation. One successful call == one iteration."""
        # Cap is checked BEFORE incrementing: a refused call is not an
        # iteration, and must not inflate iterations_used.
        if self.iteration >= self.config.max_iterations:
            self.cap_reached = True
            self._persist()
            return ToolOutcome(
                text=(
                    f"ITERATION CAP REACHED ({self.config.max_iterations} iterations used).\n"
                    "No further validation runs are available. Stop editing and call "
                    "submit_template now with your best template."
                ),
                passed=False,
            )

        template, read_error = self._read_template(file_path)
        if template is None:
            # A read failure is a harness mistake, not a generate->validate
            # cycle, so it is not counted either.
            return ToolOutcome(text=read_error, passed=False)

        results, _all_passed, _deploy = run_all_validators(
            template,
            iac_type=self.config.iac_type,
            deploy_config=DeployConfig(target=DeployTarget.NONE),
        )
        static_passed = all(r["passed"] for r in results)

        self.iteration += 1
        self.last_validated_sha = _sha(template)
        self.last_static_passed = static_passed
        self.last_validation_results = results
        # A fresh edit invalidates any previous deploy verdict.
        self.last_deploy_result = None

        self.iterations.append(
            {
                "iteration": self.iteration,
                "template_sha": self.last_validated_sha,
                "template_lines": len(template.splitlines()),
                "static_passed": static_passed,
                "stage_errors": {
                    r["stage"]: len(r.get("errors", [])) for r in results
                },
                "deploy_attempted": False,
                "deploy_passed": None,
                "timestamp": _now(),
            }
        )

        self._snapshot(template)
        self._persist()

        if static_passed:
            remaining = self.config.max_iterations - self.iteration
            return ToolOutcome(
                text=(
                    f"[iteration {self.iteration}] STATIC VALIDATION PASSED.\n"
                    f"All stages clean. Now call deploy_iac with the same file path "
                    f"to verify it actually deploys. ({remaining} iterations remaining.)"
                ),
                passed=True,
                counted_iteration=self.iteration,
            )

        errors = build_validation_error_text(
            {"validation_results": results, "deploy_validation_result": None}
        )
        return ToolOutcome(
            text=(
                f"[iteration {self.iteration}] STATIC VALIDATION FAILED.\n\n"
                f"{errors}\n\n"
                "Fix every error above in the template, then call validate_iac again. "
                "Do not suppress or comment out any check."
            ),
            passed=False,
            counted_iteration=self.iteration,
        )

    def deploy(self, file_path: str) -> ToolOutcome:
        """Live deploy. Attaches to the current iteration; never opens one."""
        if self.config.deploy_target == "none":
            return ToolOutcome(
                text=(
                    "Deployment is disabled for this run. Static validation is the "
                    "only gate — call submit_template once validate_iac passes."
                ),
                passed=True,
            )

        template, read_error = self._read_template(file_path)
        if template is None:
            return ToolOutcome(text=read_error, passed=False)

        current_sha = _sha(template)
        if current_sha != self.last_validated_sha:
            return ToolOutcome(
                text=(
                    "This template has changed since its last validate_iac call. "
                    "Call validate_iac first — deployment only runs on a template "
                    "that has passed static validation."
                ),
                passed=False,
            )
        if not self.last_static_passed:
            return ToolOutcome(
                text=(
                    "Static validation has not passed for this template. Fix the "
                    "reported errors and call validate_iac until it passes before "
                    "attempting a deployment."
                ),
                passed=False,
            )
        if self.last_deploy_result is not None:
            prior = "passed" if self.last_deploy_result.get("passed") else "failed"
            return ToolOutcome(
                text=(
                    f"This exact template was already deployed and the attempt {prior}. "
                    "Edit the template and re-run validate_iac before deploying again."
                ),
                passed=bool(self.last_deploy_result.get("passed")),
            )

        deploy_result: DeployValidationResult = validate_deployment(
            template,
            deploy_config=DeployConfig(target=DeployTarget(self.config.deploy_target)),
            iac_type=self.config.iac_type,
        )
        self.last_deploy_result = dict(deploy_result)

        self.recorder.record_deployment_log(
            iteration=self.iteration,
            iac_type=self.config.iac_type,
            target=deploy_result.get("target", self.config.deploy_target),
            deployment_logs=deploy_result.get("deployment_logs", []),
            passed=bool(deploy_result.get("passed")),
            duration_seconds=float(deploy_result.get("duration_seconds", 0.0) or 0.0),
            failed_resources=deploy_result.get("failed_resources", []),
        )

        if self.iterations:
            self.iterations[-1]["deploy_attempted"] = True
            self.iterations[-1]["deploy_passed"] = bool(deploy_result.get("passed"))

        self._snapshot(template)
        self._persist()

        if deploy_result.get("passed"):
            return ToolOutcome(
                text=(
                    f"[iteration {self.iteration}] DEPLOYMENT PASSED against "
                    f"{deploy_result.get('target')}.\n"
                    "The template is fully validated. Call submit_template and stop."
                ),
                passed=True,
            )

        errors = build_validation_error_text(
            {
                "validation_results": self.last_validation_results,
                "deploy_validation_result": self.last_deploy_result,
            }
        )
        return ToolOutcome(
            text=(
                f"[iteration {self.iteration}] DEPLOYMENT FAILED.\n\n"
                f"{errors}\n\n"
                "Fix the template so these resources deploy, then call validate_iac "
                "again followed by deploy_iac."
            ),
            passed=False,
        )

    def submit(self, file_path: str) -> ToolOutcome:
        """Record the harness's final answer.

        This is a claim, not a verdict. The runner recomputes the outcome from
        the recorded validation results, re-validating if the submitted content
        differs from what was last validated — a harness may submit a template
        it never got to pass.
        """
        template, read_error = self._read_template(file_path)
        if template is None:
            return ToolOutcome(text=read_error, passed=False)

        self.submitted = {
            "file_path": str(file_path),
            "sha": _sha(template),
            "matches_last_validated": _sha(template) == self.last_validated_sha,
            "timestamp": _now(),
        }
        self._persist()
        return ToolOutcome(
            text=(
                "Template submitted and recorded. Your work on this scenario is "
                "complete — stop here and do not make further tool calls."
            ),
            passed=True,
        )
