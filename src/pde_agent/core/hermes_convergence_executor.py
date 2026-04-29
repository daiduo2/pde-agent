"""Controlled action executor for Hermes convergence plans."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_submission_candidate import run_hermes_submission_candidate


SubmissionCandidateRunner = Callable[..., dict[str, Any]]


class HermesConvergenceExecutorError(RuntimeError):
    pass


def hermes_controlled_action_executor(
    *,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    submission_id: str,
    target_path: str | Path | None = None,
    submission_candidate_runner: SubmissionCandidateRunner | None = None,
) -> Callable[[dict[str, Any], Path], Path | None]:
    runner = submission_candidate_runner or run_hermes_submission_candidate

    def execute(plan: dict[str, Any], workspace_dir: Path) -> Path | None:
        command = plan.get("command")
        if command != "predict-unet-competition":
            return None
        arguments = plan.get("arguments")
        if not isinstance(arguments, dict):
            raise HermesConvergenceExecutorError("plan arguments must be an object")
        source = arguments.get("source_artifact_path")
        if not isinstance(source, str) or not source:
            raise HermesConvergenceExecutorError(
                "predict-unet-competition plan missing source_artifact_path"
            )
        report = runner(
            tuning_report_path=Path(source),
            output_dir=workspace_dir / "submission_candidate",
            task_id=task_id,
            test_path=Path(test_path),
            checkpoint_path=Path(checkpoint_path),
            submission_id=submission_id,
            target_path=Path(target_path) if target_path is not None else None,
            force=True,
        )
        report_path = Path(report["candidate_report_path"])
        _ensure_report_file(report_path=report_path, report=report)
        return report_path

    setattr(execute, "executes_terminal_actions", True)
    return execute


def _ensure_report_file(*, report_path: Path, report: dict[str, Any]) -> None:
    try:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, dict) and existing.get("next_command"):
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
