"""Run one complete FNO candidate iteration on an existing local window file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.candidate_execution_prep import prepare_candidate_execution_from_audit
from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file
from pde_agent.core.fno_candidate_comparison import compare_fno_candidates
from pde_agent.core.fno_candidate_evaluation import evaluate_fno_candidate
from pde_agent.core.fno_finetune_candidate import execute_fno_finetune_candidate
from pde_agent.core.fno_iteration_dryrun import run_fno_iteration_dryrun


class FNOIterationRunnerError(RuntimeError):
    pass


def run_fno_iteration(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    max_train_steps: int,
    learning_rate: float,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FNOIterationRunnerError(
            f"output_dir already contains files: {output_root}"
        )

    dryrun = run_fno_iteration_dryrun(
        windows_path=windows_file,
        checkpoint_path=checkpoint_file,
        output_dir=output_root,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    audit_path = output_root / "experiment_config_audit.json"
    audit = audit_experiment_config_from_file(
        config_path=output_root / "experiment_config.json",
        output_path=audit_path,
    )
    if audit["decision"] != "approved_for_execution":
        raise FNOIterationRunnerError(
            f"experiment config audit rejected execution: {audit['decision']}"
        )

    candidate_count = _candidate_count(audit)
    evaluation_paths: list[Path] = []
    candidates: list[dict[str, Any]] = []
    for candidate_index in range(candidate_count):
        plan = prepare_candidate_execution_from_audit(
            audit_path=audit_path,
            output_dir=output_root / "candidates",
            candidate_index=candidate_index,
            force=force,
        )
        execution = execute_fno_finetune_candidate(
            candidate_plan_path=plan["workspace"]["candidate_execution_plan_path"],
            windows_path=windows_file,
            checkpoint_path=checkpoint_file,
            max_train_steps=max_train_steps,
            learning_rate=learning_rate,
            force=force,
        )
        eval_dir = Path(plan["workspace"]["artifacts_dir"]) / "evaluation"
        evaluation = evaluate_fno_candidate(
            candidate_report_path=execution["artifacts"][
                "candidate_execution_report_path"
            ],
            target_path=windows_file,
            output_dir=eval_dir,
            baseline_score_path=output_root / "fno_score.json",
            nu=nu,
            dt=dt,
            dx=dx,
            force=force,
        )
        eval_path = eval_dir / "candidate_evaluation_report.json"
        evaluation_paths.append(eval_path)
        candidates.append(
            _candidate_summary(
                plan=plan,
                execution=execution,
                evaluation=evaluation,
                evaluation_report_path=eval_path,
            )
        )

    comparison_path = output_root / "candidate_comparison.json"
    comparison = compare_fno_candidates(
        evaluation_report_paths=evaluation_paths,
        output_path=comparison_path,
        force=force,
    )
    report_path = output_root / "fno_iteration_report.json"
    report = {
        "schema_version": 1,
        "state_hint": "FNO_ITERATION_COMPLETED",
        "inputs": {
            "windows_path": str(windows_file),
            "checkpoint_path": str(checkpoint_file),
            "max_train_steps": max_train_steps,
            "learning_rate": learning_rate,
            "nu": nu,
            "dt": dt,
            "dx": dx,
        },
        "artifacts": {
            "iteration_manifest_path": dryrun["artifacts"][
                "iteration_manifest_path"
            ],
            "experiment_config_path": dryrun["artifacts"]["experiment_config_path"],
            "experiment_config_audit_path": str(audit_path),
            "comparison_report_path": str(comparison_path),
            "iteration_report_path": str(report_path),
        },
        "baseline": {
            "competition_like_score_short": dryrun["metrics"][
                "competition_like_score_short"
            ],
            "selected_hypothesis": dryrun["metrics"]["selected_hypothesis"],
            "selected_intervention": dryrun["metrics"]["selected_intervention"],
        },
        "audit": {
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
        },
        "candidate_count": candidate_count,
        "candidates": candidates,
        "winner": comparison["winner"],
        "comparison": {
            "candidate_count": comparison["candidate_count"],
            "comparison_report_path": str(comparison_path),
        },
        "next_command": "run-fno-iteration",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _candidate_count(audit: dict[str, Any]) -> int:
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        raise FNOIterationRunnerError("audit report must include summary")
    count = summary.get("candidate_count")
    if not isinstance(count, int) or count <= 0:
        raise FNOIterationRunnerError("audit report must include positive candidate_count")
    return count


def _candidate_summary(
    *,
    plan: dict[str, Any],
    execution: dict[str, Any],
    evaluation: dict[str, Any],
    evaluation_report_path: Path,
) -> dict[str, Any]:
    candidate = plan["candidate"]
    parameters = candidate["parameters"]
    loss_trace = execution["train"]["loss_trace"]
    return {
        "candidate_index": plan["candidate_index"],
        "experiment_id": candidate["experiment_id"],
        "intervention": candidate["intervention"],
        "parameters": parameters,
        "train_steps": execution["train"]["train_steps"],
        "training_strategy": execution["train"]["training_strategy"],
        "initial_loss": loss_trace[0] if loss_trace else None,
        "final_loss": loss_trace[-1] if loss_trace else None,
        "score": evaluation["score"]["competition_like_score_short"],
        "score_delta": evaluation["baseline_comparison"]["score_delta"],
        "rel_mse": evaluation["score"]["rel_mse"],
        "rmse": evaluation["score"]["rmse"],
        "dominant_error_pattern": evaluation["error_diagnosis"][
            "dominant_error_pattern"
        ],
        "physics_issue": evaluation["physics"]["physics_issue"],
        "candidate_execution_report_path": execution["artifacts"][
            "candidate_execution_report_path"
        ],
        "evaluation_report_path": str(evaluation_report_path),
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOIterationRunnerError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOIterationRunnerError(f"{label} path is not a file: {path}")
    return path
