"""Run bounded FNO evidence sweeps across windows and train-step budgets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from pde_agent.core.candidate_execution_prep import prepare_candidate_execution_from_audit
from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file
from pde_agent.core.fno_candidate_comparison import compare_fno_candidates
from pde_agent.core.fno_candidate_evaluation import evaluate_fno_candidate
from pde_agent.core.fno_finetune_candidate import execute_fno_finetune_candidate
from pde_agent.core.fno_iteration_dryrun import run_fno_iteration_dryrun
from pde_agent.core.local_window_suite import build_local_window_suite


class FNOEvidenceSweepError(RuntimeError):
    pass


_CSV_FIELDS = [
    "start_index",
    "train_steps",
    "experiment_id",
    "training_strategy",
    "rollout_steps",
    "teacher_forcing_decay",
    "baseline_score",
    "score",
    "score_delta",
    "rel_mse",
    "rmse",
    "dominant_error_pattern",
    "physics_issue",
    "residual_rmse",
    "mass_drift",
    "energy_growth_ratio",
    "high_frequency_energy_ratio",
    "initial_loss",
    "final_loss",
]


def run_fno_evidence_sweep(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_indices: list[int],
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: list[int],
    learning_rate: float,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    source_file = _require_file(Path(source_path), "source HDF5")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_root = Path(output_dir)
    starts = _validate_unique_positive_ints(
        start_indices,
        label="start_indices",
        allow_zero=True,
    )
    steps = _validate_unique_positive_ints(
        train_steps,
        label="train_steps",
        allow_zero=False,
    )
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FNOEvidenceSweepError(f"output_dir already contains files: {output_root}")

    records: list[dict[str, Any]] = []
    comparison_winners: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []

    for start_index in starts:
        window_path = _build_window(
            source_file=source_file,
            output_root=output_root,
            dataset_key=dataset_key,
            start_index=start_index,
            sample_limit=sample_limit,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            force=force,
        )
        baseline_dir = output_root / "baselines" / f"start_{start_index:04d}"
        baseline_manifest = run_fno_iteration_dryrun(
            windows_path=window_path,
            checkpoint_path=checkpoint_file,
            output_dir=baseline_dir,
            nu=nu,
            dt=dt,
            dx=dx,
            force=force,
        )
        audit_path = baseline_dir / "experiment_config_audit.json"
        audit_experiment_config_from_file(
            config_path=baseline_dir / "experiment_config.json",
            output_path=audit_path,
        )
        baseline_score = baseline_manifest["metrics"]["competition_like_score_short"]
        baseline_records.append(
            {
                "start_index": start_index,
                "windows_path": str(window_path),
                "baseline_dir": str(baseline_dir),
                "baseline_score": baseline_score,
                "selected_intervention": baseline_manifest["metrics"][
                    "selected_intervention"
                ],
            }
        )

        for train_step_count in steps:
            cell_root = (
                output_root
                / "sweeps"
                / f"start_{start_index:04d}"
                / f"steps_{train_step_count:03d}"
            )
            eval_paths = []
            for candidate_index in (0, 1):
                plan = prepare_candidate_execution_from_audit(
                    audit_path=audit_path,
                    output_dir=cell_root / "candidates",
                    candidate_index=candidate_index,
                    force=force,
                )
                execution = execute_fno_finetune_candidate(
                    candidate_plan_path=plan["workspace"][
                        "candidate_execution_plan_path"
                    ],
                    windows_path=window_path,
                    checkpoint_path=checkpoint_file,
                    max_train_steps=train_step_count,
                    learning_rate=learning_rate,
                    force=force,
                )
                eval_dir = Path(plan["workspace"]["artifacts_dir"]) / "evaluation"
                evaluation = evaluate_fno_candidate(
                    candidate_report_path=execution["artifacts"][
                        "candidate_execution_report_path"
                    ],
                    target_path=window_path,
                    output_dir=eval_dir,
                    baseline_score_path=baseline_dir / "fno_score.json",
                    nu=nu,
                    dt=dt,
                    dx=dx,
                    force=force,
                )
                eval_path = eval_dir / "candidate_evaluation_report.json"
                eval_paths.append(eval_path)
                records.append(
                    _record(
                        start_index=start_index,
                        train_steps=train_step_count,
                        plan=plan,
                        execution=execution,
                        evaluation=evaluation,
                    )
                )

            comparison_path = cell_root / "candidate_comparison.json"
            comparison = compare_fno_candidates(
                evaluation_report_paths=eval_paths,
                output_path=comparison_path,
                force=force,
            )
            comparison_winners.append(
                {
                    "start_index": start_index,
                    "train_steps": train_step_count,
                    "comparison_report_path": str(comparison_path),
                    "winner_experiment_id": comparison["winner"]["experiment_id"],
                    "winner_training_strategy": _strategy_for_experiment(
                        records=records,
                        start_index=start_index,
                        train_steps=train_step_count,
                        experiment_id=comparison["winner"]["experiment_id"],
                    ),
                    "winner_score": comparison["winner"]["score"],
                    "winner_score_delta": comparison["winner"]["score_delta"],
                }
            )

    summary_path = output_root / "evidence_summary.json"
    csv_path = output_root / "evidence_summary.csv"
    recommendation = _recommendation(
        records=records,
        comparison_winners=comparison_winners,
    )
    summary = {
        "schema_version": 1,
        "state_hint": "FNO_EVIDENCE_SWEEP_COMPLETED",
        "inputs": {
            "source_path": str(source_file),
            "checkpoint_path": str(checkpoint_file),
            "dataset_key": dataset_key,
            "sample_limit": sample_limit,
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
            "time_reduction": time_reduction,
            "space_reduction": space_reduction,
            "start_indices": starts,
            "train_steps": steps,
            "learning_rate": learning_rate,
        },
        "artifacts": {
            "summary_json_path": str(summary_path),
            "summary_csv_path": str(csv_path),
        },
        "baseline_records": baseline_records,
        "records": records,
        "record_count": len(records),
        "comparison_winners": comparison_winners,
        "comparison_count": len(comparison_winners),
        "strategy_statistics": _strategy_statistics(records),
        "recommendation": recommendation,
        "next_command": recommendation["next_command"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, records)
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_window(
    *,
    source_file: Path,
    output_root: Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    force: bool,
) -> Path:
    manifest = build_local_window_suite(
        source_path=source_file,
        output_dir=output_root / "windows" / f"start_{start_index:04d}",
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=[forecast_steps],
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    return Path(manifest["profiles"][0]["windows_npz_path"])


def _record(
    *,
    start_index: int,
    train_steps: int,
    plan: dict[str, Any],
    execution: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    parameters = plan["candidate"]["parameters"]
    loss_trace = execution["train"]["loss_trace"]
    record = {
        "start_index": start_index,
        "train_steps": train_steps,
        "experiment_id": plan["candidate"]["experiment_id"],
        "training_strategy": parameters.get("training_strategy", "scheduled_sampling"),
        "rollout_steps": parameters["rollout_steps"],
        "teacher_forcing_decay": parameters["teacher_forcing_decay"],
        "baseline_score": evaluation["baseline_comparison"]["baseline_score"],
        "score": evaluation["score"]["competition_like_score_short"],
        "score_delta": evaluation["baseline_comparison"]["score_delta"],
        "rel_mse": evaluation["score"]["rel_mse"],
        "rmse": evaluation["score"]["rmse"],
        "dominant_error_pattern": evaluation["error_diagnosis"][
            "dominant_error_pattern"
        ],
        "physics_issue": evaluation["physics"]["physics_issue"],
        "initial_loss": loss_trace[0] if loss_trace else None,
        "final_loss": loss_trace[-1] if loss_trace else None,
    }
    physics_residual = evaluation.get("physics_residual")
    if isinstance(physics_residual, dict):
        record.update(
            {
                "residual_rmse": physics_residual["residual_rmse"],
                "mass_drift": physics_residual["mass_drift"],
                "energy_growth_ratio": physics_residual["energy_growth_ratio"],
                "high_frequency_energy_ratio": physics_residual[
                    "high_frequency_energy_ratio"
                ],
            }
        )
    return record


def _recommendation(
    *,
    records: list[dict[str, Any]],
    comparison_winners: list[dict[str, Any]],
) -> dict[str, Any]:
    winner_strategies = [
        winner["winner_training_strategy"] for winner in comparison_winners
    ]
    unique_winners = sorted(set(winner_strategies))
    all_improved = all(record["score_delta"] > 0 for record in records)
    add_more = not (len(unique_winners) == 1 and all_improved)
    recommended_strategy = unique_winners[0] if len(unique_winners) == 1 else None
    return {
        "add_more_training_strategies": add_more,
        "recommended_training_strategy": recommended_strategy,
        "winner_strategy_counts": {
            strategy: winner_strategies.count(strategy) for strategy in unique_winners
        },
        "all_candidates_improved_over_baseline": all_improved,
        "rationale": (
            "one strategy won every completed cell and all candidates improved"
            if not add_more
            else "winner varies or some candidates failed to improve"
        ),
        "next_command": "run-fno-evidence-sweep"
        if add_more
        else "run-fno-iteration-dryrun",
    }


def _strategy_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["training_strategy"], []).append(record)
    return {
        strategy: {
            "count": len(items),
            "mean_score": mean(item["score"] for item in items),
            "mean_score_delta": mean(item["score_delta"] for item in items),
            "best_score": max(item["score"] for item in items),
        }
        for strategy, items in sorted(grouped.items())
    }


def _strategy_for_experiment(
    *,
    records: list[dict[str, Any]],
    start_index: int,
    train_steps: int,
    experiment_id: str,
) -> str:
    for record in reversed(records):
        if (
            record["start_index"] == start_index
            and record["train_steps"] == train_steps
            and record["experiment_id"] == experiment_id
        ):
            return str(record["training_strategy"])
    raise FNOEvidenceSweepError(
        f"missing record for comparison winner experiment_id: {experiment_id}"
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in _CSV_FIELDS})


def _validate_unique_positive_ints(
    values: list[int],
    *,
    label: str,
    allow_zero: bool,
) -> list[int]:
    if not values:
        raise FNOEvidenceSweepError(f"{label} must be non-empty")
    seen: set[int] = set()
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise FNOEvidenceSweepError(f"{label} must contain integers")
        if allow_zero:
            if value < 0:
                raise FNOEvidenceSweepError(f"{label} values must be >= 0")
        elif value <= 0:
            raise FNOEvidenceSweepError(f"{label} values must be > 0")
        if value in seen:
            raise FNOEvidenceSweepError(f"{label} contains duplicate value: {value}")
        seen.add(value)
        normalized.append(value)
    return normalized


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOEvidenceSweepError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOEvidenceSweepError(f"{label} path is not a file: {path}")
    return path
