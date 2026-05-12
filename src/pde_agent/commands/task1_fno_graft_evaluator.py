
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.task1_fno_graft_evaluator import (
    STATE_HINT,
    run_task1_fno_graft_from_optimization_report,
    run_task1_fno_graft_evaluator,
)


def run_task1_fno_graft_evaluator_command(
    *,
    baseline_prediction_path: str,
    checkpoint_path: str,
    test_path: str,
    target_path: str,
    output_dir: str,
    dataset_key: str,
    prediction_dataset_key: str,
    target_dataset_key: str,
    windows: list[str],
    blend_weights: list[float],
    late_non_regression_tolerance: float,
    force: bool,
) -> dict[str, object]:
    report = run_task1_fno_graft_evaluator(
        baseline_prediction_path=Path(baseline_prediction_path),
        checkpoint_path=Path(checkpoint_path),
        test_path=Path(test_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        target_dataset_key=target_dataset_key,
        windows=[_parse_window(value) for value in windows] if windows else None,
        blend_weights=blend_weights or None,
        late_non_regression_tolerance=late_non_regression_tolerance,
        force=force,
    )
    best = report["best_candidate"]
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-fno-graft-evaluator",
        state_hint=STATE_HINT,
        artifacts={
            "report_path": report["artifacts"]["report_path"],
            "source_prediction_path": report["artifacts"]["source_prediction_path"],
            "best_prediction_path": report["artifacts"]["best_prediction_path"],
        },
        metrics={
            "baseline_task_prediction_score": report["baseline_score"][
                "task_prediction_score"
            ],
            "source_task_prediction_score": report["source_score"][
                "task_prediction_score"
            ],
            "best_task_prediction_score": best["task_prediction_score"],
            "best_task_prediction_score_delta": best["task_prediction_score_delta"],
            "best_late_score_delta": best["late_score_delta"],
            "late_non_regression_passed": best["late_non_regression_passed"],
            "candidate_count": len(report["candidate_records"]),
        },
        next_hint=report["next_command"],
    )


def run_task1_fno_graft_from_optimization_report_command(
    *,
    optimization_report_path: str,
    baseline_prediction_path: str,
    test_path: str,
    target_path: str,
    output_dir: str,
    dataset_key: str,
    prediction_dataset_key: str,
    target_dataset_key: str,
    windows: list[str],
    blend_weights: list[float],
    late_non_regression_tolerance: float,
    force: bool,
) -> dict[str, object]:
    report = run_task1_fno_graft_from_optimization_report(
        optimization_report_path=Path(optimization_report_path),
        baseline_prediction_path=Path(baseline_prediction_path),
        test_path=Path(test_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        target_dataset_key=target_dataset_key,
        windows=[_parse_window(value) for value in windows] if windows else None,
        blend_weights=blend_weights or None,
        late_non_regression_tolerance=late_non_regression_tolerance,
        force=force,
    )
    best = report["best_candidate"]
    selected = report["selected_training_candidate"]
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-fno-graft-from-optimization-report",
        state_hint=STATE_HINT,
        artifacts={
            "report_path": report["artifacts"]["report_path"],
            "source_prediction_path": report["artifacts"]["source_prediction_path"],
            "best_prediction_path": report["artifacts"]["best_prediction_path"],
            "optimization_report_path": report["artifacts"]["optimization_report_path"],
        },
        metrics={
            "selected_training_candidate_id": selected.get("candidate_id"),
            "baseline_task_prediction_score": report["baseline_score"][
                "task_prediction_score"
            ],
            "source_task_prediction_score": report["source_score"][
                "task_prediction_score"
            ],
            "best_task_prediction_score": best["task_prediction_score"],
            "best_task_prediction_score_delta": best["task_prediction_score_delta"],
            "best_late_score_delta": best["late_score_delta"],
            "late_non_regression_passed": best["late_non_regression_passed"],
            "candidate_count": len(report["candidate_records"]),
        },
        next_hint=report["next_command"],
    )


def _parse_window(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError("window must use START:STOP format")
    start, stop = value.split(":", maxsplit=1)
    return int(start), int(stop)
