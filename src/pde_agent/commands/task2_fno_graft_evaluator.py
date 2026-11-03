"""CLI command wrapper for Task 2 FNO graft evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import error_payload, success_payload
from pde_agent.core.task2_fno_graft_evaluator import (
    Task2FnoGraftEvaluatorError,
    run_task2_fno_graft_evaluator,
)


def run_task2_fno_graft_evaluator_command(
    *,
    base_checkpoint_path: str,
    front_checkpoint_path: str,
    output_dir: str = "/tmp/task2_fno_graft_eval",
    val_data_path: str | None = None,
    windows: list[tuple[int, int]] | None = None,
    blend_weights: list[float] | None = None,
    residual_scales: list[float] | None = None,
    late_non_regression_tolerance: float = 1.0,
    device: str = "mps",
    force: bool = False,
) -> dict[str, Any]:
    """Run Task 2 FNO graft evaluator."""
    try:
        report = run_task2_fno_graft_evaluator(
            base_checkpoint_path=base_checkpoint_path,
            front_checkpoint_path=front_checkpoint_path,
            output_dir=output_dir,
            val_data_path=val_data_path,
            windows=windows,
            blend_weights=blend_weights,
            residual_scales=residual_scales,
            late_non_regression_tolerance=late_non_regression_tolerance,
            device=device,
            force=force,
        )
    except Task2FnoGraftEvaluatorError as exc:
        return error_payload(
            run_id=Path(output_dir).name or "task2-graft",
            command="run-task2-fno-graft-evaluator",
            error_code="graft_eval_failed",
            error_type="Task2FnoGraftEvaluatorError",
            message_short=str(exc),
            recoverable=True,
            retryable=True,
        )

    return success_payload(
        run_id=Path(output_dir).name or "task2-graft",
        command="run-task2-fno-graft-evaluator",
        state_hint=report["state_hint"],
        artifacts=report.get("artifacts", {}),
        metrics={
            "base_task2_proxy": report["base_score"]["segment_score"] * 1.5,
            "front_task2_proxy": report["front_score"]["segment_score"] * 1.5,
            "best_task2_proxy": report["best_candidate"]["task2_proxy"],
            "best_segment_score_delta": report["best_candidate"]["segment_score_delta"],
            "best_late_score_delta": report["best_candidate"]["late_score_delta"],
            "candidate_count": len(report["candidate_records"]),
        },
        next_hint=report.get("next_command", ""),
    )
