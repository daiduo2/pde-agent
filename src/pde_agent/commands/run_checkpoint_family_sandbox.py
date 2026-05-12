
from pathlib import Path

from pde_agent.core.checkpoint_family_sandbox import (
    STATE_HINT,
    run_checkpoint_family_sandbox,
)
from pde_agent.core.json_output import success_payload


def run_checkpoint_family_sandbox_command(
    *,
    checkpoint_paths: list[str],
    task_id: int,
    test_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    dataset_key: str,
    prediction_dataset_key: str,
    calibration_weights: list[float],
    fno_finetune_steps: int,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
) -> dict[str, object]:
    report = run_checkpoint_family_sandbox(
        checkpoint_paths=[Path(path) for path in checkpoint_paths],
        task_id=task_id,
        test_path=Path(test_path),
        target_path=Path(target_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        calibration_weights=calibration_weights,
        fno_finetune_steps=fno_finetune_steps,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    best = report["best_candidate"]
    return success_payload(
        run_id="local",
        command="run-checkpoint-family-sandbox",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_id": report.get("task_id"),
            "candidate_count": report.get("candidate_count", 0),
            "best_candidate_id": best["candidate_id"],
            "best_score": best["score"],
            "best_model_family": best["model_family"],
            "best_source": best.get("source"),
            "best_calibration_weight": best.get("calibration_weight"),
            "best_rules_passed": best.get("rules_passed"),
            "best_train_time": best.get("train_time"),
            "best_inference_time": best.get("inference_time"),
            "best_raw_train_time": best.get("raw_train_time"),
            "best_raw_inference_time": best.get("raw_inference_time"),
            "best_timing_profile": best.get("timing_profile"),
            "best_timing_factor": best.get("timing_factor"),
        },
        next_hint=report["next_command"],
    )
