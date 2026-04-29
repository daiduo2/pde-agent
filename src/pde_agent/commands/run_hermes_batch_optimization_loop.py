from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_batch_optimization_loop import (
    STATE_HINT,
    run_hermes_batch_optimization_loop,
)
from pde_agent.core.json_output import success_payload


def run_hermes_batch_optimization_loop_command(
    *,
    output_dir: str,
    initial_submission_dir: str,
    max_rounds: int,
    patience: int,
    min_score_improvement: float,
    submit_score_threshold: float,
    learning_objective: str,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    unet_checkpoint_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    candidate_submission_dirs: list[str],
    candidate_generator: str,
    postprocess_search_mode: str,
    postprocess_strategies: list[str],
    postprocess_blend_weights: list[float],
    postprocess_adaptive_step: float,
    submission_id: str,
    deadline_seconds: float | None,
    checkpoint_every: int,
    memory_json_path: str | None,
    avoid_repeating: bool,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_batch_optimization_loop(
        output_dir=Path(output_dir),
        initial_submission_dir=Path(initial_submission_dir),
        max_rounds=max_rounds,
        patience=patience,
        min_score_improvement=min_score_improvement,
        submit_score_threshold=submit_score_threshold,
        learning_objective=learning_objective,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        unet_checkpoint_path=Path(unet_checkpoint_path)
        if unet_checkpoint_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        candidate_submission_dirs=[Path(path) for path in candidate_submission_dirs],
        candidate_generator=candidate_generator,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_adaptive_step=postprocess_adaptive_step,
        submission_id=submission_id,
        deadline_seconds=deadline_seconds,
        checkpoint_every=checkpoint_every,
        memory_json_path=Path(memory_json_path) if memory_json_path is not None else None,
        avoid_repeating=avoid_repeating,
        force=force,
    )
    best = report["best_candidate"]
    return success_payload(
        run_id="local",
        command="run-hermes-batch-optimization-loop",
        state_hint=STATE_HINT,
        artifacts={"loop_report_path": report["loop_report_path"]},
        metrics={
            "round_count": report["round_count"],
            "best_candidate_id": best["candidate_id"],
            "best_score": best["score"],
            "learning_objective": report["parameters"]["learning_objective"],
            "best_objective": best.get("objective", {}).get("selected_objective"),
            "best_objective_score": best.get("objective", {}).get("objective_score"),
            "stop_reason": report["stop_reason"],
            "decision": report["decision"],
            "memory_checkpoint_count": len(report.get("memory_checkpoints", [])),
            "skipped_candidate_count": len(report.get("skipped_candidates", [])),
        },
        next_hint=report["next_command"],
    )
