from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_heartbeat import STATE_HINT, run_hermes_heartbeat
from pde_agent.core.json_output import success_payload


def run_hermes_heartbeat_command(
    *,
    output_dir: str,
    initial_submission_dir: str,
    max_batches: int,
    batch_deadline_seconds: float,
    max_rounds_per_batch: int,
    checkpoint_every: int,
    max_stagnant_batches: int,
    min_batch_score_improvement: float,
    min_score_improvement: float,
    submit_score_threshold: float,
    learning_objective: str,
    patience: int,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    candidate_generator: str,
    postprocess_search_mode: str,
    postprocess_strategies: list[str],
    postprocess_blend_weights: list[float],
    postprocess_adaptive_step: float,
    submission_id: str,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_heartbeat(
        output_dir=Path(output_dir),
        initial_submission_dir=Path(initial_submission_dir),
        max_batches=max_batches,
        batch_deadline_seconds=batch_deadline_seconds,
        max_rounds_per_batch=max_rounds_per_batch,
        checkpoint_every=checkpoint_every,
        max_stagnant_batches=max_stagnant_batches,
        min_batch_score_improvement=min_batch_score_improvement,
        min_score_improvement=min_score_improvement,
        submit_score_threshold=submit_score_threshold,
        learning_objective=learning_objective,
        patience=patience,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        candidate_generator=candidate_generator,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_adaptive_step=postprocess_adaptive_step,
        submission_id=submission_id,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-heartbeat",
        state_hint=STATE_HINT,
        artifacts={"heartbeat_report_path": report["heartbeat_report_path"]},
        metrics={
            "batch_count": report["batch_count"],
            "best_candidate_id": report["best_candidate"].get("candidate_id"),
            "best_score": report["best_score"],
            "best_objective": report.get("best_objective"),
            "best_objective_score": report.get("best_objective_score"),
            "stop_reason": report["stop_reason"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )
