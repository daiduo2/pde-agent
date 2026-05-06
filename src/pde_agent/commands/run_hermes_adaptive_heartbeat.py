from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_execution_policy import (
    POLICY_STATE_HINT,
    run_hermes_adaptive_heartbeat,
)
from pde_agent.core.json_output import success_payload


def run_hermes_adaptive_heartbeat_command(
    *,
    output_dir: str,
    initial_submission_dir: str,
    verified_blueprint_path: str | None,
    max_policy_stages: int,
    max_batches: int,
    batch_deadline_seconds: float,
    max_rounds_per_batch: int,
    checkpoint_every: int,
    max_stagnant_batches: int,
    min_batch_score_improvement: float,
    min_score_improvement: float,
    submit_score_threshold: float,
    learning_objective: str,
    learning_objective_schedule: list[str],
    patience: int,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    unet_checkpoint_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    initial_candidate_generator: str,
    candidate_generator_schedule: list[str],
    hypothesis_reference_path: str | None,
    hypothesis_reference_sample_limit: int | None,
    hypothesis_reference_sample_limits: list[int],
    postprocess_search_mode: str,
    postprocess_strategies: list[str],
    postprocess_blend_weights: list[float],
    postprocess_adaptive_step: float,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_adaptive_heartbeat(
        output_dir=Path(output_dir),
        initial_submission_dir=Path(initial_submission_dir),
        verified_blueprint_path=Path(verified_blueprint_path)
        if verified_blueprint_path is not None
        else None,
        max_policy_stages=max_policy_stages,
        max_batches=max_batches,
        batch_deadline_seconds=batch_deadline_seconds,
        max_rounds_per_batch=max_rounds_per_batch,
        checkpoint_every=checkpoint_every,
        max_stagnant_batches=max_stagnant_batches,
        min_batch_score_improvement=min_batch_score_improvement,
        min_score_improvement=min_score_improvement,
        submit_score_threshold=submit_score_threshold,
        learning_objective=learning_objective,
        learning_objective_schedule=learning_objective_schedule,
        patience=patience,
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
        initial_candidate_generator=initial_candidate_generator,
        candidate_generator_schedule=candidate_generator_schedule,
        hypothesis_reference_path=Path(hypothesis_reference_path)
        if hypothesis_reference_path is not None
        else None,
        hypothesis_reference_sample_limit=hypothesis_reference_sample_limit,
        hypothesis_reference_sample_limits=hypothesis_reference_sample_limits,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_adaptive_step=postprocess_adaptive_step,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-adaptive-heartbeat",
        state_hint=POLICY_STATE_HINT,
        artifacts={"adaptive_report_path": report["adaptive_report_path"]},
        metrics={
            "stage_count": report["stage_count"],
            "best_candidate_id": report["best_candidate"].get("candidate_id"),
            "best_score": report["best_score"],
            "decision": report["decision"],
            "last_action": report["policy_events"][-1]["action"]
            if report["policy_events"]
            else None,
        },
        next_hint=report["next_command"],
    )
