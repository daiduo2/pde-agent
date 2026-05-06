from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_heartbeat import STATE_HINT, run_hermes_heartbeat
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
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
    learning_objective_schedule: list[str],
    patience: int,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    unet_checkpoint_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    candidate_generator: str,
    candidate_generator_schedule: list[str],
    hypothesis_blueprint_path: str | None,
    hypothesis_reference_path: str | None,
    hypothesis_reference_sample_limit: int | None,
    hypothesis_reference_sample_limits: list[int],
    hypothesis_kpi_report_path: str | None = None,
    repair_policy_summary_path: str | None = None,
    saturation_probe_interval_batches: int = 0,
    postprocess_search_mode: str,
    postprocess_strategies: list[str],
    postprocess_blend_weights: list[float],
    postprocess_adaptive_step: float,
    submission_id: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
    trace_path: str | None = None,
    doom_loop_report_path: str | None = None,
    candidate_generator_fallback: str = "auto",
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
        learning_objective_schedule=learning_objective_schedule or None,
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
        candidate_generator=candidate_generator,
        candidate_generator_schedule=candidate_generator_schedule or None,
        doom_loop_report_path=Path(doom_loop_report_path)
        if doom_loop_report_path is not None
        else None,
        candidate_generator_fallback=candidate_generator_fallback,
        hypothesis_blueprint_path=Path(hypothesis_blueprint_path)
        if hypothesis_blueprint_path is not None
        else None,
        hypothesis_reference_path=Path(hypothesis_reference_path)
        if hypothesis_reference_path is not None
        else None,
        hypothesis_reference_sample_limit=hypothesis_reference_sample_limit,
        hypothesis_reference_sample_limits=hypothesis_reference_sample_limits,
        hypothesis_kpi_report_path=Path(hypothesis_kpi_report_path)
        if hypothesis_kpi_report_path is not None
        else None,
        repair_policy_summary_path=Path(repair_policy_summary_path)
        if repair_policy_summary_path is not None
        else None,
        saturation_probe_interval_batches=saturation_probe_interval_batches,
        postprocess_search_mode=postprocess_search_mode,
        postprocess_strategies=postprocess_strategies,
        postprocess_blend_weights=postprocess_blend_weights,
        postprocess_adaptive_step=postprocess_adaptive_step,
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        parameters = (
            report.get("parameters") if isinstance(report.get("parameters"), dict) else {}
        )
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hermes_heartbeat_run",
            payload={
                "algorithm_candidate_id": report["best_candidate"].get(
                    "candidate_id"
                ),
                "candidate_family": report["best_candidate"].get("family"),
                "outcome": report["decision"],
                "best_score": report["best_score"],
                "score_delta": None,
                "batch_count": report["batch_count"],
                "stop_reason": report["stop_reason"],
                "guard_active": bool(report.get("guard", {}).get("guard_active"))
                if isinstance(report.get("guard"), dict)
                else False,
                "saturated_candidate_families": parameters.get(
                    "saturated_candidate_families",
                    [],
                ),
                "persistent_search_guidance": report.get(
                    "persistent_search_guidance",
                    [],
                ),
                "repair_policy_recommended_actions": report.get(
                    "repair_policy_recommended_actions",
                    [],
                ),
            },
            run_id="local",
            source_artifact_path=report["heartbeat_report_path"],
            tags=["hermes", "heartbeat"],
        )
    return success_payload(
        run_id="local",
        command="run-hermes-heartbeat",
        state_hint=STATE_HINT,
        artifacts={
            "heartbeat_report_path": report["heartbeat_report_path"],
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            "batch_count": report["batch_count"],
            "best_candidate_id": report["best_candidate"].get("candidate_id"),
            "best_score": report["best_score"],
            "best_objective": report.get("best_objective"),
            "best_objective_score": report.get("best_objective_score"),
            "stop_reason": report["stop_reason"],
            "decision": report["decision"],
            "repair_policy_recommended_action_count": (
                report.get("parameters", {}).get(
                    "repair_policy_recommended_action_count",
                    0,
                )
                if isinstance(report.get("parameters"), dict)
                else 0
            ),
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )
