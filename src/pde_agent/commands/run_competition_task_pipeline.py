
from pathlib import Path

from pde_agent.core.competition_task_pipeline import (
    STATE_HINT,
    run_competition_task_pipeline,
)
from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_leaderboard import record_local_strategy_result
from pde_agent.core.local_strategy_result_ingestion import ingest_local_strategy_result


def run_competition_task_pipeline_command(
    *,
    task_id: int,
    model_family: str,
    test_path: str,
    checkpoint_path: str,
    output_dir: str,
    submission_id: str,
    target_path: str | None,
    dataset_key: str,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
    local_strategy_schedule_path: str | None = None,
    local_strategy_experiment_id: str | None = None,
    local_strategy_leaderboard_path: str | None = None,
    local_strategy_baseline_score: float | None = None,
) -> dict[str, object]:
    report = run_competition_task_pipeline(
        task_id=task_id,
        model_family=model_family,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    materialized = report["materialized"]
    score = materialized.get("score")
    local_strategy = _maybe_record_local_strategy_result(
        report=report,
        output_dir=Path(output_dir),
        schedule_path=local_strategy_schedule_path,
        experiment_id=local_strategy_experiment_id,
        leaderboard_path=local_strategy_leaderboard_path,
        baseline_score=local_strategy_baseline_score,
    )
    next_hint = (
        str(local_strategy["decision"]["next_command"])
        if local_strategy is not None
        else str(report["next_command"])
    )
    return success_payload(
        run_id="local",
        command="run-competition-task-pipeline",
        state_hint=STATE_HINT,
        artifacts={
            **report["artifacts"],
            **(
                {
                    "local_strategy_result_path": local_strategy["result_path"],
                    "local_strategy_leaderboard_path": local_strategy[
                        "leaderboard_path"
                    ],
                }
                if local_strategy is not None
                else {}
            ),
        },
        metrics={
            "task_id": report["task_id"],
            "model_family": report["model_family"],
            "audit_decision": report["audit"]["decision"],
            "audit_issue_count": report["audit"]["issue_count"],
            "has_score": score is not None,
            "segment_score": score["segment_score"] if score is not None else None,
            "local_strategy_decision": local_strategy["decision"]["decision"]
            if local_strategy is not None
            else None,
        },
        next_hint=next_hint,
    )


def _maybe_record_local_strategy_result(
    *,
    report: dict[str, object],
    output_dir: Path,
    schedule_path: str | None,
    experiment_id: str | None,
    leaderboard_path: str | None,
    baseline_score: float | None,
) -> dict[str, object] | None:
    if schedule_path is None and experiment_id is None and leaderboard_path is None:
        return None
    if not schedule_path or not experiment_id or not leaderboard_path:
        raise ValueError(
            "local strategy result recording requires schedule path, "
            "experiment id, and leaderboard path"
        )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("competition pipeline report must include artifacts")
    source_path = artifacts.get("score_json_path") or artifacts.get(
        "pipeline_report_path"
    )
    if not isinstance(source_path, str):
        raise ValueError("competition pipeline report has no ingestible source artifact")
    result_path = output_dir / "local_strategy_result.json"
    result = ingest_local_strategy_result(
        source_path=Path(source_path),
        output_path=result_path,
        baseline_score=baseline_score,
        result_status="completed",
        failure_reason=None,
    )
    leaderboard = record_local_strategy_result(
        schedule_path=Path(schedule_path),
        experiment_id=experiment_id,
        result_summary_path=result_path,
        leaderboard_path=Path(leaderboard_path),
        result_status=str(result["result_status"]),
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    return {
        "result_path": str(result_path),
        "leaderboard_path": str(leaderboard_path),
        "decision": leaderboard["decision"],
    }
