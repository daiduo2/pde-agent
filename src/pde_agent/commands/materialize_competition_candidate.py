from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_candidate_materialization import (
    STATE_HINT,
    materialize_competition_candidate,
)
from pde_agent.core.json_output import success_payload


def materialize_competition_candidate_command(
    *,
    task_id: int,
    candidate_report_path: str,
    test_path: str,
    output_dir: str,
    submission_id: str,
    target_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    source_prediction_key: str,
    train_time: float | None,
    force: bool,
) -> dict[str, object]:
    report = materialize_competition_candidate(
        task_id=task_id,
        candidate_report_path=Path(candidate_report_path),
        test_path=Path(test_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        source_prediction_key=source_prediction_key,
        train_time=train_time,
        force=force,
    )
    score = report["score"]
    return success_payload(
        run_id="local",
        command="materialize-competition-candidate",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_id": report["task_id"],
            "has_score": score is not None,
            "segment_score": score["segment_score"] if score is not None else None,
            "task_prediction_score": score["task_prediction_score"]
            if score is not None
            else None,
        },
        next_hint=str(report["next_command"]),
    )
