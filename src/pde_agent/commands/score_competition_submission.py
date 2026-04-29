from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_scoring import (
    STATE_HINT,
    score_competition_prediction,
)
from pde_agent.core.json_output import success_payload


def score_competition_submission_command(
    *,
    task_id: int,
    prediction_path: str,
    target_path: str,
    output_path: str,
    prediction_dataset_key: str,
    target_dataset_key: str,
) -> dict[str, object]:
    score = score_competition_prediction(
        task_id=task_id,
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
        prediction_dataset_key=prediction_dataset_key,
        target_dataset_key=target_dataset_key,
    )
    return success_payload(
        run_id="local",
        command="score-competition-submission",
        state_hint=STATE_HINT,
        artifacts={"score_json_path": score["score_path"]},
        metrics={
            "task_id": score["task_id"],
            "segment_score": score["segment_score"],
            "task_prediction_score": score["task_prediction_score"],
            "sample_count": score["sample_count"],
        },
        next_hint="use-score-to-select-competition-strategy",
    )
