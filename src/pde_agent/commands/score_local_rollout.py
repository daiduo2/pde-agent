from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_scoring import score_prediction_npz


def score_local_rollout(
    *,
    prediction_path: str,
    target_path: str,
    output_path: str,
) -> dict[str, object]:
    score = score_prediction_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="score-local-rollout",
        state_hint="LOCAL_ROLLOUT_SCORED",
        artifacts={"score_json_path": str(output_path)},
        metrics={
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
        },
        next_hint="use this score to compare local model variants before hidden long-horizon submission",
    )
