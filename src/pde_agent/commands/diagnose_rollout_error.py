from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


def diagnose_rollout_error(
    *,
    prediction_path: str,
    target_path: str,
    score_path: str,
    output_path: str,
) -> dict[str, object]:
    diagnosis = diagnose_rollout_error_npz(
        prediction_path=Path(prediction_path),
        target_path=Path(target_path),
        score_path=Path(score_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="diagnose-rollout-error",
        state_hint="ROLLOUT_ERROR_DIAGNOSED",
        artifacts={"rollout_error_diagnosis_path": str(output_path)},
        metrics={
            "dominant_error_pattern": diagnosis["dominant_error_pattern"],
            "global_rmse": diagnosis["global_rmse"],
            "time_error_slope": diagnosis["time_error_slope"],
        },
        next_hint="check-burgers-physics-local",
    )
