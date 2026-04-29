from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_result_diagnosis import STATE_HINT, diagnose_hermes_result
from pde_agent.core.json_output import success_payload


def diagnose_hermes_result_command(
    *,
    artifact_path: str,
    output_path: str,
    max_inference_seconds: float,
    min_score_delta: float,
) -> dict[str, object]:
    diagnosis = diagnose_hermes_result(
        artifact_path=Path(artifact_path),
        output_path=Path(output_path),
        max_inference_seconds=max_inference_seconds,
        min_score_delta=min_score_delta,
    )
    return success_payload(
        run_id="local",
        command="diagnose-hermes-result",
        state_hint=STATE_HINT,
        artifacts={"diagnosis_path": diagnosis["diagnosis_path"]},
        metrics={
            "primary_bottleneck": diagnosis["primary_bottleneck"],
            "recommended_action_family": diagnosis["recommended_action_family"],
            "operator_family": diagnosis["operator_family"],
            "confidence": diagnosis["confidence"],
        },
        next_hint=diagnosis["next_command"],
    )
