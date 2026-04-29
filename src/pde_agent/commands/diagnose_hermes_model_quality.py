from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_model_quality_diagnosis import (
    STATE_HINT,
    diagnose_hermes_model_quality,
)
from pde_agent.core.json_output import success_payload


def diagnose_hermes_model_quality_command(
    *,
    validation_report_path: str,
    output_path: str,
    min_segment_score: float,
    min_task_prediction_score: float,
    max_late_rmse: float,
    max_inference_seconds: float,
) -> dict[str, object]:
    report = diagnose_hermes_model_quality(
        validation_report_path=Path(validation_report_path),
        output_path=Path(output_path),
        min_segment_score=min_segment_score,
        min_task_prediction_score=min_task_prediction_score,
        max_late_rmse=max_late_rmse,
        max_inference_seconds=max_inference_seconds,
    )
    return success_payload(
        run_id="local",
        command="diagnose-hermes-model-quality",
        state_hint=STATE_HINT,
        artifacts={"diagnosis_path": report["diagnosis_path"]},
        metrics={
            "operator_family": report["operator_family"],
            "primary_bottleneck": report["primary_bottleneck"],
            "recommended_strategy_family": report["recommended_strategy_family"],
            "confidence": report["confidence"],
        },
        next_hint=report["next_command"],
    )
