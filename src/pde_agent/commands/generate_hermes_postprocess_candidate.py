from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_postprocess_candidate import (
    STATE_HINT,
    generate_hermes_postprocess_candidate,
)
from pde_agent.core.json_output import success_payload


def generate_hermes_postprocess_candidate_command(
    *,
    base_submission_dir: str,
    output_dir: str,
    task_id: int,
    test_path: str,
    strategy: str,
    blend_weight: float,
    submission_id: str,
    dataset_key: str,
    prediction_dataset_key: str,
    timing_profile: str | None,
    timing_factor: float | None,
    force: bool,
) -> dict[str, object]:
    report = generate_hermes_postprocess_candidate(
        base_submission_dir=Path(base_submission_dir),
        output_dir=Path(output_dir),
        task_id=task_id,
        test_path=Path(test_path),
        strategy=strategy,
        blend_weight=blend_weight,
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="generate-hermes-postprocess-candidate",
        state_hint=STATE_HINT,
        artifacts={
            "candidate_report_path": report["candidate_report_path"],
            "submission_dir": report["submission_dir"],
            "submission_zip_path": report["artifacts"]["submission_zip_path"],
        },
        metrics={
            "candidate_id": report["candidate_id"],
            "strategy": report["strategy"],
            "blend_weight": report.get("blend_weight"),
            "task_id": report.get("task_id"),
        },
        next_hint=str(report.get("next_command", "judge-hermes-submission")),
    )
