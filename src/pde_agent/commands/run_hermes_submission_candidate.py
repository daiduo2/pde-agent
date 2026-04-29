from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_submission_candidate import (
    STATE_HINT,
    run_hermes_submission_candidate,
)
from pde_agent.core.json_output import success_payload


def run_hermes_submission_candidate_command(
    *,
    tuning_report_path: str,
    output_dir: str,
    task_id: int,
    test_path: str,
    checkpoint_path: str,
    submission_id: str,
    target_path: str | None,
    dataset_key: str,
    prediction_dataset_key: str,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_submission_candidate(
        tuning_report_path=Path(tuning_report_path),
        output_dir=Path(output_dir),
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        force=force,
    )
    selected = report["selected_config"]
    return success_payload(
        run_id="local",
        command="run-hermes-submission-candidate",
        state_hint=STATE_HINT,
        artifacts={
            "hermes_submission_candidate_path": report["candidate_report_path"],
            "submission_dir": report["materialized"]["artifacts"]["submission_dir"],
            "submission_zip_path": report["materialized"]["artifacts"][
                "submission_zip_path"
            ],
            "chronicle_path": report["chronicle"]["artifacts"]["chronicle_path"],
            "audit_path": report["audit"]["audit_path"],
        },
        metrics={
            "task_id": report["task_id"],
            "inference_batch_size": selected["inference_batch_size"],
            "torch_num_threads": selected["torch_num_threads"],
            "audit_decision": report["audit"]["decision"],
            "audit_issue_count": report["audit"]["issue_count"],
        },
        next_hint=report["next_command"],
    )
