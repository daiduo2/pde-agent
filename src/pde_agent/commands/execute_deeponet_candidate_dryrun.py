from __future__ import annotations

from pathlib import Path

from pde_agent.core.deeponet_candidate_dryrun import (
    execute_deeponet_candidate_dryrun,
)
from pde_agent.core.json_output import success_payload


def execute_deeponet_candidate_dryrun_command(
    *,
    windows_path: str,
    output_dir: str,
    start_index: int,
    train_steps: int,
    training_strategy: str,
    force: bool,
) -> dict[str, object]:
    report = execute_deeponet_candidate_dryrun(
        windows_path=Path(windows_path),
        output_dir=Path(output_dir),
        start_index=start_index,
        train_steps=train_steps,
        training_strategy=training_strategy,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="execute-deeponet-candidate-dryrun",
        state_hint="DEEPONET_CANDIDATE_EXECUTED",
        artifacts={
            "candidate_prediction_path": report["artifacts"][
                "candidate_prediction_path"
            ],
            "candidate_execution_report_path": report["artifacts"][
                "candidate_execution_report_path"
            ],
        },
        metrics={
            "experiment_id": report["candidate"]["experiment_id"],
            "start_index": report["candidate"]["start_index"],
            "train_steps": report["candidate"]["train_steps"],
            "training_strategy": report["candidate"]["training_strategy"],
            "forecast_steps": report["inputs"]["forecast_steps"],
        },
        next_hint="evaluate-deeponet-candidate",
    )
