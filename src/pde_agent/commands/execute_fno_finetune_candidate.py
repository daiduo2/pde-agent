from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_finetune_candidate import execute_fno_finetune_candidate
from pde_agent.core.json_output import success_payload


def execute_fno_finetune_candidate_command(
    *,
    candidate_plan_path: str,
    windows_path: str,
    checkpoint_path: str,
    max_train_steps: int,
    learning_rate: float,
    force: bool,
) -> dict[str, object]:
    report = execute_fno_finetune_candidate(
        candidate_plan_path=Path(candidate_plan_path),
        windows_path=Path(windows_path),
        checkpoint_path=Path(checkpoint_path),
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        force=force,
    )
    loss_trace = report["train"]["loss_trace"]
    return success_payload(
        run_id="local",
        command="execute-fno-finetune-candidate",
        state_hint="FNO_FINETUNE_CANDIDATE_EXECUTED",
        artifacts={
            "finetuned_checkpoint_path": report["artifacts"][
                "finetuned_checkpoint_path"
            ],
            "candidate_prediction_path": report["artifacts"][
                "candidate_prediction_path"
            ],
            "candidate_execution_report_path": report["artifacts"][
                "candidate_execution_report_path"
            ],
        },
        metrics={
            "experiment_id": report["candidate"]["experiment_id"],
            "intervention": report["candidate"]["intervention"],
            "train_steps": report["train"]["train_steps"],
            "initial_loss": loss_trace[0] if loss_trace else None,
            "final_loss": loss_trace[-1] if loss_trace else None,
            "prediction_shape": report["prediction_shape"],
        },
        next_hint="score-local-rollout",
    )
