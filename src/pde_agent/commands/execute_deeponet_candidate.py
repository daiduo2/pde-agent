from __future__ import annotations

from pathlib import Path

from pde_agent.core.deeponet_candidate_execution import execute_deeponet_candidate
from pde_agent.core.json_output import success_payload


def execute_deeponet_candidate_command(
    *,
    windows_path: str,
    output_dir: str,
    start_index: int,
    train_steps: int,
    training_strategy: str,
    hidden_width: int,
    latent_dim: int,
    learning_rate: float,
    seed: int,
    strategy_label: str | None,
    force: bool,
) -> dict[str, object]:
    report = execute_deeponet_candidate(
        windows_path=Path(windows_path),
        output_dir=Path(output_dir),
        start_index=start_index,
        train_steps=train_steps,
        training_strategy=training_strategy,
        hidden_width=hidden_width,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        seed=seed,
        strategy_label=strategy_label,
        force=force,
    )
    loss_trace = report["train"]["loss_trace"]
    return success_payload(
        run_id="local",
        command="execute-deeponet-candidate",
        state_hint="DEEPONET_CANDIDATE_EXECUTED",
        artifacts={
            "candidate_prediction_path": report["artifacts"][
                "candidate_prediction_path"
            ],
            "candidate_checkpoint_path": report["artifacts"][
                "candidate_checkpoint_path"
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
            "strategy_label": report["candidate"]["strategy_label"],
            "initial_loss": loss_trace[0] if loss_trace else None,
            "final_loss": loss_trace[-1] if loss_trace else None,
            "hidden_width": report["train"]["hidden_width"],
            "latent_dim": report["train"]["latent_dim"],
            "learning_rate": report["train"]["learning_rate"],
            "forecast_steps": report["inputs"]["forecast_steps"],
        },
        next_hint="evaluate-deeponet-candidate",
    )
