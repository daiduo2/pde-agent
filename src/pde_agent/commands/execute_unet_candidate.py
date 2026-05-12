
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_candidate_execution import execute_unet_candidate


def execute_unet_candidate_command(
    *,
    windows_path: str,
    output_dir: str,
    start_index: int,
    train_steps: int,
    training_strategy: str,
    base_channels: int,
    depth: int,
    learning_rate: float,
    seed: int,
    strategy_label: str | None,
    nu: float,
    dt: float,
    dx: float,
    spectral_weight: float,
    physics_weight: float,
    force: bool,
) -> dict[str, object]:
    report = execute_unet_candidate(
        windows_path=Path(windows_path),
        output_dir=Path(output_dir),
        start_index=start_index,
        train_steps=train_steps,
        training_strategy=training_strategy,
        base_channels=base_channels,
        depth=depth,
        learning_rate=learning_rate,
        seed=seed,
        strategy_label=strategy_label,
        nu=nu,
        dt=dt,
        dx=dx,
        spectral_weight=spectral_weight,
        physics_weight=physics_weight,
        force=force,
    )
    loss_trace = report["train"]["loss_trace"]
    return success_payload(
        run_id="local",
        command="execute-unet-candidate",
        state_hint="UNET_CANDIDATE_EXECUTED",
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
            "base_channels": report["train"]["base_channels"],
            "depth": report["train"]["depth"],
            "learning_rate": report["train"]["learning_rate"],
            "loss_terms": report["train"]["loss_terms"],
            "loss_weights": report["train"]["loss_weights"],
            "forecast_steps": report["inputs"]["forecast_steps"],
        },
        next_hint="evaluate-unet-candidate",
    )
