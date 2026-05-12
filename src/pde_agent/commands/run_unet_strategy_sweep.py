
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_strategy_sweep import run_unet_strategy_sweep


def run_unet_strategy_sweep_command(
    *,
    windows_path: str,
    output_dir: str,
    baseline_score_path: str,
    start_indices: list[int],
    training_strategies: list[str],
    train_steps: list[int],
    base_channels: list[int],
    depths: list[int],
    learning_rates: list[float],
    seeds: list[int],
    spectral_weights: list[float],
    physics_weights: list[float],
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, object]:
    summary = run_unet_strategy_sweep(
        windows_path=Path(windows_path),
        output_dir=Path(output_dir),
        baseline_score_path=Path(baseline_score_path),
        start_indices=start_indices,
        training_strategies=training_strategies,
        train_steps=train_steps,
        base_channels=base_channels,
        depths=depths,
        learning_rates=learning_rates,
        seeds=seeds,
        spectral_weights=spectral_weights,
        physics_weights=physics_weights,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-unet-strategy-sweep",
        state_hint="UNET_STRATEGY_SWEEP_COMPLETED",
        artifacts={
            "summary_json_path": summary["artifacts"]["summary_json_path"],
            "summary_csv_path": summary["artifacts"]["summary_csv_path"],
            "evidence_summary_json_path": summary["artifacts"][
                "evidence_summary_json_path"
            ],
        },
        metrics={
            "record_count": summary["record_count"],
            "evaluation_window_count": summary["evaluation_window_count"],
            "winner_experiment_id": summary["winner"]["experiment_id"],
            "winner_strategy_label": summary["winner"]["strategy_label"],
            "best_score": _best_score(summary),
            "add_more_training_strategies": summary["recommendation"][
                "add_more_training_strategies"
            ],
        },
        next_hint=summary["next_command"],
    )


def _best_score(summary: dict[str, Any]) -> float | None:
    best: float | None = None
    for stats in summary["strategy_statistics"].values():
        score = stats.get("best_score")
        if isinstance(score, int | float):
            best = float(score) if best is None else max(best, float(score))
    return best
