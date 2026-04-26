from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.spectral_residual_baseline import (
    train_spectral_residual_baseline as train_model,
)


def train_spectral_residual_baseline(
    *,
    windows_npz: str,
    checkpoint_path: str,
    modes: int,
    ridge: float,
) -> dict[str, object]:
    summary = train_model(
        windows_path=Path(windows_npz),
        checkpoint_path=Path(checkpoint_path),
        modes=modes,
        ridge=ridge,
        force=False,
    )
    return success_payload(
        run_id="local",
        command="train-spectral-residual-baseline",
        state_hint="LOCAL_MODEL_TRAINED",
        artifacts={"checkpoint_path": summary["checkpoint_path"]},
        metrics={
            "model_type": summary["model_type"],
            "modes": summary["modes"],
            "ridge": summary["ridge"],
            "train_sample_count": summary["train_sample_count"],
            "forecast_steps": summary["forecast_steps"],
            "space_points": summary["space_points"],
        },
        next_hint="predict-local-model",
    )
