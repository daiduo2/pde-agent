from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.baseline_prediction import generate_baseline_prediction
from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.burgers_residual_evaluator import evaluate_burgers_residual_file
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.local_windows import build_local_windows
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


STATE_HINT = "REAL_HDF5_BASELINE_SMOKE_COMPLETED"


class RealHDF5BaselineSmokeError(RuntimeError):
    pass


def run_real_hdf5_baseline_smoke(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
    force: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    paths = _paths(output_root, forecast_steps)
    _validate_overwrite(paths=paths, force=force)

    window_summary = build_local_windows(
        source_path=source_path,
        output_path=paths["windows_npz_path"],
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    prediction = generate_baseline_prediction(
        windows_path=paths["windows_npz_path"],
        output_path=paths["prediction_npz_path"],
        method="persistence",
        force=force,
    )
    score = score_prediction_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=paths["windows_npz_path"],
        output_path=paths["score_path"],
    )
    rollout_error = diagnose_rollout_error_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=paths["windows_npz_path"],
        score_path=paths["score_path"],
        output_path=paths["rollout_error_path"],
    )
    physics = check_burgers_physics_npz(
        prediction_path=paths["prediction_npz_path"],
        target_path=paths["windows_npz_path"],
        output_path=paths["burgers_physics_path"],
        nu=nu,
        dt=dt,
        dx=dx,
    )
    residual = evaluate_burgers_residual_file(
        trajectory_path=paths["prediction_npz_path"],
        output_path=paths["burgers_residual_path"],
        array_key="prediction",
        nu=nu,
        dt=dt,
        dx=dx,
    )

    summary = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "inputs": {
            "source_path": str(Path(source_path)),
            "dataset_key": dataset_key,
            "start_index": start_index,
            "sample_limit": sample_limit,
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
            "time_reduction": time_reduction,
            "space_reduction": space_reduction,
            "nu": nu,
            "dt": dt,
            "dx": dx,
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "window_summary": window_summary,
        "baseline": {
            "method": prediction["method"],
            "prediction_shape": prediction["prediction_shape"],
        },
        "metrics": {
            "sample_count": score["sample_count"],
            "forecast_steps": score["time_steps"],
            "space_points": score["space_points"],
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
            "dominant_error_pattern": rollout_error["dominant_error_pattern"],
            "physics_issue": physics["physics_issue"],
            "residual_issue": residual["physics_issue"],
            "residual_rmse": residual["residual_rmse"],
            "mass_drift": residual["mass_drift"],
            "energy_growth_ratio": residual["energy_growth_ratio"],
            "high_frequency_energy_ratio": residual["high_frequency_energy_ratio"],
        },
        "next_command": "inspect-real-baseline-smoke-summary",
    }
    paths["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_path"].write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _paths(output_root: Path, forecast_steps: int) -> dict[str, Path]:
    return {
        "windows_npz_path": output_root / f"windows_f{forecast_steps:03d}.npz",
        "prediction_npz_path": output_root / "persistence_prediction.npz",
        "score_path": output_root / "persistence_score.json",
        "rollout_error_path": output_root / "persistence_rollout_error.json",
        "burgers_physics_path": output_root / "persistence_burgers_physics.json",
        "burgers_residual_path": output_root / "persistence_burgers_residual.json",
        "summary_path": output_root / "real_hdf5_baseline_smoke_summary.json",
    }


def _validate_overwrite(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise RealHDF5BaselineSmokeError(f"smoke artifacts already exist: {names}")
