"""Compare official PDEBench operator checkpoints on one Burgers local window."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.fno_rollout_diagnosis import diagnose_fno_rollout
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz
from pde_agent.core.spectral_residual_baseline import (
    predict_spectral_residual_baseline,
    train_spectral_residual_baseline,
)
from pde_agent.core.unet_rollout_diagnosis import diagnose_unet_rollout


STATE_HINT = "OFFICIAL_OPERATOR_COMPARISON_COMPLETED"


class OfficialOperatorComparisonError(RuntimeError):
    pass


def run_official_operator_comparison(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    fno_checkpoint_path: str | Path,
    unet_checkpoint_path: str | Path,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    spectral_modes: int = 16,
    spectral_ridge: float = 1e-3,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    source_file = _require_file(Path(source_path), "source HDF5")
    fno_checkpoint = _require_file(Path(fno_checkpoint_path), "FNO checkpoint")
    unet_checkpoint = _require_file(Path(unet_checkpoint_path), "UNet checkpoint")
    output_root = Path(output_dir)
    _validate_inputs(
        dataset_key=dataset_key,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        spectral_modes=spectral_modes,
        spectral_ridge=spectral_ridge,
    )
    paths = _paths(output_root)
    if paths["comparison_report_path"].exists() and not force:
        raise OfficialOperatorComparisonError(
            f"comparison report already exists: {paths['comparison_report_path']}"
        )

    baseline = run_real_hdf5_baseline_smoke(
        source_path=source_file,
        output_dir=paths["baseline_dir"],
        dataset_key=dataset_key,
        start_index=0,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    windows_path = Path(baseline["artifacts"]["windows_npz_path"])
    spectral = _run_spectral(
        windows_path=windows_path,
        output_dir=paths["spectral_dir"],
        modes=spectral_modes,
        ridge=spectral_ridge,
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    fno = diagnose_fno_rollout(
        windows_path=windows_path,
        checkpoint_path=fno_checkpoint,
        output_dir=paths["fno_dir"],
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )
    unet = diagnose_unet_rollout(
        windows_path=windows_path,
        checkpoint_path=unet_checkpoint,
        output_dir=paths["unet_dir"],
        nu=nu,
        dt=dt,
        dx=dx,
        force=force,
    )

    persistence_score = float(baseline["metrics"]["competition_like_score_short"])
    records = [
        _record_from_baseline(baseline, persistence_score=persistence_score),
        _record_from_spectral(spectral, persistence_score=persistence_score),
        _record_from_diagnosis(
            model_id="official_fno",
            model_family="FNO",
            source_type="official_checkpoint",
            summary=fno,
            persistence_score=persistence_score,
        ),
        _record_from_diagnosis(
            model_id="official_unet",
            model_family="UNet",
            source_type="official_checkpoint",
            summary=unet,
            persistence_score=persistence_score,
        ),
    ]
    leaderboard = _leaderboard(records)
    best = leaderboard[0]
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "output_dir": str(output_root),
        "inputs": {
            "source_path": str(source_file),
            "dataset_key": dataset_key,
            "fno_checkpoint_path": str(fno_checkpoint),
            "unet_checkpoint_path": str(unet_checkpoint),
            "sample_limit": sample_limit,
            "context_steps": context_steps,
            "forecast_steps": forecast_steps,
            "time_reduction": time_reduction,
            "space_reduction": space_reduction,
            "spectral_modes": spectral_modes,
            "spectral_ridge": spectral_ridge,
            "nu": nu,
            "dt": dt,
            "dx": dx,
        },
        "model_count": len(records),
        "official_model_count": sum(
            record["source_type"] == "official_checkpoint" for record in records
        ),
        "best_model_id": best["model_id"],
        "best_model_family": best["model_family"],
        "best_score": best["score"],
        "best_score_delta_vs_persistence": best["score_delta_vs_persistence"],
        "model_records": records,
        "leaderboard": leaderboard,
        "recommendation": _recommendation(best),
        "artifacts": {
            "comparison_report_path": str(paths["comparison_report_path"]),
            "baseline_summary_path": str(
                paths["baseline_dir"] / "real_hdf5_baseline_smoke_summary.json"
            ),
            "spectral_summary_path": str(
                paths["spectral_dir"] / "spectral_residual_summary.json"
            ),
            "fno_summary_path": str(paths["fno_dir"] / "fno_rollout_summary.json"),
            "unet_summary_path": str(paths["unet_dir"] / "unet_rollout_summary.json"),
        },
        "next_command": _recommendation(best)["next_action"],
    }
    paths["comparison_report_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["comparison_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _run_spectral(
    *,
    windows_path: Path,
    output_dir: Path,
    modes: int,
    ridge: float,
    nu: float,
    dt: float,
    dx: float,
    force: bool,
) -> dict[str, Any]:
    paths = {
        "checkpoint_path": output_dir / "spectral_residual_checkpoint.npz",
        "prediction_path": output_dir / "spectral_residual_prediction.npz",
        "score_path": output_dir / "spectral_residual_score.json",
        "rollout_error_path": output_dir / "spectral_residual_rollout_error.json",
        "physics_path": output_dir / "spectral_residual_burgers_physics.json",
        "summary_path": output_dir / "spectral_residual_summary.json",
    }
    if paths["summary_path"].exists() and not force:
        raise OfficialOperatorComparisonError(
            f"spectral summary already exists: {paths['summary_path']}"
        )
    train = train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=paths["checkpoint_path"],
        modes=modes,
        ridge=ridge,
        force=force,
    )
    prediction = predict_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=paths["checkpoint_path"],
        output_path=paths["prediction_path"],
        force=force,
    )
    score = score_prediction_npz(
        prediction_path=paths["prediction_path"],
        target_path=windows_path,
        output_path=paths["score_path"],
    )
    error = diagnose_rollout_error_npz(
        prediction_path=paths["prediction_path"],
        target_path=windows_path,
        score_path=paths["score_path"],
        output_path=paths["rollout_error_path"],
    )
    physics = check_burgers_physics_npz(
        prediction_path=paths["prediction_path"],
        target_path=windows_path,
        output_path=paths["physics_path"],
        nu=nu,
        dt=dt,
        dx=dx,
    )
    summary = {
        "schema_version": 1,
        "state_hint": "SPECTRAL_RESIDUAL_DIAGNOSED",
        "model": {
            "model_type": "spectral_residual",
            "adapter_type": "spectral_residual_ridge",
            "modes": train["modes"],
            "ridge": train["ridge"],
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
        "score": {
            "competition_like_score_short": score["competition_like_score_short"],
            "rel_mse": score["rel_mse"],
            "rmse": score["rmse"],
        },
        "error_diagnosis": {
            "dominant_error_pattern": error["dominant_error_pattern"],
            "late_rel_mse": error["late_rel_mse"],
            "time_error_slope": error["time_error_slope"],
            "shock_region_error_ratio": error["shock_region_error_ratio"],
        },
        "physics": {
            "physics_issue": physics["physics_issue"],
            "residual_ratio": physics["residual_ratio"],
            "total_variation_ratio": physics["total_variation_ratio"],
            "gradient_error_ratio": physics["gradient_error_ratio"],
        },
        "prediction": prediction,
    }
    paths["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_path"].write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _record_from_baseline(
    summary: dict[str, Any],
    *,
    persistence_score: float,
) -> dict[str, Any]:
    metrics = summary["metrics"]
    return _record(
        model_id="persistence",
        model_family="Persistence",
        source_type="baseline",
        adapter_type="persistence",
        score=float(metrics["competition_like_score_short"]),
        rel_mse=float(metrics["rel_mse"]),
        rmse=float(metrics["rmse"]),
        dominant_error_pattern=str(metrics["dominant_error_pattern"]),
        physics_issue=str(metrics["physics_issue"]),
        artifacts=summary["artifacts"],
        persistence_score=persistence_score,
    )


def _record_from_spectral(
    summary: dict[str, Any],
    *,
    persistence_score: float,
) -> dict[str, Any]:
    return _record(
        model_id="spectral_residual",
        model_family="SpectralResidual",
        source_type="local_baseline",
        adapter_type=str(summary["model"]["adapter_type"]),
        score=float(summary["score"]["competition_like_score_short"]),
        rel_mse=float(summary["score"]["rel_mse"]),
        rmse=float(summary["score"]["rmse"]),
        dominant_error_pattern=str(
            summary["error_diagnosis"]["dominant_error_pattern"]
        ),
        physics_issue=str(summary["physics"]["physics_issue"]),
        artifacts=summary["artifacts"],
        persistence_score=persistence_score,
    )


def _record_from_diagnosis(
    *,
    model_id: str,
    model_family: str,
    source_type: str,
    summary: dict[str, Any],
    persistence_score: float,
) -> dict[str, Any]:
    return _record(
        model_id=model_id,
        model_family=model_family,
        source_type=source_type,
        adapter_type=str(summary["model"]["adapter_type"]),
        score=float(summary["score"]["competition_like_score_short"]),
        rel_mse=float(summary["score"]["rel_mse"]),
        rmse=float(summary["score"]["rmse"]),
        dominant_error_pattern=str(
            summary["error_diagnosis"]["dominant_error_pattern"]
        ),
        physics_issue=str(summary["physics"]["physics_issue"]),
        artifacts=summary["artifacts"],
        persistence_score=persistence_score,
    )


def _record(
    *,
    model_id: str,
    model_family: str,
    source_type: str,
    adapter_type: str,
    score: float,
    rel_mse: float,
    rmse: float,
    dominant_error_pattern: str,
    physics_issue: str,
    artifacts: dict[str, Any],
    persistence_score: float,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_family": model_family,
        "source_type": source_type,
        "adapter_type": adapter_type,
        "score": score,
        "score_delta_vs_persistence": score - persistence_score,
        "rel_mse": rel_mse,
        "rmse": rmse,
        "dominant_error_pattern": dominant_error_pattern,
        "physics_issue": physics_issue,
        "artifacts": artifacts,
    }


def _leaderboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            -float(record["score"]),
            float(record["rel_mse"]),
            str(record["model_id"]),
        ),
    )
    return [
        {
            "rank": rank,
            "model_id": record["model_id"],
            "model_family": record["model_family"],
            "source_type": record["source_type"],
            "score": record["score"],
            "score_delta_vs_persistence": record["score_delta_vs_persistence"],
            "rel_mse": record["rel_mse"],
            "rmse": record["rmse"],
            "dominant_error_pattern": record["dominant_error_pattern"],
            "physics_issue": record["physics_issue"],
        }
        for rank, record in enumerate(ranked, start=1)
    ]


def _recommendation(best: dict[str, Any]) -> dict[str, Any]:
    if (
        best["source_type"] == "official_checkpoint"
        and float(best["score_delta_vs_persistence"]) > 0.0
    ):
        return {
            "next_action": "promote_official_model_candidate",
            "rationale": (
                "The best official checkpoint beats persistence on the same local "
                "Burgers window."
            ),
        }
    return {
        "next_action": "continue_strategy_search",
        "rationale": (
            "No official checkpoint beats the available non-official baseline on "
            "this local window."
        ),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "baseline_dir": output_root / "baseline",
        "spectral_dir": output_root / "spectral",
        "fno_dir": output_root / "official_fno",
        "unet_dir": output_root / "official_unet",
        "comparison_report_path": output_root
        / "official_operator_comparison_report.json",
    }


def _validate_inputs(
    *,
    dataset_key: str,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    spectral_modes: int,
    spectral_ridge: float,
) -> None:
    if not dataset_key:
        raise OfficialOperatorComparisonError("dataset_key must be non-empty")
    for name, value in (
        ("sample_limit", sample_limit),
        ("context_steps", context_steps),
        ("forecast_steps", forecast_steps),
        ("time_reduction", time_reduction),
        ("space_reduction", space_reduction),
        ("spectral_modes", spectral_modes),
    ):
        if value < 1:
            raise OfficialOperatorComparisonError(f"{name} must be >= 1")
    if spectral_ridge < 0.0:
        raise OfficialOperatorComparisonError("spectral_ridge must be non-negative")


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise OfficialOperatorComparisonError(f"missing {label} path: {path}")
    if not path.is_file():
        raise OfficialOperatorComparisonError(f"{label} path is not a file: {path}")
    return path
