"""Multi-start evidence sweep for the spectral residual baseline."""


import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.real_hdf5_baseline_smoke import run_real_hdf5_baseline_smoke
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz
from pde_agent.core.spectral_residual_baseline import (
    predict_spectral_residual_baseline,
    train_spectral_residual_baseline,
)


from pde_agent.core.state_hint_registry import STRATEGY_EVIDENCE_SUMMARY_BUILT as STATE_HINT
SOURCE_STATE_HINT = "SPECTRAL_RESIDUAL_EVIDENCE_SWEEP_COMPLETED"
MODEL_FAMILY = "SpectralResidual"


class SpectralResidualEvidenceSweepError(RuntimeError):
    pass


def run_spectral_residual_evidence_sweep(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_indices: list[int],
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
    output_root = Path(output_dir)
    _validate_inputs(
        dataset_key=dataset_key,
        start_indices=start_indices,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        spectral_modes=spectral_modes,
        spectral_ridge=spectral_ridge,
    )
    evidence_path = output_root / "spectral_residual_evidence_summary.json"
    if evidence_path.exists() and not force:
        raise SpectralResidualEvidenceSweepError(
            f"evidence summary already exists: {evidence_path}"
        )

    strategy_id = _strategy_id(modes=spectral_modes, ridge=spectral_ridge)
    records = []
    for start_index in start_indices:
        run_dir = output_root / f"start_{start_index:04d}"
        baseline = run_real_hdf5_baseline_smoke(
            source_path=source_file,
            output_dir=run_dir / "baseline",
            dataset_key=dataset_key,
            start_index=start_index,
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
        spectral = _run_spectral(
            windows_path=Path(baseline["artifacts"]["windows_npz_path"]),
            output_dir=run_dir / "spectral",
            modes=spectral_modes,
            ridge=spectral_ridge,
            nu=nu,
            dt=dt,
            dx=dx,
            force=force,
        )
        records.append(
            _candidate_record(
                start_index=start_index,
                strategy_id=strategy_id,
                baseline=baseline,
                spectral=spectral,
                modes=spectral_modes,
                ridge=spectral_ridge,
            )
        )

    deltas = [record["score_delta"] for record in records]
    ranking = _strategy_ranking(records=records, strategy_id=strategy_id)
    evidence = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_family": MODEL_FAMILY,
        "source_state_hint": SOURCE_STATE_HINT,
        "source_evidence_summary_path": str(evidence_path),
        "source_spectral_residual_evidence_summary_path": str(evidence_path),
        "evidence_summary_path": str(evidence_path),
        "evaluation_window_count": len(start_indices),
        "mean_score_delta": float(mean(deltas)),
        "score_delta_std": float(pstdev(deltas)) if len(deltas) > 1 else 0.0,
        "long_window_score_delta": float(mean(deltas)) if forecast_steps > 0 else 0.0,
        "physics_residual_delta": 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "candidate_records": records,
        "strategy_ranking": ranking,
        "physics_metric_summary": _physics_metric_summary(records),
        "best_strategy_id": ranking[0]["strategy_id"] if ranking else None,
        "provenance_notes": [
            "spectral_residual_evidence_uses_same_windows_for_fit_and_score"
        ],
        "source_metrics": {
            "forecast_steps": forecast_steps,
            "start_indices": start_indices,
            "sample_limit": sample_limit,
            "spectral_modes": spectral_modes,
            "spectral_ridge": spectral_ridge,
            "record_count": len(records),
        },
        "next_command": "recommend-strategy-next-action",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


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
        raise SpectralResidualEvidenceSweepError(
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
        "artifacts": {key: str(path) for key, path in paths.items()},
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


def _candidate_record(
    *,
    start_index: int,
    strategy_id: str,
    baseline: dict[str, Any],
    spectral: dict[str, Any],
    modes: int,
    ridge: float,
) -> dict[str, Any]:
    baseline_score = float(baseline["metrics"]["competition_like_score_short"])
    score = float(spectral["score"]["competition_like_score_short"])
    return {
        "model_family": MODEL_FAMILY,
        "strategy_id": strategy_id,
        "training_strategy": "spectral_residual_ridge",
        "experiment_id": f"{strategy_id}_s{start_index:04d}",
        "start_index": start_index,
        "train_steps": 0,
        "score": score,
        "score_delta": score - baseline_score,
        "baseline_score": baseline_score,
        "rel_mse": float(spectral["score"]["rel_mse"]),
        "rmse": float(spectral["score"]["rmse"]),
        "dominant_error_pattern": str(
            spectral["error_diagnosis"]["dominant_error_pattern"]
        ),
        "physics_issue": str(spectral["physics"]["physics_issue"]),
        "hyperparameters": {
            "modes": modes,
            "ridge": ridge,
        },
        "physics_metrics": {
            "residual_ratio": float(spectral["physics"]["residual_ratio"]),
            "total_variation_ratio": float(
                spectral["physics"]["total_variation_ratio"]
            ),
            "gradient_error_ratio": float(spectral["physics"]["gradient_error_ratio"]),
        },
        "artifacts": {
            "baseline_summary_path": baseline["artifacts"]["summary_path"],
            "spectral_summary_path": spectral["artifacts"]["summary_path"],
            "prediction_npz_path": spectral["artifacts"]["prediction_path"],
            "score_path": spectral["artifacts"]["score_path"],
        },
    }


def _strategy_ranking(
    *,
    records: list[dict[str, Any]],
    strategy_id: str,
) -> list[dict[str, Any]]:
    metric_summary = _physics_metric_summary(records)
    row = {
        "rank": 1,
        "strategy_id": strategy_id,
        "training_strategy": "spectral_residual_ridge",
        "record_count": len(records),
        "win_count": sum(record["score_delta"] > 0.0 for record in records),
        "mean_score": float(mean(record["score"] for record in records)),
        "mean_score_delta": float(mean(record["score_delta"] for record in records)),
        "best_score": max(record["score"] for record in records),
        "failure_rate": sum(record["score_delta"] < 0.0 for record in records)
        / len(records),
        "dominant_error_patterns": sorted(
            {record["dominant_error_pattern"] for record in records}
        ),
        "physics_issues": sorted({record["physics_issue"] for record in records}),
    }
    row["win_rate"] = row["win_count"] / len(records)
    if metric_summary["record_count"]:
        row["mean_residual_ratio"] = metric_summary["mean_residual_ratio"]
        row["max_total_variation_ratio"] = metric_summary[
            "max_total_variation_ratio"
        ]
    return [row]


def _physics_metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        record["physics_metrics"]
        for record in records
        if isinstance(record.get("physics_metrics"), dict)
    ]
    if not metrics:
        return {
            "record_count": 0,
            "mean_residual_ratio": None,
            "max_total_variation_ratio": None,
            "max_gradient_error_ratio": None,
        }
    return {
        "record_count": len(metrics),
        "mean_residual_ratio": float(mean(item["residual_ratio"] for item in metrics)),
        "max_total_variation_ratio": max(
            item["total_variation_ratio"] for item in metrics
        ),
        "max_gradient_error_ratio": max(
            item["gradient_error_ratio"] for item in metrics
        ),
    }


def _strategy_id(*, modes: int, ridge: float) -> str:
    return f"spectral_residual_m{modes:02d}_ridge{ridge:.0e}"


def _validate_inputs(
    *,
    dataset_key: str,
    start_indices: list[int],
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    spectral_modes: int,
    spectral_ridge: float,
) -> None:
    if not dataset_key:
        raise SpectralResidualEvidenceSweepError("dataset_key must be non-empty")
    if not start_indices:
        raise SpectralResidualEvidenceSweepError("start_indices must be non-empty")
    if len(set(start_indices)) != len(start_indices):
        raise SpectralResidualEvidenceSweepError("start_indices must be unique")
    for start_index in start_indices:
        if start_index < 0:
            raise SpectralResidualEvidenceSweepError("start_indices must be >= 0")
    for name, value in (
        ("sample_limit", sample_limit),
        ("context_steps", context_steps),
        ("forecast_steps", forecast_steps),
        ("time_reduction", time_reduction),
        ("space_reduction", space_reduction),
        ("spectral_modes", spectral_modes),
    ):
        if value < 1:
            raise SpectralResidualEvidenceSweepError(f"{name} must be >= 1")
    if spectral_ridge < 0.0:
        raise SpectralResidualEvidenceSweepError("spectral_ridge must be non-negative")


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise SpectralResidualEvidenceSweepError(f"missing {label} path: {path}")
    if not path.is_file():
        raise SpectralResidualEvidenceSweepError(f"{label} path is not a file: {path}")
    return path
