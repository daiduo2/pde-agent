"""Orchestrate local official UNet rollout diagnostics into one artifact bundle."""


import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from pde_agent.core.burgers_physics_local import check_burgers_physics_npz
from pde_agent.core.local_scoring import score_prediction_npz
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz
from pde_agent.core.unet_backend import predict_unet_local


class UNetRolloutDiagnosisError(RuntimeError):
    pass


def diagnose_unet_rollout(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_root = Path(output_dir)
    paths = _artifact_paths(output_root)
    _prepare_output_paths(paths=paths, force=force)
    preexisting_artifacts = {path for path in paths.values() if path.exists()}

    try:
        prediction_summary = predict_unet_local(
            windows_path=windows_file,
            checkpoint_path=checkpoint_file,
            output_path=paths["prediction_npz_path"],
            metadata_path=None,
            force=force,
        )
        score = score_prediction_npz(
            prediction_path=paths["prediction_npz_path"],
            target_path=windows_file,
            output_path=paths["score_json_path"],
        )
        error = diagnose_rollout_error_npz(
            prediction_path=paths["prediction_npz_path"],
            target_path=windows_file,
            score_path=paths["score_json_path"],
            output_path=paths["rollout_error_json_path"],
        )
        physics = check_burgers_physics_npz(
            prediction_path=paths["prediction_npz_path"],
            target_path=windows_file,
            output_path=paths["burgers_physics_json_path"],
            nu=nu,
            dt=dt,
            dx=dx,
        )
        summary = _build_summary(
            prediction_summary=prediction_summary,
            score=score,
            error=error,
            physics=physics,
            paths=paths,
        )
        summary["summary_path"] = str(paths["summary_json_path"])
        paths["summary_json_path"].write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
    except Exception:
        _cleanup_artifacts(paths, preexisting_artifacts=preexisting_artifacts)
        raise


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "prediction_npz_path": output_dir / "unet_prediction.npz",
        "score_json_path": output_dir / "unet_score.json",
        "rollout_error_json_path": output_dir / "unet_rollout_error.json",
        "burgers_physics_json_path": output_dir / "unet_burgers_physics.json",
        "summary_json_path": output_dir / "unet_rollout_summary.json",
    }


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise UNetRolloutDiagnosisError(
            f"diagnosis artifact already exists: {names}"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _cleanup_artifacts(
    paths: dict[str, Path],
    *,
    preexisting_artifacts: set[Path],
) -> None:
    for path in paths.values():
        if path in preexisting_artifacts:
            continue
        with suppress(OSError):
            if path.exists():
                path.unlink()


def _build_summary(
    *,
    prediction_summary: dict[str, Any],
    score: dict[str, Any],
    error: dict[str, Any],
    physics: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    model = {
        "model_type": prediction_summary["model_type"],
        "adapter_type": prediction_summary.get("adapter_type", "unknown"),
        "checkpoint_path": prediction_summary["checkpoint_path"],
        "checkpoint_format": prediction_summary["checkpoint_format"],
    }
    for key in ("archive_member", "history_steps", "base_channels", "depth"):
        if key in prediction_summary:
            model[key] = prediction_summary[key]
    return {
        "schema_version": 1,
        "state_hint": "UNET_ROLLOUT_DIAGNOSED",
        "model": model,
        "artifacts": {
            "prediction_npz_path": str(paths["prediction_npz_path"]),
            "score_json_path": str(paths["score_json_path"]),
            "rollout_error_json_path": str(paths["rollout_error_json_path"]),
            "burgers_physics_json_path": str(paths["burgers_physics_json_path"]),
        },
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
        "recommendation": {
            "next_command": "select-pde-hypothesis",
            "rationale": (
                "Use existing hypothesis selection with this score, error diagnosis, "
                "and physics check."
            ),
        },
    }


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise UNetRolloutDiagnosisError(f"missing {label} path: {path}")
    if not path.is_file():
        raise UNetRolloutDiagnosisError(f"{label} path is not a file: {path}")
    return path
