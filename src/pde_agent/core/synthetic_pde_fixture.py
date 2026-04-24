from pathlib import Path


SYNTHETIC_PDE_MODE = "synthetic_pde_fixture"
SYNTHETIC_ONE_STEP_NRMSE = 0.031
SYNTHETIC_LONG_HORIZON_NRMSE = 0.087
SYNTHETIC_TRAIN_TIME_SECONDS = 12450.0
SYNTHETIC_INFERENCE_TIME_SECONDS = 61.2


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_placeholder_file(path: Path, content: bytes) -> str:
    ensure_parent(path).write_bytes(content)
    return str(path)


def synthetic_rollout_payload(run_id: str, run_root: Path) -> dict[str, object]:
    curve_path = run_root / "plots" / "error_curve.png"
    prediction_path = run_root / "artifacts" / "prediction.hdf5"
    write_placeholder_file(curve_path, b"synthetic-error-curve")
    write_placeholder_file(prediction_path, b"synthetic-hdf5-placeholder")
    return {
        "run_id": run_id,
        "one_step": {"nrmse": SYNTHETIC_ONE_STEP_NRMSE},
        "long_horizon": {"nrmse": SYNTHETIC_LONG_HORIZON_NRMSE},
        "curve": {"path": str(curve_path)},
        "prediction": {"path": str(prediction_path)},
        "mode": SYNTHETIC_PDE_MODE,
    }


def synthetic_timing_payload(run_id: str, run_root: Path) -> dict[str, object]:
    time_csv_path = run_root / "submission" / "time.csv"
    ensure_parent(time_csv_path).write_text(
        "train_time,inference_time\n12450.0,61.2\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "train_time_seconds": SYNTHETIC_TRAIN_TIME_SECONDS,
        "inference_time_seconds": SYNTHETIC_INFERENCE_TIME_SECONDS,
        "time_csv_path": str(time_csv_path),
        "mode": SYNTHETIC_PDE_MODE,
    }
