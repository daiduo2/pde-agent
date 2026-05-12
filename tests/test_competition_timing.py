import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _write_npz(path: Path, prediction: np.ndarray) -> Path:
    np.savez(path, prediction=prediction)
    return path


def test_mac_mini_m4_profile_converts_raw_seconds_to_a100_equivalent() -> None:
    from pde_agent.core.competition_timing import normalize_timing

    timing = normalize_timing(
        raw_train_time=200.0,
        raw_inference_time=121.0,
        timing_profile="mac_mini_m4_local_dev",
    )

    assert timing["timing_profile"] == "mac_mini_m4_local_dev"
    assert timing["timing_factor"] == pytest.approx(0.03)
    assert timing["raw_train_time"] == pytest.approx(200.0)
    assert timing["raw_inference_time"] == pytest.approx(121.0)
    assert timing["train_time"] == pytest.approx(6.0)
    assert timing["inference_time"] == pytest.approx(3.63)
    assert timing["timing_valid_for_submission"] is False


def test_build_submission_writes_a100_equivalent_time_csv_for_timing_profile(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission

    context = np.ones((1, 10, 256), dtype=np.float32)
    future = np.ones((1, 190, 256), dtype=np.float32)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    source_path = _write_npz(tmp_path / "future.npz", future)

    report = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "submission_out",
        submission_id="team_default",
        method="external_prediction",
        source_prediction_path=source_path,
        train_time=200.0,
        inference_time=121.0,
        timing_profile="mac_mini_m4_local_dev",
        force=True,
    )

    rows = list(
        csv.DictReader(Path(report["artifacts"]["time_csv_path"]).open(encoding="utf-8"))
    )
    assert float(rows[0]["train_time"]) == pytest.approx(6.0)
    assert float(rows[0]["inference_time"]) == pytest.approx(3.63)
    assert report["metrics"]["raw_train_time"] == pytest.approx(200.0)
    assert report["metrics"]["raw_inference_time"] == pytest.approx(121.0)
    assert report["metrics"]["timing_factor"] == pytest.approx(0.03)
    assert report["metrics"]["timing_valid_for_submission"] is False


def test_checkpoint_family_sandbox_scores_with_a100_equivalent_time(
    tmp_path: Path,
) -> None:
    from pde_agent.core.checkpoint_family_sandbox import run_checkpoint_family_sandbox

    context = np.ones((1, 10, 256), dtype=np.float32)
    target = np.ones((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_target.hdf5", target)
    checkpoint = tmp_path / "burgers_Unet.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_unet_runner(**kwargs: Any) -> dict[str, Any]:
        return _fake_candidate_report(
            output_dir=Path(kwargs["output_dir"]),
            family="UNet",
            value=1.0,
            inference_seconds=121.0,
        )

    report = run_checkpoint_family_sandbox(
        checkpoint_paths=[checkpoint],
        task_id=1,
        test_path=test_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_default",
        calibration_weights=[1.0],
        unet_runner=fake_unet_runner,
        timing_profile="mac_mini_m4_local_dev",
        force=False,
    )

    best = report["best_candidate"]
    assert best["score"] > 0.0
    assert best["raw_inference_time"] == pytest.approx(121.0, abs=0.01)
    assert best["inference_time"] == pytest.approx(3.63, abs=0.01)
    assert best["timing_factor"] == pytest.approx(0.03)
    assert best["timeout_zeroed"] is False


def _fake_candidate_report(
    *,
    output_dir: Path,
    family: str,
    value: float,
    inference_seconds: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "candidate_prediction.npz"
    prediction = np.full((1, 190, 256), value, dtype=np.float32)
    np.savez_compressed(prediction_path, prediction=prediction)
    report_path = output_dir / "candidate_execution_report.json"
    report = {
        "state_hint": f"{family.upper()}_COMPETITION_PREDICTION_READY",
        "candidate": {
            "experiment_id": f"{family.lower()}_raw",
            "model_family": family,
            "task_id": 1,
        },
        "artifacts": {
            "candidate_prediction_path": str(prediction_path),
            "candidate_execution_report_path": str(report_path),
        },
        "train": {"mode": "official_checkpoint_inference", "elapsed_seconds": 0.0},
        "inference": {"elapsed_seconds": inference_seconds},
    }
    report_path.write_text(str(report), encoding="utf-8")
    return report
