import json
from pathlib import Path

import pytest
import numpy as np

from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.record_timing import record_timing
from pde_agent.core.run_layout import create_run_tree


def test_eval_rollout_writes_phase2_synthetic_outputs(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")

    result = eval_rollout(run_id="run_001", run_root=str(run_root))

    rollout_path = run_root / "metrics" / "rollout.json"
    error_curve_path = run_root / "plots" / "error_curve.png"
    prediction_hdf5_path = run_root / "artifacts" / "prediction.hdf5"

    assert result["artifacts"]["rollout_metrics_path"] == str(rollout_path)
    assert rollout_path.exists()
    assert error_curve_path.exists()
    assert prediction_hdf5_path.exists()

    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    assert rollout_payload["mode"] == "synthetic_pde_fixture"
    assert rollout_payload["one_step"]["nrmse"] == 0.031
    assert rollout_payload["long_horizon"]["nrmse"] == 0.087
    assert rollout_payload["curve"]["path"] == str(error_curve_path)
    assert rollout_payload["prediction"]["path"] == str(prediction_hdf5_path)


def test_eval_rollout_creates_metrics_when_only_run_root_exists(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    run_root.mkdir()

    result = eval_rollout(run_id="run_001", run_root=str(run_root))

    rollout_path = run_root / "metrics" / "rollout.json"
    error_curve_path = run_root / "plots" / "error_curve.png"
    prediction_hdf5_path = run_root / "artifacts" / "prediction.hdf5"

    assert result["artifacts"]["rollout_metrics_path"] == str(rollout_path)
    assert rollout_path.exists()
    assert error_curve_path.exists()
    assert prediction_hdf5_path.exists()


def test_eval_rollout_writes_real_npz_metrics_when_configured(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.npz"
    target = np.ones((2, 4, 5), dtype=np.float64)
    prediction = target * 1.1
    np.savez(
        source_path,
        context=np.zeros((2, 3, 5), dtype=np.float64),
        prediction=prediction,
        target=target,
    )
    run_root = create_run_tree(tmp_path, "run_real_001")
    config_path = run_root / "config" / "resolved.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(source_path),
                    "context_steps": 3,
                    "forecast_steps": 4,
                }
            }
        ),
        encoding="utf-8",
    )

    result = eval_rollout(run_id="run_real_001", run_root=str(run_root))

    rollout_path = run_root / "metrics" / "rollout.json"
    error_curve_path = run_root / "metrics" / "error_curve.json"
    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    error_curve = json.loads(error_curve_path.read_text(encoding="utf-8"))

    assert result["metrics"]["source_mode"] == "npz_fixture"
    assert result["metrics"]["one_step_nrmse"] == pytest.approx(0.1)
    assert result["metrics"]["long_horizon_nrmse"] == pytest.approx(0.1)
    assert result["artifacts"]["error_curve_path"] == str(error_curve_path)
    assert result["artifacts"]["prediction_hdf5_path"] == str(source_path)
    assert rollout_payload["mode"] == "real_array_diagnostic"
    assert rollout_payload["source_mode"] == "npz_fixture"
    assert rollout_payload["run_id"] == "run_real_001"
    assert rollout_payload["curve"]["path"] == str(error_curve_path)
    assert rollout_payload["prediction"]["path"] == str(source_path)
    assert error_curve == rollout_payload["curve"]


def test_record_timing_writes_phase2_synthetic_outputs(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")

    result = record_timing(run_id="run_001", run_root=str(run_root))

    timing_path = run_root / "metrics" / "timing.json"
    time_csv_path = run_root / "submission" / "time.csv"

    assert result["artifacts"]["timing_metrics_path"] == str(timing_path)
    assert result["artifacts"]["time_csv_path"] == str(time_csv_path)
    assert timing_path.exists()
    assert time_csv_path.exists()

    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    assert timing_payload["mode"] == "synthetic_pde_fixture"
    assert timing_payload["train_time_seconds"] == 12450.0
    assert timing_payload["inference_time_seconds"] == 61.2
    assert timing_payload["time_csv_path"] == str(time_csv_path)


def test_record_timing_creates_metrics_when_only_run_root_exists(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    run_root.mkdir()

    result = record_timing(run_id="run_001", run_root=str(run_root))

    timing_path = run_root / "metrics" / "timing.json"
    time_csv_path = run_root / "submission" / "time.csv"

    assert result["artifacts"]["timing_metrics_path"] == str(timing_path)
    assert result["artifacts"]["time_csv_path"] == str(time_csv_path)
    assert timing_path.exists()
    assert time_csv_path.exists()


def test_collect_artifacts_copies_phase2_synthetic_summary(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")

    rollout_path = run_root / "metrics" / "rollout.json"
    timing_path = run_root / "metrics" / "timing.json"
    prediction_hdf5_path = run_root / "artifacts" / "prediction.hdf5"
    time_csv_path = run_root / "submission" / "time.csv"

    prediction_hdf5_path.write_bytes(b"synthetic-pde")
    time_csv_path.write_text("phase,value\nsynthetic,1\n", encoding="utf-8")
    rollout_payload = {
        "mode": "synthetic_pde_fixture",
        "one_step": {"nrmse": 0.031},
        "long_horizon": {"nrmse": 0.087},
        "curve": {"path": str(run_root / "plots" / "error_curve.png")},
        "prediction": {"path": str(prediction_hdf5_path)},
    }
    timing_payload = {
        "mode": "synthetic_pde_fixture",
        "train_time_seconds": 12450.0,
        "inference_time_seconds": 61.2,
        "time_csv_path": str(time_csv_path),
    }
    rollout_path.write_text(json.dumps(rollout_payload, indent=2), encoding="utf-8")
    timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")

    result = collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="abc123")

    manifest_path = Path(result["artifacts"]["artifact_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["phase"] == "results_collected"
    assert manifest["summary"]["one_step_nrmse"] == 0.031
    assert manifest["summary"]["long_horizon_nrmse"] == 0.087
    assert manifest["summary"]["train_time_seconds"] == 12450.0
    assert manifest["summary"]["inference_time_seconds"] == 61.2
    assert manifest["artifacts"]["prediction_hdf5_path"] == str(prediction_hdf5_path)
    assert manifest["artifacts"]["time_csv_path"] == str(time_csv_path)
    assert manifest["completeness"]["has_data_inspection"] is False
    assert result["metrics"]["artifact_count"] == len(manifest["artifacts"])


def test_collect_artifacts_fails_on_missing_required_rollout_fields(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")
    rollout_path = run_root / "metrics" / "rollout.json"
    timing_path = run_root / "metrics" / "timing.json"
    time_csv_path = run_root / "submission" / "time.csv"

    time_csv_path.write_text("phase,value\nsynthetic,1\n", encoding="utf-8")
    rollout_path.write_text(
        json.dumps(
            {
                "mode": "synthetic_pde_fixture",
                "curve": {"path": str(run_root / "plots" / "error_curve.png")},
                "prediction": {"path": str(run_root / "artifacts" / "prediction.hdf5")},
                "long_horizon": {"nrmse": 0.087},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    timing_path.write_text(
        json.dumps(
            {
                "mode": "synthetic_pde_fixture",
                "train_time_seconds": 12450.0,
                "inference_time_seconds": 61.2,
                "time_csv_path": str(time_csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="abc123")


def test_collect_artifacts_fails_on_missing_required_timing_fields(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")
    rollout_path = run_root / "metrics" / "rollout.json"
    timing_path = run_root / "metrics" / "timing.json"
    prediction_hdf5_path = run_root / "artifacts" / "prediction.hdf5"

    prediction_hdf5_path.write_bytes(b"synthetic-pde")
    rollout_path.write_text(
        json.dumps(
            {
                "mode": "synthetic_pde_fixture",
                "one_step": {"nrmse": 0.031},
                "long_horizon": {"nrmse": 0.087},
                "curve": {"path": str(run_root / "plots" / "error_curve.png")},
                "prediction": {"path": str(prediction_hdf5_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    timing_path.write_text(
        json.dumps(
            {
                "mode": "synthetic_pde_fixture",
                "inference_time_seconds": 61.2,
                "time_csv_path": str(run_root / "submission" / "time.csv"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="abc123")
