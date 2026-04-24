import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.record_timing import record_timing


def _make_run_tree(tmp_path: Path, run_id: str = "run_001") -> Path:
    run_root = tmp_path / run_id
    for subdir in [
        "config",
        "metrics",
        "plots",
        "artifacts",
        "submission",
    ]:
        (run_root / subdir).mkdir(parents=True, exist_ok=True)
    return run_root


def test_eval_rollout_writes_phase2_synthetic_outputs(tmp_path: Path) -> None:
    run_root = _make_run_tree(tmp_path)

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
    assert rollout_payload["prediction_hdf5_path"] == str(prediction_hdf5_path)


def test_record_timing_writes_phase2_synthetic_outputs(tmp_path: Path) -> None:
    run_root = _make_run_tree(tmp_path)

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


def test_collect_artifacts_copies_phase2_synthetic_summary(tmp_path: Path) -> None:
    run_root = _make_run_tree(tmp_path)

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
        "prediction_hdf5_path": str(prediction_hdf5_path),
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
