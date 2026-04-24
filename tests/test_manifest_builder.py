import json
from pathlib import Path

from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.record_timing import record_timing
from pde_agent.core.manifest_builder import build_manifest


def test_build_manifest_contains_summary_and_completeness() -> None:
    manifest = build_manifest(
        run_id="run_001",
        config={
            "resolved_config_path": "config/resolved.json",
            "config_hash": "abc123",
        },
        artifacts={"rollout_metrics_path": "metrics/rollout.json"},
        summary={"one_step_nrmse": 0.1},
        completeness={
            "has_rollout": True,
            "has_spectrum": False,
            "has_physics_check": False,
            "has_timing": True,
        },
    )

    assert manifest["status"] == "ok"
    assert manifest["summary"]["one_step_nrmse"] == 0.1
    assert manifest["completeness"]["has_rollout"] is True


def test_eval_rollout_and_record_timing_write_backbone_files(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    for subdir in ("metrics", "plots", "submission", "artifacts"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    rollout_result = eval_rollout(run_id="run_001", run_root=str(run_root))
    timing_result = record_timing(run_id="run_001", run_root=str(run_root))

    rollout_path = Path(rollout_result["artifacts"]["rollout_metrics_path"])
    timing_path = Path(timing_result["artifacts"]["timing_metrics_path"])
    error_curve_path = Path(rollout_result["artifacts"]["error_curve_path"])
    prediction_hdf5_path = Path(rollout_result["artifacts"]["prediction_hdf5_path"])
    time_csv_path = Path(timing_result["artifacts"]["time_csv_path"])

    assert rollout_result["status"] == "ok"
    assert timing_result["status"] == "ok"
    assert rollout_path.exists()
    assert timing_path.exists()
    assert error_curve_path.exists()
    assert prediction_hdf5_path.exists()
    assert time_csv_path.exists()

    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))

    assert rollout_payload["one_step"]["nrmse"] == 0.031
    assert rollout_payload["long_horizon"]["nrmse"] == 0.087
    assert rollout_payload["prediction"]["path"] == str(prediction_hdf5_path)
    assert timing_payload["train_time_seconds"] == 12450.0
    assert timing_payload["inference_time_seconds"] == 61.2
    assert timing_payload["time_csv_path"] == str(time_csv_path)


def test_collect_artifacts_writes_manifest_json(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    for subdir in ("config", "metrics", "artifacts", "plots", "submission"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "prediction.hdf5").write_bytes(b"synthetic-pde")
    (run_root / "submission" / "time.csv").write_text("phase,value\nsynthetic,1\n", encoding="utf-8")
    (run_root / "metrics" / "rollout.json").write_text(
        json.dumps(
            {
                "one_step": {"nrmse": 0.031},
                "long_horizon": {"nrmse": 0.087},
                "curve": {"path": str(run_root / "plots" / "error_curve.png")},
                "prediction": {"path": str(run_root / "artifacts" / "prediction.hdf5")},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "metrics" / "timing.json").write_text(
        json.dumps(
            {
                "train_time_seconds": 12450.0,
                "inference_time_seconds": 61.2,
                "time_csv_path": str(run_root / "submission" / "time.csv"),
            }
        ),
        encoding="utf-8",
    )

    result = collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="abc123")
    manifest_path = Path(result["artifacts"]["artifact_manifest_path"])

    assert result["status"] == "ok"
    assert manifest_path.exists()
    assert result["metrics"]["artifact_count"] == 5

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["config_hash"] == "abc123"
    assert manifest["completeness"]["has_rollout"] is True
    assert manifest["completeness"]["has_timing"] is True
