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
    for subdir in ("metrics", "plots", "submission"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    rollout_result = eval_rollout(run_id="run_001", run_root=str(run_root))
    timing_result = record_timing(run_id="run_001", run_root=str(run_root))

    rollout_path = Path(rollout_result["artifacts"]["rollout_metrics_path"])
    timing_path = Path(timing_result["artifacts"]["timing_metrics_path"])

    assert rollout_result["status"] == "ok"
    assert timing_result["status"] == "ok"
    assert rollout_path.exists()
    assert timing_path.exists()

    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))

    assert rollout_payload["one_step"]["nrmse"] == 0.0
    assert rollout_payload["long_horizon"]["nrmse"] == 0.0
    assert timing_payload["train_time_seconds"] == 0.0
    assert timing_payload["inference_time_seconds"] == 0.0


def test_collect_artifacts_writes_manifest_json(tmp_path: Path) -> None:
    run_root = tmp_path / "run_001"
    for subdir in ("config", "metrics", "artifacts"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")
    (run_root / "metrics" / "rollout.json").write_text("{}", encoding="utf-8")
    (run_root / "metrics" / "timing.json").write_text("{}", encoding="utf-8")

    result = collect_artifacts(run_id="run_001", run_root=str(run_root), config_hash="abc123")
    manifest_path = Path(result["artifacts"]["artifact_manifest_path"])

    assert result["status"] == "ok"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["config_hash"] == "abc123"
    assert manifest["completeness"]["has_rollout"] is True
    assert manifest["completeness"]["has_timing"] is True
