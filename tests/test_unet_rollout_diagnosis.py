import json
from pathlib import Path

import pytest
import pde_agent.core.unet_rollout_diagnosis as unet_rollout_diagnosis

from pde_agent.core.unet_rollout_diagnosis import (
    UNetRolloutDiagnosisError,
    diagnose_unet_rollout,
)
from tests.test_unet_backend_v0 import _official_unet_state_dict, _write_windows

torch = pytest.importorskip("torch", reason="UNet rollout diagnosis tests require torch")


def _write_checkpoint(path: Path) -> None:
    torch.save({"model_state_dict": _official_unet_state_dict(torch=torch)}, path)


def test_diagnose_unet_rollout_writes_artifact_bundle(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    summary = diagnose_unet_rollout(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    expected_files = {
        "unet_prediction.npz",
        "unet_score.json",
        "unet_rollout_error.json",
        "unet_burgers_physics.json",
        "unet_rollout_summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert summary["state_hint"] == "UNET_ROLLOUT_DIAGNOSED"
    assert summary["model"]["adapter_type"] == "pdebench_unet1d"
    assert "competition_like_score_short" in summary["score"]
    assert "dominant_error_pattern" in summary["error_diagnosis"]
    assert "physics_issue" in summary["physics"]
    written = json.loads((output_dir / "unet_rollout_summary.json").read_text())
    assert written == summary


def test_diagnose_unet_rollout_rejects_existing_artifact_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_dir = tmp_path / "diagnosis"
    output_dir.mkdir()
    (output_dir / "unet_score.json").write_text("previous", encoding="utf-8")
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    with pytest.raises(UNetRolloutDiagnosisError, match="already exists"):
        diagnose_unet_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )

    assert (output_dir / "unet_score.json").read_text(encoding="utf-8") == "previous"


def test_diagnose_unet_rollout_cleans_partial_artifacts_on_failure(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)
    output_dir.mkdir()
    marker_path = output_dir / "keep.txt"
    marker_path.write_text("not a diagnosis artifact", encoding="utf-8")

    with pytest.raises(Exception, match="dt must be > 0"):
        diagnose_unet_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=0.0,
            dx=1.0,
            force=False,
        )

    for artifact_name in {
        "unet_prediction.npz",
        "unet_score.json",
        "unet_rollout_error.json",
        "unet_burgers_physics.json",
        "unet_rollout_summary.json",
    }:
        assert not (output_dir / artifact_name).exists()
    assert marker_path.read_text(encoding="utf-8") == "not a diagnosis artifact"


def test_diagnose_unet_rollout_force_failure_preserves_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_dir = tmp_path / "diagnosis"
    output_dir.mkdir()
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)
    score_path = output_dir / "unet_score.json"
    summary_path = output_dir / "unet_rollout_summary.json"
    score_path.write_text("old score", encoding="utf-8")
    summary_path.write_text("old summary", encoding="utf-8")

    def fail_before_writing(**_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        unet_rollout_diagnosis,
        "predict_unet_local",
        fail_before_writing,
    )

    with pytest.raises(RuntimeError, match="boom"):
        diagnose_unet_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=True,
        )

    assert score_path.read_text(encoding="utf-8") == "old score"
    assert summary_path.read_text(encoding="utf-8") == "old summary"


def test_diagnose_unet_rollout_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "unet.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "diagnose-unet-rollout",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "diagnose-unet-rollout"
    assert payload["state_hint"] == "UNET_ROLLOUT_DIAGNOSED"
    assert payload["artifacts"]["unet_rollout_summary_json_path"] == str(
        output_dir / "unet_rollout_summary.json"
    )
    assert payload["metrics"]["adapter_type"] == "pdebench_unet1d"
    assert payload["next_hint"] == "select-pde-hypothesis"
