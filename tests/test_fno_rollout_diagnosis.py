import json
from pathlib import Path

import pytest

from pde_agent.core.fno_rollout_diagnosis import (
    FNORolloutDiagnosisError,
    diagnose_fno_rollout,
)

torch = pytest.importorskip("torch", reason="FNO rollout diagnosis tests require torch")
np = pytest.importorskip("numpy")


def _write_fixture_checkpoint(path: Path) -> None:
    torch.save(
        {
            "model_type": "fno_v0_fixture",
            "prediction_rule": "delta_extrapolation",
            "config": {"modes": 2, "width": 4},
            "model_state_dict": {"weight": torch.ones(1)},
        },
        path,
    )


def _write_windows(path: Path) -> tuple[object, object]:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.04 * np.cos(x) + 0.01 * sample
        for time_index in range(4):
            context[sample, time_index, :] = base + time_index * delta
        for time_index in range(3):
            target[sample, time_index, :] = base + (4 + time_index) * delta
    np.savez(path, context=context, target=target)
    return context, target


def test_diagnose_fno_rollout_writes_artifact_bundle(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    summary = diagnose_fno_rollout(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    expected_files = {
        "fno_prediction.npz",
        "fno_score.json",
        "fno_rollout_error.json",
        "fno_burgers_physics.json",
        "fno_rollout_summary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert summary["state_hint"] == "FNO_ROLLOUT_DIAGNOSED"
    assert summary["model"]["adapter_type"] == "fixture_delta_extrapolation"
    assert summary["score"]["competition_like_score_short"] == pytest.approx(100.0)
    assert "dominant_error_pattern" in summary["error_diagnosis"]
    assert "physics_issue" in summary["physics"]
    assert summary["recommendation"]["next_command"] == "select-pde-hypothesis"
    written = json.loads((output_dir / "fno_rollout_summary.json").read_text())
    assert written == summary


def test_diagnose_fno_rollout_rejects_existing_artifact_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    output_dir.mkdir()
    (output_dir / "fno_score.json").write_text("previous", encoding="utf-8")
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    with pytest.raises(FNORolloutDiagnosisError, match="already exists"):
        diagnose_fno_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )

    assert (output_dir / "fno_score.json").read_text(encoding="utf-8") == "previous"


def test_diagnose_fno_rollout_cleans_partial_artifacts_on_failure(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "diagnosis"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)
    output_dir.mkdir()
    marker_path = output_dir / "keep.txt"
    marker_path.write_text("not a diagnosis artifact", encoding="utf-8")

    with pytest.raises(Exception, match="dt must be > 0"):
        diagnose_fno_rollout(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=0.0,
            dx=1.0,
            force=False,
        )

    artifact_names = {
        "fno_prediction.npz",
        "fno_score.json",
        "fno_rollout_error.json",
        "fno_burgers_physics.json",
        "fno_rollout_summary.json",
    }
    for artifact_name in artifact_names:
        assert not (output_dir / artifact_name).exists()
    assert marker_path.read_text(encoding="utf-8") == "not a diagnosis artifact"
