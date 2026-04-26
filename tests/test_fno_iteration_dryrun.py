import json
from pathlib import Path

import pytest

from pde_agent.core.fno_iteration_dryrun import (
    FNOIterationDryRunError,
    run_fno_iteration_dryrun,
)

torch = pytest.importorskip("torch", reason="FNO dry-run tests require torch")
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


def _write_windows(path: Path) -> None:
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


def test_run_fno_iteration_dryrun_writes_full_artifact_bundle(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "fno_iteration"
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    manifest = run_fno_iteration_dryrun(
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
        "selected_pde_hypothesis.json",
        "next_experiment_plan.json",
        "experiment_config.json",
        "iteration_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert manifest["state_hint"] == "FNO_ITERATION_DRYRUN_COMPLETED"
    assert manifest["artifacts"]["iteration_manifest_path"] == str(
        output_dir / "iteration_manifest.json"
    )
    assert manifest["artifacts"]["experiment_config_path"] == str(
        output_dir / "experiment_config.json"
    )
    assert manifest["metrics"]["candidate_count"] >= 1
    assert manifest["metrics"]["selected_hypothesis"]
    assert manifest["metrics"]["selected_intervention"]
    assert manifest["next_command"] == "execute-candidate-experiment"
    written = json.loads((output_dir / "iteration_manifest.json").read_text())
    assert written == manifest


def test_run_fno_iteration_dryrun_rejects_existing_artifact_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "fixture_fno.pt"
    output_dir = tmp_path / "fno_iteration"
    output_dir.mkdir()
    existing_manifest = output_dir / "iteration_manifest.json"
    existing_manifest.write_text("previous", encoding="utf-8")
    _write_windows(windows_path)
    _write_fixture_checkpoint(checkpoint_path)

    with pytest.raises(FNOIterationDryRunError, match="already exists"):
        run_fno_iteration_dryrun(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )

    assert existing_manifest.read_text(encoding="utf-8") == "previous"
