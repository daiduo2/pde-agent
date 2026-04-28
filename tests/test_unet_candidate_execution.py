import json
from pathlib import Path

from pde_agent.core.unet_candidate_execution import execute_unet_candidate


def _write_windows(path: Path) -> None:
    import numpy as np

    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    context = np.zeros((3, 4, 12), dtype=np.float32)
    target = np.zeros((3, 5, 12), dtype=np.float32)
    for sample in range(3):
        base = np.sin(2.0 * np.pi * (x + 0.02 * sample))
        trend = 0.03 * np.cos(2.0 * np.pi * x) + 0.01 * sample
        for time_index in range(context.shape[1]):
            context[sample, time_index, :] = base + time_index * trend
        for time_index in range(target.shape[1]):
            target[sample, time_index, :] = base + (time_index + 4) * trend
    np.savez(path, context=context, target=target)


def test_execute_unet_candidate_trains_torch_model_and_writes_report(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    report = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=output_dir,
        start_index=2,
        train_steps=6,
        training_strategy="unet_supervised",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=123,
        force=False,
    )

    assert report["state_hint"] == "UNET_CANDIDATE_EXECUTED"
    assert report["candidate"] == {
        "experiment_id": "unet_unet_supervised_s0002_steps0006",
        "start_index": 2,
        "train_steps": 6,
        "training_strategy": "unet_supervised",
        "strategy_label": None,
    }
    assert report["train"]["mode"] == "torch_unet1d"
    assert report["train"]["loss_terms"] == ["mse"]
    assert len(report["train"]["loss_trace"]) == 6
    assert all(loss >= 0.0 for loss in report["train"]["loss_trace"])
    assert report["train"]["final_loss"] == report["train"]["loss_trace"][-1]
    assert Path(report["artifacts"]["candidate_prediction_path"]).exists()
    assert Path(report["artifacts"]["candidate_checkpoint_path"]).exists()
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()

    import numpy as np

    prediction_npz = np.load(report["artifacts"]["candidate_prediction_path"])
    prediction = prediction_npz["prediction"]
    assert prediction.shape == (3, 5, 12)
    metadata = json.loads(str(prediction_npz["metadata_json"]))
    assert metadata["adapter_type"] == "torch_unet1d_minimal"
    written = json.loads(
        Path(report["artifacts"]["candidate_execution_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_execute_unet_candidate_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    result = CliRunner().invoke(
        app,
        [
            "execute-unet-candidate",
            "--windows-path",
            str(windows_path),
            "--output-dir",
            str(output_dir),
            "--start-index",
            "2",
            "--train-steps",
            "4",
            "--training-strategy",
            "unet_supervised",
            "--base-channels",
            "8",
            "--depth",
            "2",
            "--learning-rate",
            "0.01",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "execute-unet-candidate"
    assert payload["state_hint"] == "UNET_CANDIDATE_EXECUTED"
    assert payload["metrics"]["experiment_id"] == (
        "unet_unet_supervised_s0002_steps0004"
    )
    assert payload["metrics"]["training_strategy"] == "unet_supervised"
    assert payload["metrics"]["final_loss"] >= 0.0
    assert payload["artifacts"]["candidate_execution_report_path"]
    assert payload["next_hint"] == "evaluate-unet-candidate"


def test_execute_unet_candidate_uses_strategy_label_for_unique_ids(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    _write_windows(windows_path)

    first = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=tmp_path / "candidate_a",
        start_index=2,
        train_steps=4,
        training_strategy="unet_supervised",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=123,
        strategy_label="c8_d2_lr1e2_seed123",
        force=False,
    )
    second = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=tmp_path / "candidate_b",
        start_index=2,
        train_steps=4,
        training_strategy="unet_supervised",
        base_channels=12,
        depth=2,
        learning_rate=1e-2,
        seed=123,
        strategy_label="c12_d2_lr1e2_seed123",
        force=False,
    )

    assert first["candidate"]["strategy_label"] == "c8_d2_lr1e2_seed123"
    assert second["candidate"]["strategy_label"] == "c12_d2_lr1e2_seed123"
    assert first["candidate"]["experiment_id"] == (
        "unet_unet_supervised_c8_d2_lr1e2_seed123_s0002_steps0004"
    )
    assert second["candidate"]["experiment_id"] == (
        "unet_unet_supervised_c12_d2_lr1e2_seed123_s0002_steps0004"
    )
    assert first["candidate"]["experiment_id"] != second["candidate"]["experiment_id"]


def test_execute_unet_candidate_supports_pde_aware_training_strategies(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    _write_windows(windows_path)

    spectral = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=tmp_path / "spectral",
        start_index=1,
        train_steps=4,
        training_strategy="unet_spectral",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=7,
        spectral_weight=0.25,
        force=False,
    )
    physics = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=tmp_path / "physics",
        start_index=1,
        train_steps=4,
        training_strategy="unet_physics_residual",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=7,
        nu=0.001,
        dt=1.0,
        dx=0.1,
        physics_weight=0.02,
        force=False,
    )

    assert spectral["candidate"]["training_strategy"] == "unet_spectral"
    assert spectral["train"]["loss_terms"] == ["mse", "spectral"]
    assert spectral["train"]["loss_weights"] == {
        "spectral": 0.25,
        "burgers_residual": 0.01,
    }
    assert len(spectral["train"]["loss_trace"]) == 4
    assert spectral["train"]["final_loss"] >= 0.0
    assert physics["candidate"]["training_strategy"] == "unet_physics_residual"
    assert physics["train"]["loss_terms"] == ["mse", "burgers_residual"]
    assert physics["train"]["loss_weights"] == {
        "spectral": 0.1,
        "burgers_residual": 0.02,
    }
    assert physics["train"]["pde"] == {"nu": 0.001, "dt": 1.0, "dx": 0.1}
    assert len(physics["train"]["loss_trace"]) == 4
    assert physics["train"]["final_loss"] >= 0.0
