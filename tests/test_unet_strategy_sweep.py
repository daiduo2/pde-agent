import json
from pathlib import Path

from pde_agent.core.unet_strategy_sweep import run_unet_strategy_sweep


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


def test_run_unet_strategy_sweep_writes_summary_and_evidence(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "strategy_sweep"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    summary = run_unet_strategy_sweep(
        windows_path=windows_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        start_index=2,
        train_steps=[3],
        base_channels=[8, 12],
        depths=[2],
        learning_rates=[1e-2],
        seeds=[5],
        force=False,
    )

    assert summary["state_hint"] == "UNET_STRATEGY_SWEEP_COMPLETED"
    assert summary["record_count"] == 2
    assert summary["evidence_sweep"]["state_hint"] == "UNET_EVIDENCE_SWEEP_COMPLETED"
    assert Path(summary["artifacts"]["summary_json_path"]).exists()
    assert Path(summary["artifacts"]["summary_csv_path"]).exists()
    assert Path(summary["artifacts"]["evidence_summary_json_path"]).exists()
    assert {
        record["strategy_label"] for record in summary["records"]
    } == {
        "c8_d2_lr1e2_seed5",
        "c12_d2_lr1e2_seed5",
    }
    assert summary["winner"]["experiment_id"]
    assert summary["recommendation"]["next_command"] == "run-unet-strategy-sweep"
    written = json.loads(
        (output_dir / "unet_strategy_sweep_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == summary


def test_run_unet_strategy_sweep_accepts_multiple_start_indices(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "strategy_sweep"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    summary = run_unet_strategy_sweep(
        windows_path=windows_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        start_indices=[1, 2],
        train_steps=[3],
        base_channels=[8, 12],
        depths=[2],
        learning_rates=[1e-2],
        seeds=[5],
        force=False,
    )

    assert summary["inputs"]["start_indices"] == [1, 2]
    assert summary["record_count"] == 4
    assert summary["evidence_sweep"]["comparison_count"] == 2
    assert {
        (record["start_index"], record["strategy_label"])
        for record in summary["records"]
    } == {
        (1, "c8_d2_lr1e2_seed5"),
        (1, "c12_d2_lr1e2_seed5"),
        (2, "c8_d2_lr1e2_seed5"),
        (2, "c12_d2_lr1e2_seed5"),
    }
    assert summary["strategy_statistics"]["c8_d2_lr1e2_seed5"]["count"] == 2
    assert summary["strategy_statistics"]["c12_d2_lr1e2_seed5"]["count"] == 2


def test_run_unet_strategy_sweep_accepts_multiple_training_strategies(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "strategy_sweep"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    summary = run_unet_strategy_sweep(
        windows_path=windows_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        start_indices=[1, 2],
        training_strategies=[
            "unet_supervised",
            "unet_spectral",
            "unet_physics_residual",
        ],
        train_steps=[3],
        base_channels=[8],
        depths=[2],
        learning_rates=[1e-2],
        seeds=[5],
        nu=0.001,
        dt=1.0,
        dx=0.1,
        force=False,
    )

    assert summary["inputs"]["training_strategies"] == [
        "unet_supervised",
        "unet_spectral",
        "unet_physics_residual",
    ]
    assert summary["record_count"] == 6
    assert summary["evidence_sweep"]["comparison_count"] == 2
    assert {
        record["training_strategy"] for record in summary["records"]
    } == {
        "unet_supervised",
        "unet_spectral",
        "unet_physics_residual",
    }
    assert {
        record["strategy_label"] for record in summary["records"]
    } == {
        "supervised_c8_d2_lr1e2_seed5",
        "spectral_c8_d2_lr1e2_seed5",
        "physics_residual_c8_d2_lr1e2_seed5",
    }


def test_run_unet_strategy_sweep_expands_loss_weights_by_strategy(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "strategy_sweep"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    summary = run_unet_strategy_sweep(
        windows_path=windows_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        start_indices=[1],
        training_strategies=[
            "unet_spectral",
            "unet_physics_residual",
        ],
        train_steps=[3],
        base_channels=[8],
        depths=[2],
        learning_rates=[1e-2],
        seeds=[5],
        spectral_weights=[0.05, 0.1],
        physics_weights=[0.005, 0.01],
        force=False,
    )

    assert summary["inputs"]["spectral_weights"] == [0.05, 0.1]
    assert summary["inputs"]["physics_weights"] == [0.005, 0.01]
    assert summary["record_count"] == 4
    assert {
        (
            record["training_strategy"],
            record["spectral_weight"],
            record["physics_weight"],
            record["strategy_label"],
        )
        for record in summary["records"]
    } == {
        ("unet_spectral", 0.05, 0.01, "spectral_sw5e2_c8_d2_lr1e2_seed5"),
        ("unet_spectral", 0.1, 0.01, "spectral_sw1e1_c8_d2_lr1e2_seed5"),
        (
            "unet_physics_residual",
            0.1,
            0.005,
            "physics_residual_pw5e3_c8_d2_lr1e2_seed5",
        ),
        (
            "unet_physics_residual",
            0.1,
            0.01,
            "physics_residual_pw1e2_c8_d2_lr1e2_seed5",
        ),
    }


def test_run_unet_strategy_sweep_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "strategy_sweep"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "run-unet-strategy-sweep",
            "--windows-path",
            str(windows_path),
            "--output-dir",
            str(output_dir),
            "--baseline-score-path",
            str(baseline_score_path),
            "--start-index",
            "2",
            "--start-index",
            "1",
            "--train-steps",
            "3",
            "--base-channels",
            "8",
            "--base-channels",
            "12",
            "--depth",
            "2",
            "--learning-rate",
            "0.01",
            "--seed",
            "5",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-unet-strategy-sweep"
    assert payload["state_hint"] == "UNET_STRATEGY_SWEEP_COMPLETED"
    assert payload["metrics"]["record_count"] == 4
    assert payload["metrics"]["evaluation_window_count"] == 2
    assert payload["metrics"]["winner_experiment_id"]
    assert payload["metrics"]["best_score"] > 0
    assert payload["next_hint"] == "run-unet-strategy-sweep"
