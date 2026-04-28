import json
from pathlib import Path

from pde_agent.core.unet_candidate_evaluation import evaluate_unet_candidate
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


def test_evaluate_unet_candidate_writes_diagnostic_bundle(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    candidate_dir = tmp_path / "candidate"
    evaluation_dir = tmp_path / "evaluation"
    baseline_score_path = tmp_path / "baseline_score.json"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )
    candidate = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=candidate_dir,
        start_index=1,
        train_steps=4,
        training_strategy="unet_supervised",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=7,
        force=False,
    )

    report = evaluate_unet_candidate(
        candidate_report_path=candidate["artifacts"]["candidate_execution_report_path"],
        target_path=windows_path,
        output_dir=evaluation_dir,
        baseline_score_path=baseline_score_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "UNET_CANDIDATE_EVALUATED"
    assert report["candidate"]["training_strategy"] == "unet_supervised"
    assert report["score"]["competition_like_score_short"] > 0
    assert report["baseline_comparison"]["score_delta"] == (
        report["score"]["competition_like_score_short"] - 1.0
    )
    assert report["physics_residual"]["high_frequency_energy_ratio"] >= 0.0
    assert report["next_command"] == "build-unet-evidence-sweep"
    assert Path(report["artifacts"]["score_path"]).exists()
    assert Path(report["artifacts"]["rollout_error_path"]).exists()
    assert Path(report["artifacts"]["burgers_physics_path"]).exists()
    assert Path(report["artifacts"]["burgers_residual_path"]).exists()
    assert Path(report["artifacts"]["selected_hypothesis_path"]).exists()
    assert Path(report["artifacts"]["evaluation_report_path"]).exists()
    written = json.loads(
        Path(report["artifacts"]["evaluation_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_evaluate_unet_candidate_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    candidate_dir = tmp_path / "candidate"
    evaluation_dir = tmp_path / "evaluation"
    baseline_score_path = tmp_path / "baseline_score.json"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )
    candidate = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=candidate_dir,
        start_index=1,
        train_steps=4,
        training_strategy="unet_supervised",
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=7,
        force=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-unet-candidate",
            "--candidate-report-path",
            candidate["artifacts"]["candidate_execution_report_path"],
            "--target-path",
            str(windows_path),
            "--output-dir",
            str(evaluation_dir),
            "--baseline-score-path",
            str(baseline_score_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "evaluate-unet-candidate"
    assert payload["state_hint"] == "UNET_CANDIDATE_EVALUATED"
    assert payload["metrics"]["experiment_id"] == (
        "unet_unet_supervised_s0001_steps0004"
    )
    assert payload["metrics"]["training_strategy"] == "unet_supervised"
    assert payload["metrics"]["score"] > 0
    assert payload["artifacts"]["evaluation_report_path"]
    assert payload["next_hint"] == "build-unet-evidence-sweep"
