import json
from pathlib import Path

from pde_agent.core.deeponet_candidate_evaluation import evaluate_deeponet_candidate


def _write_prediction_and_target(root: Path) -> tuple[Path, Path]:
    import numpy as np

    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    target = np.zeros((2, 4, 8), dtype=np.float32)
    prediction = np.zeros((2, 4, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.2)
        drift = 0.03 * np.cos(x) + 0.005 * sample
        for time_index in range(4):
            target[sample, time_index, :] = base + time_index * drift
            prediction[sample, time_index, :] = target[sample, time_index, :] + 0.01

    prediction_path = root / "deeponet_candidate_prediction.npz"
    target_path = root / "windows.npz"
    np.savez(prediction_path, prediction=prediction)
    np.savez(target_path, context=target[:, :2, :], target=target)
    return prediction_path, target_path


def _write_candidate_report(root: Path, prediction_path: Path) -> Path:
    report_path = root / "deeponet_candidate_execution_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "DEEPONET_CANDIDATE_EXECUTED",
                "candidate": {
                    "experiment_id": "deeponet_branch_trunk_rollout_001",
                    "start_index": 0,
                    "train_steps": 3,
                    "training_strategy": "branch_trunk_rollout",
                },
                "inputs": {"forecast_steps": 4},
                "artifacts": {
                    "candidate_prediction_path": str(prediction_path),
                },
                "next_command": "evaluate-deeponet-candidate",
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _write_baseline_score(path: Path, score: float) -> None:
    path.write_text(
        json.dumps(
            {
                "competition_like_score_short": score,
                "rel_mse": 0.4,
                "rmse": 0.4,
            }
        ),
        encoding="utf-8",
    )


def test_evaluate_deeponet_candidate_writes_evidence_ready_report(
    tmp_path: Path,
) -> None:
    prediction_path, target_path = _write_prediction_and_target(tmp_path)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "evaluation"
    _write_baseline_score(baseline_score_path, score=40.0)

    report = evaluate_deeponet_candidate(
        candidate_report_path=candidate_report_path,
        target_path=target_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    expected_files = {
        "candidate_score.json",
        "candidate_rollout_error.json",
        "candidate_burgers_physics.json",
        "candidate_burgers_residual.json",
        "candidate_selected_pde_hypothesis.json",
        "candidate_evaluation_report.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert report["state_hint"] == "DEEPONET_CANDIDATE_EVALUATED"
    assert report["candidate"] == {
        "experiment_id": "deeponet_branch_trunk_rollout_001",
        "start_index": 0,
        "train_steps": 3,
        "training_strategy": "branch_trunk_rollout",
    }
    assert report["inputs"]["forecast_steps"] == 4
    assert report["score"]["competition_like_score_short"] > 0
    assert report["artifacts"]["burgers_residual_path"] == str(
        output_dir / "candidate_burgers_residual.json"
    )
    assert report["physics_residual"]["residual_rmse"] >= 0.0
    assert report["physics_residual"]["physics_issue"]
    assert report["baseline_comparison"]["baseline_score"] == 40.0
    assert report["baseline_comparison"]["score_delta"] == (
        report["score"]["competition_like_score_short"] - 40.0
    )
    assert report["selected_hypothesis"]["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert report["next_command"] == "build-deeponet-evidence-sweep"
    written = json.loads(
        (output_dir / "candidate_evaluation_report.json").read_text(encoding="utf-8")
    )
    assert written == report


def test_evaluate_deeponet_candidate_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    prediction_path, target_path = _write_prediction_and_target(tmp_path)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "evaluation"
    _write_baseline_score(baseline_score_path, score=40.0)

    result = CliRunner().invoke(
        app,
        [
            "evaluate-deeponet-candidate",
            "--candidate-report-path",
            str(candidate_report_path),
            "--target-path",
            str(target_path),
            "--output-dir",
            str(output_dir),
            "--baseline-score-path",
            str(baseline_score_path),
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
    assert payload["command"] == "evaluate-deeponet-candidate"
    assert payload["state_hint"] == "DEEPONET_CANDIDATE_EVALUATED"
    assert payload["metrics"]["experiment_id"] == "deeponet_branch_trunk_rollout_001"
    assert payload["metrics"]["training_strategy"] == "branch_trunk_rollout"
    assert payload["metrics"]["residual_rmse"] >= 0.0
    assert payload["artifacts"]["candidate_burgers_residual_path"]
    assert payload["metrics"]["score_delta"] == (
        payload["metrics"]["competition_like_score_short"] - 40.0
    )
    assert payload["artifacts"]["candidate_evaluation_report_path"]
    assert payload["next_hint"] == "build-deeponet-evidence-sweep"
