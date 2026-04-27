import json
from pathlib import Path

from pde_agent.core.deeponet_candidate_dryrun import execute_deeponet_candidate_dryrun


def _write_windows(path: Path) -> None:
    import numpy as np

    x = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    context = np.stack(
        [
            np.stack([np.sin(x), np.sin(x) + 0.1], axis=0),
            np.stack([np.cos(x), np.cos(x) + 0.2], axis=0),
        ],
        axis=0,
    ).astype(np.float32)
    target = np.stack(
        [
            np.stack([np.sin(x) + 0.2, np.sin(x) + 0.3, np.sin(x) + 0.4], axis=0),
            np.stack([np.cos(x) + 0.4, np.cos(x) + 0.6, np.cos(x) + 0.8], axis=0),
        ],
        axis=0,
    ).astype(np.float32)
    np.savez(path, context=context, target=target)


def test_execute_deeponet_candidate_dryrun_writes_prediction_and_report(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    report = execute_deeponet_candidate_dryrun(
        windows_path=windows_path,
        output_dir=output_dir,
        start_index=7,
        train_steps=5,
        training_strategy="branch_trunk_rollout",
        force=False,
    )

    assert report["state_hint"] == "DEEPONET_CANDIDATE_EXECUTED"
    assert report["candidate"] == {
        "experiment_id": "deeponet_branch_trunk_rollout_s0007_steps0005",
        "start_index": 7,
        "train_steps": 5,
        "training_strategy": "branch_trunk_rollout",
    }
    assert report["inputs"]["forecast_steps"] == 3
    assert report["inputs"]["context_steps"] == 2
    assert Path(report["artifacts"]["candidate_prediction_path"]).exists()
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()
    assert report["next_command"] == "evaluate-deeponet-candidate"

    import numpy as np

    prediction = np.load(report["artifacts"]["candidate_prediction_path"])["prediction"]
    assert prediction.shape == (2, 3, 6)
    written = json.loads(
        Path(report["artifacts"]["candidate_execution_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_execute_deeponet_candidate_dryrun_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    result = CliRunner().invoke(
        app,
        [
            "execute-deeponet-candidate-dryrun",
            "--windows-path",
            str(windows_path),
            "--output-dir",
            str(output_dir),
            "--start-index",
            "7",
            "--train-steps",
            "5",
            "--training-strategy",
            "branch_trunk_rollout",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "execute-deeponet-candidate-dryrun"
    assert payload["state_hint"] == "DEEPONET_CANDIDATE_EXECUTED"
    assert payload["metrics"]["experiment_id"] == (
        "deeponet_branch_trunk_rollout_s0007_steps0005"
    )
    assert payload["metrics"]["training_strategy"] == "branch_trunk_rollout"
    assert payload["artifacts"]["candidate_execution_report_path"]
    assert payload["next_hint"] == "evaluate-deeponet-candidate"


def test_deeponet_dryrun_evaluation_report_enters_evidence_sweep(
    tmp_path: Path,
) -> None:
    from pde_agent.core.deeponet_candidate_evaluation import (
        evaluate_deeponet_candidate,
    )
    from pde_agent.core.deeponet_evidence_sweep import build_deeponet_evidence_sweep

    windows_path = tmp_path / "windows.npz"
    _write_windows(windows_path)
    baseline_score_path = tmp_path / "baseline_score.json"
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )
    execution = execute_deeponet_candidate_dryrun(
        windows_path=windows_path,
        output_dir=tmp_path / "candidate",
        start_index=7,
        train_steps=5,
        training_strategy="branch_trunk_rollout",
        force=False,
    )

    evaluation = evaluate_deeponet_candidate(
        candidate_report_path=execution["artifacts"][
            "candidate_execution_report_path"
        ],
        target_path=windows_path,
        output_dir=tmp_path / "evaluation",
        baseline_score_path=baseline_score_path,
        force=False,
    )
    summary = build_deeponet_evidence_sweep(
        evaluation_report_paths=[
            evaluation["artifacts"]["evaluation_report_path"],
        ],
        output_dir=tmp_path / "evidence",
    )

    assert evaluation["state_hint"] == "DEEPONET_CANDIDATE_EVALUATED"
    assert summary["state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert summary["records"][0]["experiment_id"] == (
        "deeponet_branch_trunk_rollout_s0007_steps0005"
    )
