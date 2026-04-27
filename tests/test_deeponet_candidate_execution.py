import json
from pathlib import Path

from pde_agent.core.deeponet_candidate_execution import execute_deeponet_candidate
from pde_agent.core.deeponet_candidate_evaluation import evaluate_deeponet_candidate
from pde_agent.core.deeponet_evidence_sweep import build_deeponet_evidence_sweep


def _write_windows(path: Path) -> None:
    import numpy as np

    x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    context = np.zeros((3, 3, 10), dtype=np.float32)
    target = np.zeros((3, 4, 10), dtype=np.float32)
    for sample in range(3):
        base = np.sin(2.0 * np.pi * (x + 0.03 * sample))
        trend = 0.04 * np.cos(2.0 * np.pi * x) + 0.01 * sample
        for time_index in range(context.shape[1]):
            context[sample, time_index, :] = base + time_index * trend
        for time_index in range(target.shape[1]):
            target[sample, time_index, :] = base + (time_index + 3) * trend
    np.savez(path, context=context, target=target)


def test_execute_deeponet_candidate_trains_torch_model_and_writes_report(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    report = execute_deeponet_candidate(
        windows_path=windows_path,
        output_dir=output_dir,
        start_index=2,
        train_steps=8,
        training_strategy="branch_trunk_supervised",
        hidden_width=16,
        latent_dim=8,
        learning_rate=1e-2,
        seed=123,
        force=False,
    )

    assert report["state_hint"] == "DEEPONET_CANDIDATE_EXECUTED"
    assert report["candidate"] == {
        "experiment_id": "deeponet_branch_trunk_supervised_s0002_steps0008",
        "start_index": 2,
        "train_steps": 8,
        "training_strategy": "branch_trunk_supervised",
    }
    assert report["train"]["mode"] == "torch_deeponet"
    assert len(report["train"]["loss_trace"]) == 8
    assert all(loss >= 0.0 for loss in report["train"]["loss_trace"])
    assert report["train"]["final_loss"] == report["train"]["loss_trace"][-1]
    assert Path(report["artifacts"]["candidate_prediction_path"]).exists()
    assert Path(report["artifacts"]["candidate_checkpoint_path"]).exists()
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()

    import numpy as np

    prediction_npz = np.load(report["artifacts"]["candidate_prediction_path"])
    prediction = prediction_npz["prediction"]
    assert prediction.shape == (3, 4, 10)
    metadata = json.loads(str(prediction_npz["metadata_json"]))
    assert metadata["adapter_type"] == "torch_deeponet_minimal"
    written = json.loads(
        Path(report["artifacts"]["candidate_execution_report_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_execute_deeponet_candidate_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    output_dir = tmp_path / "candidate"
    _write_windows(windows_path)

    result = CliRunner().invoke(
        app,
        [
            "execute-deeponet-candidate",
            "--windows-path",
            str(windows_path),
            "--output-dir",
            str(output_dir),
            "--start-index",
            "2",
            "--train-steps",
            "4",
            "--training-strategy",
            "branch_trunk_supervised",
            "--hidden-width",
            "16",
            "--latent-dim",
            "8",
            "--learning-rate",
            "0.01",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "execute-deeponet-candidate"
    assert payload["state_hint"] == "DEEPONET_CANDIDATE_EXECUTED"
    assert payload["metrics"]["experiment_id"] == (
        "deeponet_branch_trunk_supervised_s0002_steps0004"
    )
    assert payload["metrics"]["training_strategy"] == "branch_trunk_supervised"
    assert payload["metrics"]["final_loss"] >= 0.0
    assert payload["artifacts"]["candidate_execution_report_path"]
    assert payload["next_hint"] == "evaluate-deeponet-candidate"


def test_trainable_deeponet_candidate_enters_evidence_sweep(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    candidate_dir = tmp_path / "candidate"
    evaluation_dir = tmp_path / "evaluation"
    sweep_dir = tmp_path / "sweep"
    baseline_score_path = tmp_path / "baseline_score.json"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )

    candidate_report = execute_deeponet_candidate(
        windows_path=windows_path,
        output_dir=candidate_dir,
        start_index=1,
        train_steps=4,
        training_strategy="branch_trunk_supervised",
        hidden_width=16,
        latent_dim=8,
        learning_rate=1e-2,
        seed=7,
        force=False,
    )
    evaluation_report = evaluate_deeponet_candidate(
        candidate_report_path=candidate_report["artifacts"][
            "candidate_execution_report_path"
        ],
        target_path=windows_path,
        output_dir=evaluation_dir,
        baseline_score_path=baseline_score_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )
    sweep = build_deeponet_evidence_sweep(
        evaluation_report_paths=[
            evaluation_report["artifacts"]["evaluation_report_path"]
        ],
        output_dir=sweep_dir,
        force=False,
    )

    assert evaluation_report["state_hint"] == "DEEPONET_CANDIDATE_EVALUATED"
    assert sweep["state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert sweep["record_count"] == 1
    assert sweep["records"][0]["experiment_id"] == (
        candidate_report["candidate"]["experiment_id"]
    )
    assert sweep["comparison_winners"][0]["winner_training_strategy"] == (
        "branch_trunk_supervised"
    )
