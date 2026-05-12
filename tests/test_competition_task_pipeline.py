import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _official_fno_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(37)
    state: dict[str, object] = {
        "fc0.weight": 0.001 * torch.randn(width, history_steps + 1, generator=generator),
        "fc0.bias": torch.zeros(width),
        "fc1.weight": 0.001 * torch.randn(fc_hidden, width, generator=generator),
        "fc1.bias": torch.zeros(fc_hidden),
        "fc2.weight": 0.001 * torch.randn(1, fc_hidden, generator=generator),
        "fc2.bias": torch.zeros(1),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = 0.001 * torch.randn(
            width,
            width,
            modes,
            dtype=torch.complex64,
            generator=generator,
        )
        state[f"w{index}.weight"] = 0.001 * torch.randn(
            width,
            width,
            1,
            generator=generator,
        )
        state[f"w{index}.bias"] = torch.zeros(width)
    return state


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="competition task pipeline requires torch")
def test_run_competition_task_pipeline_builds_audited_submission(tmp_path: Path) -> None:
    import torch

    from pde_agent.core.competition_task_pipeline import run_competition_task_pipeline

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "fno.pt"
    torch.save({"model_state_dict": _official_fno_state_dict(torch=torch)}, checkpoint_path)

    report = run_competition_task_pipeline(
        task_id=1,
        model_family="fno",
        test_path=test_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "pipeline",
        submission_id="team_default",
        dataset_key="tensor",
        force=False,
    )

    assert report["state_hint"] == "COMPETITION_TASK_PIPELINE_COMPLETED"
    assert report["model_family"] == "fno"
    assert report["audit"]["decision"] == "passed"
    assert Path(report["artifacts"]["submission_zip_path"]).exists()
    assert Path(report["artifacts"]["pipeline_report_path"]).exists()
    log_text = (
        Path(report["artifacts"]["submission_dir"]) / "task1_logs.log"
    ).read_text(encoding="utf-8")
    assert "Agent Trace" in log_text
    assert "Experiment Tracking" in log_text
    assert "Inference Log" in log_text
    assert "Validation/Audit" in log_text
    assert "Limitations" in log_text
    assert "model_family: fno" in log_text
    assert str(checkpoint_path) in log_text
    assert "audit_decision: passed" in log_text
    assert report["next_command"] == "submit-if-audit-passed"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="competition task pipeline requires torch")
def test_run_competition_task_pipeline_cli_returns_payload(tmp_path: Path) -> None:
    import torch
    from typer.testing import CliRunner

    from pde_agent.cli import app

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "fno.pt"
    torch.save({"model_state_dict": _official_fno_state_dict(torch=torch)}, checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "run-competition-task-pipeline",
            "--task-id",
            "1",
            "--model-family",
            "fno",
            "--test-path",
            str(test_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(tmp_path / "pipeline"),
            "--submission-id",
            "team_default",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-competition-task-pipeline"
    assert payload["state_hint"] == "COMPETITION_TASK_PIPELINE_COMPLETED"
    assert payload["metrics"]["audit_decision"] == "passed"


def test_run_competition_task_pipeline_command_can_record_local_strategy_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_agent.commands.run_competition_task_pipeline as command_module

    schedule_path = tmp_path / "schedule.json"
    leaderboard_path = tmp_path / "leaderboard.json"
    score_path = tmp_path / "score.json"
    schedule_path.write_text(
        json.dumps(
            {
                "state_hint": "LOCAL_STRATEGY_EVALUATION_SCHEDULED",
                "experiments": [
                    {
                        "experiment_id": "fno_spectral_viscosity_filter_v0_001",
                        "strategy_id": "fno_spectral_viscosity_filter_v0",
                        "title": "FNO spectral viscosity filter",
                        "priority_rank": 1,
                        "selection_score": 12,
                        "acceptance_criteria": [],
                        "evaluation_protocol": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    score_path.write_text(
        json.dumps(
            {
                "state_hint": "COMPETITION_PREDICTION_SCORED",
                "segment_score": 82.5,
                "baseline_score": 80.0,
                "inference_time": 59.7,
            }
        ),
        encoding="utf-8",
    )

    def fake_pipeline(**kwargs: object) -> dict[str, object]:
        return {
            "state_hint": "COMPETITION_TASK_PIPELINE_COMPLETED",
            "task_id": 1,
            "model_family": "fno",
            "audit": {"decision": "passed", "issue_count": 0},
            "materialized": {"score": {"segment_score": 82.5}},
            "artifacts": {
                "pipeline_report_path": str(tmp_path / "pipeline.json"),
                "score_json_path": str(score_path),
            },
            "next_command": "submit-if-audit-passed",
        }

    monkeypatch.setattr(command_module, "run_competition_task_pipeline", fake_pipeline)

    payload = command_module.run_competition_task_pipeline_command(
        task_id=1,
        model_family="fno",
        test_path=str(tmp_path / "test.hdf5"),
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        output_dir=str(tmp_path / "pipeline"),
        submission_id="team_default",
        target_path=None,
        dataset_key="tensor",
        force=False,
        local_strategy_schedule_path=str(schedule_path),
        local_strategy_experiment_id="fno_spectral_viscosity_filter_v0_001",
        local_strategy_leaderboard_path=str(leaderboard_path),
        local_strategy_baseline_score=None,
    )

    assert payload["status"] == "ok"
    assert payload["metrics"]["local_strategy_decision"] == "accept_best_strategy"
    assert payload["next_hint"] == "record-strategy-proposal"
    assert leaderboard_path.exists()
