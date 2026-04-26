import json
from pathlib import Path

import pytest

from pde_agent.core.fno_finetune_candidate import (
    FNOFinetuneCandidateError,
    execute_fno_finetune_candidate,
)

torch = pytest.importorskip("torch", reason="FNO finetune tests require torch")
np = pytest.importorskip("numpy")


def _official_fno_state_dict(
    *,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(11)
    state: dict[str, object] = {
        "fc0.weight": torch.randn(width, history_steps + 1, generator=generator),
        "fc0.bias": torch.randn(width, generator=generator),
        "fc1.weight": torch.randn(fc_hidden, width, generator=generator),
        "fc1.bias": torch.randn(fc_hidden, generator=generator),
        "fc2.weight": torch.randn(1, fc_hidden, generator=generator),
        "fc2.bias": torch.randn(1, generator=generator),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = torch.randn(
            width,
            width,
            modes,
            dtype=torch.complex64,
            generator=generator,
        )
        state[f"w{index}.weight"] = torch.randn(
            width,
            width,
            1,
            generator=generator,
        )
        state[f"w{index}.bias"] = torch.randn(width, generator=generator)
    return state


def _write_windows(path: Path) -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.03 * np.cos(x) + 0.01 * sample
        for time_index in range(4):
            context[sample, time_index, :] = base + time_index * delta
        for time_index in range(3):
            target[sample, time_index, :] = base + (4 + time_index) * delta
    np.savez(path, context=context, target=target)


def _write_checkpoint(path: Path) -> None:
    torch.save({"model_state_dict": _official_fno_state_dict()}, path)


def _write_candidate_plan(
    path: Path,
    *,
    intervention: str = "use_scheduled_sampling_rollout_finetune",
) -> None:
    workspace = path.parent
    (workspace / "artifacts").mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "CANDIDATE_EXECUTION_PREPARED",
                "candidate_index": 0,
                "candidate": {
                    "experiment_id": f"{intervention}_001",
                    "intervention": intervention,
                    "change_scope": "training_schedule_only",
                    "requires_training": True,
                    "parameters": {
                        "teacher_forcing_decay": 0.95,
                        "rollout_steps": 3,
                    },
                },
                "workspace": {
                    "workspace_dir": str(workspace),
                    "artifacts_dir": str(workspace / "artifacts"),
                    "logs_dir": str(workspace / "logs"),
                    "candidate_execution_plan_path": str(path),
                },
                "execution_status": "not_started",
                "next_command": "execute-candidate-experiment",
            }
        ),
        encoding="utf-8",
    )


def test_execute_fno_finetune_candidate_writes_outputs(tmp_path: Path) -> None:
    candidate_plan_path = tmp_path / "candidate" / "candidate_execution_plan.json"
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    _write_candidate_plan(candidate_plan_path)
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    report = execute_fno_finetune_candidate(
        candidate_plan_path=candidate_plan_path,
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        max_train_steps=1,
        learning_rate=1e-4,
        force=False,
    )

    artifacts_dir = candidate_plan_path.parent / "artifacts"
    assert report["state_hint"] == "FNO_FINETUNE_CANDIDATE_EXECUTED"
    assert report["candidate"]["experiment_id"] == (
        "use_scheduled_sampling_rollout_finetune_001"
    )
    assert report["train"]["train_steps"] == 1
    assert len(report["train"]["loss_trace"]) == 1
    assert Path(report["artifacts"]["finetuned_checkpoint_path"]).exists()
    assert Path(report["artifacts"]["candidate_prediction_path"]).exists()
    assert Path(report["artifacts"]["candidate_execution_report_path"]).exists()
    with np.load(artifacts_dir / "candidate_prediction.npz", allow_pickle=False) as data:
        prediction = data["prediction"]
    assert prediction.shape == (2, 3, 8)
    assert np.all(np.isfinite(prediction))


def test_execute_fno_finetune_candidate_rejects_unsupported_intervention(
    tmp_path: Path,
) -> None:
    candidate_plan_path = tmp_path / "candidate" / "candidate_execution_plan.json"
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    _write_candidate_plan(candidate_plan_path, intervention="apply_low_pass_stabilization")
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    with pytest.raises(FNOFinetuneCandidateError, match="unsupported intervention"):
        execute_fno_finetune_candidate(
            candidate_plan_path=candidate_plan_path,
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            max_train_steps=1,
            learning_rate=1e-4,
            force=False,
        )


def test_execute_fno_finetune_candidate_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    candidate_plan_path = tmp_path / "candidate" / "candidate_execution_plan.json"
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    _write_candidate_plan(candidate_plan_path)
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "execute-fno-finetune-candidate",
            "--candidate-plan-path",
            str(candidate_plan_path),
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--max-train-steps",
            "1",
            "--learning-rate",
            "0.0001",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "execute-fno-finetune-candidate"
    assert payload["state_hint"] == "FNO_FINETUNE_CANDIDATE_EXECUTED"
    assert payload["metrics"]["train_steps"] == 1
    assert payload["metrics"]["experiment_id"] == (
        "use_scheduled_sampling_rollout_finetune_001"
    )
    assert payload["artifacts"]["candidate_prediction_path"]
    assert payload["next_hint"] == "score-local-rollout"
