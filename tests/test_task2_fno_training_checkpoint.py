"""Tests for Task 2 FNO training checkpoint module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent.core.task2_fno_training_checkpoint import (
    Task2FnoTrainingCheckpointError,
    checkpoint_task2_training_loop,
    list_task2_training_checkpoints,
    resume_task2_training_loop,
)


def test_checkpoint_and_resume(tmp_path: Path) -> None:
    memory = {"state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT", "iteration_index": 2}
    convergence = {"decision": "continue"}
    checkpoint_path = checkpoint_task2_training_loop(
        workspace_dir=tmp_path,
        iteration_index=2,
        memory=memory,
        convergence=convergence,
        best_config={"modes": 64},
        trajectory=[{"score": 76.0}],
        force=True,
    )
    assert checkpoint_path.is_file()
    assert checkpoint_path.name == "task2_fno_checkpoint_0002.json"

    state = resume_task2_training_loop(checkpoint_path=checkpoint_path)
    assert state["iteration_index"] == 2
    assert state["memory"]["iteration_index"] == 2
    assert state["convergence"]["decision"] == "continue"
    assert state["best_config"] == {"modes": 64}
    assert state["trajectory"] == [{"score": 76.0}]
    assert "timestamp" in state


def test_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    memory = {"iteration_index": 0}
    convergence = {}
    checkpoint_task2_training_loop(
        workspace_dir=tmp_path,
        iteration_index=0,
        memory=memory,
        convergence=convergence,
        force=True,
    )
    with pytest.raises(Task2FnoTrainingCheckpointError):
        checkpoint_task2_training_loop(
            workspace_dir=tmp_path,
            iteration_index=0,
            memory=memory,
            convergence=convergence,
            force=False,
        )


def test_resume_missing_checkpoint() -> None:
    with pytest.raises(Task2FnoTrainingCheckpointError):
        resume_task2_training_loop(checkpoint_path="/nonexistent/path.json")


def test_resume_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json")
    with pytest.raises(Task2FnoTrainingCheckpointError):
        resume_task2_training_loop(checkpoint_path=bad_path)


def test_resume_wrong_version(tmp_path: Path) -> None:
    bad_path = tmp_path / "wrong_version.json"
    bad_path.write_text(json.dumps({"checkpoint_version": 99}))
    with pytest.raises(Task2FnoTrainingCheckpointError):
        resume_task2_training_loop(checkpoint_path=bad_path)


def test_list_checkpoints(tmp_path: Path) -> None:
    for i in range(3):
        checkpoint_task2_training_loop(
            workspace_dir=tmp_path,
            iteration_index=i,
            memory={"iteration_index": i},
            convergence={},
            force=True,
        )
    checkpoints = list_task2_training_checkpoints(workspace_dir=tmp_path)
    assert len(checkpoints) == 3
    assert checkpoints[0].name == "task2_fno_checkpoint_0000.json"
    assert checkpoints[2].name == "task2_fno_checkpoint_0002.json"


def test_list_checkpoints_empty(tmp_path: Path) -> None:
    checkpoints = list_task2_training_checkpoints(workspace_dir=tmp_path)
    assert checkpoints == []
