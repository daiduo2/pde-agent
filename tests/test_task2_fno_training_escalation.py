"""Tests for Task 2 FNO training escalation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent.core.task2_fno_training_escalation import (
    Task2FnoTrainingEscalationError,
    run_task2_fno_training_escalation_pipeline,
)


def _make_memory(**overrides: object) -> dict:
    base = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "iteration_index": 3,
        "current_best": {
            "candidate_id": "c3",
            "score": 72.5,
            "config": {
                "hidden_channels": 96,
                "modes": 64,
                "depth": 5,
                "physics_weight": 0.01,
                "learning_rate": 0.001,
                "train_steps": 10000,
                "batch_size": 32,
            },
        },
        "best_so_far": {
            "candidate_id": "c1",
            "score": 74.0,
            "config": {
                "hidden_channels": 96,
                "modes": 64,
                "depth": 5,
                "physics_weight": 0.01,
                "learning_rate": 0.001,
                "train_steps": 10000,
                "batch_size": 32,
            },
        },
        "trajectory_summary": [
            {
                "candidate_id": "c1",
                "score": 74.0,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c2",
                "score": 73.0,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c3",
                "score": 72.5,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
        ],
        "blocked_configs": [
            {
                "candidate_id": "c0",
                "config": {
                    "hidden_channels": 64,
                    "modes": 32,
                    "depth": 3,
                    "physics_weight": 0.0,
                    "learning_rate": 0.001,
                    "train_steps": 5000,
                    "batch_size": 16,
                },
                "reason": "zero_or_negative_score",
            }
        ],
        "search_guidance": [
            {
                "guidance_type": "avoid_config",
                "reason": "poor_performance",
                "config_signature": "h64_m32_d3_pw0.0_lr0.001_s5000_b16",
            }
        ],
        "convergence_history": {
            "no_improvement_count": 2,
            "best_score_history": [74.0, 73.0, 72.5],
        },
    }
    base.update(overrides)
    return base


def test_escalation_doom_loop(tmp_path: Path) -> None:
    # Scores must be within min_score_delta (0.5) for doom_loop
    memory = _make_memory(
        trajectory_summary=[
            {
                "candidate_id": "c1",
                "score": 74.0,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c2",
                "score": 74.1,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c3",
                "score": 73.9,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
        ],
        convergence_history={
            "no_improvement_count": 2,
            "best_score_history": [74.0, 74.1, 73.9],
        },
        current_best={"candidate_id": "c3", "score": 73.9, "config": {}},
        best_so_far={"candidate_id": "c1", "score": 74.0, "config": {}},
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    assert report["state_hint"] == "TASK2_FNO_TRAINING_ESCALATION_RECOMMENDATIONS_GENERATED"
    assert report["next_command"] == "run-task2-fno-training-iteration"

    failure = report["failure_analysis"]
    assert failure["primary_failure_mode"]["mode"] == "doom_loop"

    recs = report["strategy_recommendations"]
    assert any(r["strategy"] == "escape_doom_loop" for r in recs)
    assert any(r["strategy"] == "fno_lite_task2_specific" for r in recs)


def test_escalation_stagnation(tmp_path: Path) -> None:
    # Different configs but no improvement
    memory = _make_memory(
        trajectory_summary=[
            {
                "candidate_id": "c1",
                "score": 74.0,
                "config": {
                    "hidden_channels": 96,
                    "modes": 64,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c2",
                "score": 73.5,
                "config": {
                    "hidden_channels": 128,
                    "modes": 48,
                    "depth": 4,
                    "physics_weight": 0.005,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
            {
                "candidate_id": "c3",
                "score": 73.0,
                "config": {
                    "hidden_channels": 64,
                    "modes": 80,
                    "depth": 6,
                    "physics_weight": 0.02,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
            },
        ],
        convergence_history={
            "no_improvement_count": 3,
            "best_score_history": [74.0, 73.5, 73.0],
        },
        current_best={"candidate_id": "c3", "score": 73.0, "config": {}},
        best_so_far={"candidate_id": "c1", "score": 74.0, "config": {}},
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    failure = report["failure_analysis"]
    assert failure["primary_failure_mode"]["mode"] == "stagnation"
    recs = report["strategy_recommendations"]
    assert any(r["strategy"] == "expand_search_radius" for r in recs)


def test_escalation_poor_performance(tmp_path: Path) -> None:
    memory = _make_memory(
        best_so_far={"candidate_id": "c1", "score": -5.0, "config": {}},
        current_best={"candidate_id": "c1", "score": -5.0, "config": {}},
        trajectory_summary=[
            {"candidate_id": "c1", "score": -5.0, "config": {}}
        ],
        convergence_history={"no_improvement_count": 0, "best_score_history": [-5.0]},
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    failure = report["failure_analysis"]
    assert failure["primary_failure_mode"]["mode"] == "poor_absolute_performance"
    recs = report["strategy_recommendations"]
    assert any(r["strategy"] == "increase_model_capacity" for r in recs)


def test_escalation_regression(tmp_path: Path) -> None:
    memory = _make_memory(
        current_best={"candidate_id": "c3", "score": 70.0, "config": {}},
        best_so_far={"candidate_id": "c1", "score": 74.0, "config": {}},
        convergence_history={
            "no_improvement_count": 0,
            "best_score_history": [74.0, 70.0],
        },
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    failure = report["failure_analysis"]
    assert failure["primary_failure_mode"]["mode"] == "regression"
    recs = report["strategy_recommendations"]
    assert any(r["strategy"] == "return_to_best" for r in recs)


def test_escalation_search_space_exhaustion(tmp_path: Path) -> None:
    memory = _make_memory(
        blocked_configs=[{"candidate_id": f"c{i}", "config": {}, "reason": "bad"} for i in range(10)],
        trajectory_summary=[{"candidate_id": f"c{i}", "score": 50.0, "config": {}} for i in range(12)],
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    failure = report["failure_analysis"]
    assert any(fm["mode"] == "search_space_exhaustion" for fm in failure["failure_modes"])


def test_escalation_missing_memory(tmp_path: Path) -> None:
    with pytest.raises(Task2FnoTrainingEscalationError):
        run_task2_fno_training_escalation_pipeline(
            workspace_dir=tmp_path,
            memory_path="/nonexistent/memory.json",
        )


def test_escalation_refuses_overwrite(tmp_path: Path) -> None:
    memory = _make_memory()
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")
    out_path = tmp_path / "report.json"
    out_path.write_text("{}", encoding="utf-8")

    with pytest.raises(Task2FnoTrainingEscalationError):
        run_task2_fno_training_escalation_pipeline(
            workspace_dir=tmp_path,
            memory_path=memory_path,
            output_path=out_path,
            force=False,
        )


def test_escalation_output_written(tmp_path: Path) -> None:
    memory = _make_memory()
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory), encoding="utf-8")

    report = run_task2_fno_training_escalation_pipeline(
        workspace_dir=tmp_path,
        memory_path=memory_path,
        force=True,
    )

    out_file = tmp_path / "task2_fno_training_escalation_report.json"
    assert out_file.is_file()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["state_hint"] == report["state_hint"]
