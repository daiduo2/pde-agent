"""Tests for Task 2 FNO training closed loop."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent.core.task2_fno_training_closed_loop import (
    Task2FnoTrainingClosedLoopError,
    build_task2_fno_training_closed_loop_contract,
    build_task2_fno_training_final_report,
    build_task2_fno_training_health_report,
    build_task2_fno_training_iteration_report,
    build_task2_fno_training_memory,
    check_task2_fno_training_convergence,
    evaluate_task2_fno_training_iteration,
)


@pytest.fixture
def sample_opt_report() -> dict:
    return {
        "schema_version": 1,
        "state_hint": "TASK2_FNO_OPTIMIZATION_LOOP_RAN",
        "candidate_count": 3,
        "completed_count": 3,
        "failed_count": 0,
        "candidate_records": [
            {
                "candidate_id": "task2_h64_m32_d3_pw0.0_lr0.0005_s5000_b16",
                "candidate_family": "fno_lite_task2",
                "config": {
                    "hidden_channels": 64,
                    "modes": 32,
                    "depth": 3,
                    "physics_weight": 0.0,
                    "learning_rate": 0.0005,
                    "train_steps": 5000,
                    "batch_size": 16,
                },
                "score": 70.0,
                "segment_score": 46.67,
                "status": "completed",
            },
            {
                "candidate_id": "task2_h96_m48_d5_pw0.01_lr0.001_s10000_b32",
                "candidate_family": "fno_lite_task2",
                "config": {
                    "hidden_channels": 96,
                    "modes": 48,
                    "depth": 5,
                    "physics_weight": 0.01,
                    "learning_rate": 0.001,
                    "train_steps": 10000,
                    "batch_size": 32,
                },
                "score": 76.0,
                "segment_score": 50.67,
                "status": "completed",
            },
            {
                "candidate_id": "task2_h128_m64_d6_pw0.02_lr0.002_s20000_b32",
                "candidate_family": "fno_lite_task2",
                "config": {
                    "hidden_channels": 128,
                    "modes": 64,
                    "depth": 6,
                    "physics_weight": 0.02,
                    "learning_rate": 0.002,
                    "train_steps": 20000,
                    "batch_size": 32,
                },
                "score": 72.0,
                "segment_score": 48.0,
                "status": "completed",
            },
        ],
        "selected_candidate": {
            "candidate_id": "task2_h96_m48_d5_pw0.01_lr0.001_s10000_b32",
            "candidate_family": "fno_lite_task2",
            "score": 76.0,
            "config": {
                "hidden_channels": 96,
                "modes": 48,
                "depth": 5,
            },
        },
        "best_score": 76.0,
    }


def test_evaluate_task2_fno_training_iteration(sample_opt_report: dict) -> None:
    eval_report = evaluate_task2_fno_training_iteration(opt_report=sample_opt_report)
    assert eval_report["state_hint"] == "TASK2_FNO_TRAINING_ITERATION_EVALUATED"
    assert eval_report["best_candidate_id"] == "task2_h96_m48_d5_pw0.01_lr0.001_s10000_b32"
    assert eval_report["best_score"] == 76.0
    assert eval_report["candidate_count"] == 3
    assert eval_report["score_mean"] == pytest.approx(72.67, rel=0.01)
    assert eval_report["score_max"] == 76.0
    assert eval_report["score_min"] == 70.0


def test_evaluate_invalid_state_hint(sample_opt_report: dict) -> None:
    bad_report = dict(sample_opt_report)
    bad_report["state_hint"] = "INVALID"
    with pytest.raises(Task2FnoTrainingClosedLoopError):
        evaluate_task2_fno_training_iteration(opt_report=bad_report)


def test_build_task2_fno_training_memory(sample_opt_report: dict) -> None:
    memory = build_task2_fno_training_memory(
        optimization_report=sample_opt_report,
        iteration_index=0,
        previous_memory=None,
    )
    assert memory["state_hint"] == "TASK2_FNO_TRAINING_MEMORY_BUILT"
    assert memory["iteration_index"] == 0
    assert memory["current_best"]["score"] == 76.0
    assert memory["best_so_far"]["score"] == 76.0
    assert len(memory["trajectory_summary"]) == 3
    assert len(memory["blocked_configs"]) == 0
    assert memory["convergence_history"]["no_improvement_count"] == 0


def test_build_memory_with_previous(sample_opt_report: dict) -> None:
    first = build_task2_fno_training_memory(
        optimization_report=sample_opt_report,
        iteration_index=0,
        previous_memory=None,
    )
    # Second iteration with worse score
    worse_report = dict(sample_opt_report)
    worse_report["selected_candidate"] = {
        "candidate_id": "task2_worse",
        "candidate_family": "fno_lite_task2",
        "score": 70.0,
        "config": {"hidden_channels": 64},
    }
    worse_report["candidate_records"] = [
        {
            "candidate_id": "task2_worse",
            "candidate_family": "fno_lite_task2",
            "config": {"hidden_channels": 64},
            "score": 70.0,
            "status": "completed",
        }
    ]
    second = build_task2_fno_training_memory(
        optimization_report=worse_report,
        iteration_index=1,
        previous_memory=first,
    )
    assert second["best_so_far"]["score"] == 76.0  # preserved from first
    assert second["current_best"]["score"] == 70.0
    assert second["convergence_history"]["no_improvement_count"] == 1


def test_check_convergence_continue() -> None:
    memory = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "best_so_far": {"score": 70.0},
        "trajectory_summary": [],
        "convergence_history": {
            "no_improvement_count": 0,
            "best_score_history": [70.0],
        },
    }
    result = check_task2_fno_training_convergence(memory=memory, patience=3)
    assert result["decision"] == "continue"
    assert result["confidence"] == 0.5


def test_check_convergence_target_met() -> None:
    memory = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "best_so_far": {"score": 120.0},
        "trajectory_summary": [],
        "convergence_history": {
            "no_improvement_count": 0,
            "best_score_history": [120.0],
        },
    }
    result = check_task2_fno_training_convergence(
        memory=memory, target_score=100.0, patience=3
    )
    assert result["decision"] == "converged"
    assert result["confidence"] == 0.95


def test_check_convergence_stagnant() -> None:
    memory = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "best_so_far": {"score": 70.0},
        "trajectory_summary": [
            {"candidate_id": "a", "score": 70.0, "config": {"modes": 32}},
        ],
        "convergence_history": {
            "no_improvement_count": 3,
            "best_score_history": [70.0, 70.0, 70.0, 70.0],
        },
    }
    result = check_task2_fno_training_convergence(memory=memory, patience=3)
    assert result["decision"] == "stagnant"
    assert any(a["action"] == "expand_search_radius" for a in result["recommended_actions"])


def test_check_convergence_doom_loop() -> None:
    memory = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "best_so_far": {"score": 70.0},
        "trajectory_summary": [
            {"candidate_id": "a", "score": 70.0, "config": {"modes": 32}},
            {"candidate_id": "b", "score": 70.1, "config": {"modes": 32}},
            {"candidate_id": "c", "score": 70.0, "config": {"modes": 32}},
        ],
        "convergence_history": {
            "no_improvement_count": 2,
            "best_score_history": [70.0, 70.1, 70.0],
        },
    }
    result = check_task2_fno_training_convergence(memory=memory, patience=3)
    assert result["decision"] == "doom_loop"
    assert result["confidence"] == 0.85


def test_build_contract(tmp_path: Path) -> None:
    contract = build_task2_fno_training_closed_loop_contract(
        workspace_dir=tmp_path,
        force=True,
    )
    assert contract["state_hint"] == "TASK2_FNO_TRAINING_CONTRACT_BUILT"
    assert "run-task2-fno-training-iteration" in contract["required_commands"]
    assert (tmp_path / "task2_fno_training_contract.json").is_file()


def test_build_final_report(tmp_path: Path) -> None:
    memory = {
        "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
        "iteration_index": 2,
        "best_so_far": {
            "candidate_id": "best",
            "candidate_family": "fno_lite_task2",
            "score": 80.0,
            "config": {"modes": 64},
        },
        "trajectory_summary": [
            {"candidate_id": "best", "score": 80.0, "config": {"modes": 64}},
        ],
    }
    memory_path = tmp_path / "task2_fno_memory_0002.json"
    memory_path.write_text(json.dumps(memory))
    report = build_task2_fno_training_final_report(
        memory_path=memory_path,
        workspace_dir=tmp_path,
        force=True,
    )
    assert report["state_hint"] == "TASK2_FNO_TRAINING_FINAL_REPORT_BUILT"
    assert report["best_candidate"]["score"] == 80.0
    assert report["provenance"]["iteration_count"] == 3


def test_build_health_report(tmp_path: Path) -> None:
    # Create two iteration memories
    for i, score in enumerate([70.0, 76.0]):
        memory = {
            "state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT",
            "iteration_index": i,
            "current_best": {"score": score},
            "best_so_far": {"score": max(70.0, score)},
            "trajectory_summary": [],
            "convergence_history": {},
        }
        (tmp_path / f"task2_fno_memory_{i:04d}.json").write_text(json.dumps(memory))
        convergence = {
            "state_hint": "TASK2_FNO_TRAINING_CONVERGENCE_CHECKED",
            "decision": "continue",
        }
        (tmp_path / f"task2_fno_convergence_{i:04d}.json").write_text(json.dumps(convergence))

    report = build_task2_fno_training_health_report(
        workspace_dir=tmp_path,
        force=True,
    )
    assert report["state_hint"] == "TASK2_FNO_TRAINING_HEALTH_REPORT_BUILT"
    assert report["iteration_count"] == 2
    assert report["score"]["global_best"] == 76.0
    assert report["health"]["status"] == "healthy"


def test_build_iteration_report() -> None:
    report = build_task2_fno_training_iteration_report(
        iteration_index=0,
        summary={"best_score": 76.0},
        convergence={"decision": "continue"},
        memory={},
        artifacts={},
    )
    assert report["state_hint"] == "TASK2_FNO_TRAINING_ITERATION_REPORTED"
    assert report["iteration_index"] == 0
