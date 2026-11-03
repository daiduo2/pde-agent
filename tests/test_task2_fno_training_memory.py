"""Tests for Task 2 FNO training memory module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_agent.core.task2_fno_training_memory import (
    Task2FnoTrainingMemoryError,
    build_task2_fno_training_memory,
    load_task2_fno_training_memory,
    merge_task2_training_memory_with_hypothesis_feedback,
)


def _make_opt_report(**overrides: object) -> dict:
    base = {
        "state_hint": "TASK2_FNO_OPTIMIZATION_LOOP_RAN",
        "candidate_records": [
            {
                "candidate_id": "c1",
                "score": 76.0,
                "segment_score": 50.67,
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
            },
            {
                "candidate_id": "c2",
                "score": 70.0,
                "segment_score": 46.67,
                "candidate_family": "fno_lite_task2",
                "config": {
                    "hidden_channels": 64,
                    "modes": 32,
                    "depth": 3,
                    "physics_weight": 0.0,
                    "learning_rate": 0.001,
                    "train_steps": 5000,
                    "batch_size": 16,
                },
            },
        ],
    }
    base.update(overrides)
    return base


def test_build_memory_basic() -> None:
    report = _make_opt_report()
    memory = build_task2_fno_training_memory(
        optimization_report=report,
        iteration_index=0,
        previous_memory=None,
    )
    assert memory["state_hint"] == "TASK2_FNO_TRAINING_MEMORY_BUILT"
    assert memory["iteration_index"] == 0
    assert memory["current_best"]["score"] == 76.0
    assert memory["best_so_far"]["score"] == 76.0
    assert len(memory["trajectory_summary"]) == 2
    assert len(memory["blocked_configs"]) == 0  # score 70 > 0, not blocked
    assert memory["convergence_history"]["no_improvement_count"] == 0


def test_build_memory_invalid_state_hint() -> None:
    report = _make_opt_report(state_hint="INVALID")
    with pytest.raises(Task2FnoTrainingMemoryError):
        build_task2_fno_training_memory(
            optimization_report=report,
            iteration_index=0,
            previous_memory=None,
        )


def test_build_memory_with_previous() -> None:
    first = build_task2_fno_training_memory(
        optimization_report=_make_opt_report(),
        iteration_index=0,
        previous_memory=None,
    )
    # Second iteration with worse score
    worse_report = _make_opt_report(
        candidate_records=[
            {
                "candidate_id": "c3",
                "score": 72.0,
                "segment_score": 48.0,
                "candidate_family": "fno_lite_task2",
                "config": {"hidden_channels": 80, "modes": 40, "depth": 4},
            }
        ]
    )
    second = build_task2_fno_training_memory(
        optimization_report=worse_report,
        iteration_index=1,
        previous_memory=first,
    )
    assert second["best_so_far"]["score"] == 76.0  # preserved from first
    assert second["current_best"]["score"] == 72.0
    assert second["convergence_history"]["no_improvement_count"] == 1


def test_load_memory_missing_file(tmp_path: Path) -> None:
    result = load_task2_fno_training_memory(tmp_path / "nonexistent.json")
    assert result is None


def test_load_memory_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    result = load_task2_fno_training_memory(path)
    assert result is None


def test_load_memory_wrong_state_hint(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"state_hint": "WRONG"}), encoding="utf-8")
    result = load_task2_fno_training_memory(path)
    assert result is None


def test_load_memory_valid(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    data = {"state_hint": "TASK2_FNO_TRAINING_MEMORY_BUILT", "iteration_index": 3}
    path.write_text(json.dumps(data), encoding="utf-8")
    result = load_task2_fno_training_memory(path)
    assert result == data


def test_merge_hypothesis_feedback() -> None:
    memory = build_task2_fno_training_memory(
        optimization_report=_make_opt_report(),
        iteration_index=0,
        previous_memory=None,
    )
    hypothesis_report = {
        "state_hint": "TASK2_FNO_TRAINING_ESCALATION_RECOMMENDATIONS_GENERATED",
        "strategy_recommendations": [
            {
                "strategy": "expand_search_radius",
                "priority": 1,
                "actions": ["Try modes=128"],
            }
        ],
        "blocked_configs": [
            {"candidate_id": "c99", "config": {"modes": 16}, "reason": "bad"}
        ],
        "search_guidance": [
            {"guidance_type": "avoid_config", "reason": "test", "config_signature": "h64_m32"}
        ],
    }
    merged = merge_task2_training_memory_with_hypothesis_feedback(
        memory=memory,
        hypothesis_report=hypothesis_report,
    )
    assert merged["state_hint"] == "TASK2_FNO_TRAINING_MEMORY_BUILT"
    assert any("expand_search_radius" in h for h in merged.get("next_hints", []))
    assert any(b["candidate_id"] == "c99" for b in merged["blocked_configs"])
    assert any(sg["config_signature"] == "h64_m32" for sg in merged["search_guidance"])


def test_merge_hypothesis_feedback_none_report() -> None:
    memory = build_task2_fno_training_memory(
        optimization_report=_make_opt_report(),
        iteration_index=0,
        previous_memory=None,
    )
    merged = merge_task2_training_memory_with_hypothesis_feedback(
        memory=memory,
        hypothesis_report=None,
    )
    assert merged == memory
