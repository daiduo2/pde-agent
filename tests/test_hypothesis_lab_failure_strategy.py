"""Tests for hypothesis_lab_failure_strategy — Layer 3 prescriptive rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pde_agent.core.hypothesis_lab_failure_strategy import (
    _extract_session_diagnosis,
    _family_has_historical_success,
    _family_tried_in_session,
    _get_blocked_families,
    _match_rule_condition,
    extract_rules_from_memory,
    recommend_family_for_sandbox,
    recommend_family_given_diagnosis,
    score_family_for_context,
)
from pde_agent.core.hypothesis_lab_memory import HypothesisLabMemory
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


# -- extract_rules_from_memory --


def test_extract_rules_empty_memory():
    memory = HypothesisLabMemory(None)
    rules = extract_rules_from_memory(memory)
    assert rules == []


def test_extract_rules_no_failures(tmp_path: Path):
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")
    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "fno",
            "task1_total_score_proxy": 80.0,
        }
    })
    rules = extract_rules_from_memory(memory)
    assert rules == []


def test_extract_rules_with_failure_and_success(tmp_path: Path):
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")
    # Failure for family_a
    memory.record_failure_pattern(
        "sess_001",
        "family_a",
        {"root_cause": "early_shock_smearing", "worst_segment": "early"},
    )
    # Success for family_b in same session
    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "family_b",
            "task1_total_score_proxy": 85.0,
        }
    })
    rules = extract_rules_from_memory(memory)
    assert len(rules) == 1
    assert rules[0]["target"] == "family_b"
    assert rules[0]["condition"]["failed_family"] == "family_a"
    assert rules[0]["condition"]["root_cause"] == "early_shock_smearing"
    assert rules[0]["confidence"] == 1.0
    assert rules[0]["support"] == 1


def test_extract_rules_multiple_sessions_boost_confidence(tmp_path: Path):
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")
    for sid in ("sess_001", "sess_002"):
        memory.record_failure_pattern(
            sid,
            "family_a",
            {"root_cause": "oom", "worst_segment": "mid"},
        )
        memory.record_experiment(sid, {
            "best_candidate": {
                "candidate_family": "family_c",
                "task1_total_score_proxy": 90.0,
            }
        })
    rules = extract_rules_from_memory(memory)
    assert len(rules) == 1
    assert rules[0]["target"] == "family_c"
    assert rules[0]["confidence"] == 1.0
    assert rules[0]["support"] == 2
    assert set(rules[0]["supporting_sessions"]) == {"sess_001", "sess_002"}


def test_extract_rules_ignores_same_family_success(tmp_path: Path):
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")
    memory.record_failure_pattern(
        "sess_001",
        "family_a",
        {"root_cause": "timeout", "worst_segment": "late"},
    )
    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "family_a",
            "task1_total_score_proxy": 85.0,
        }
    })
    rules = extract_rules_from_memory(memory)
    assert rules == []


def test_extract_rules_low_confidence_filtered(tmp_path: Path):
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")
    # One failure, one low-score success (doesn't count as success)
    memory.record_failure_pattern(
        "sess_001",
        "family_a",
        {"root_cause": "oom", "worst_segment": "early"},
    )
    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "family_b",
            "task1_total_score_proxy": 10.0,
        }
    })
    rules = extract_rules_from_memory(memory)
    assert rules == []


# -- recommend_family_given_diagnosis --


def test_recommend_family_no_match():
    diagnosis = {"failed_family": "fam_a", "root_cause": "oom"}
    rules = [
        {
            "rule_id": "r1",
            "condition": {"failed_family": "fam_b", "root_cause": "oom"},
            "target": "fam_c",
            "confidence": 0.8,
            "support": 3,
        }
    ]
    result = recommend_family_given_diagnosis(diagnosis, rules, {"fam_c"})
    assert result is None


def test_recommend_family_match():
    diagnosis = {"failed_family": "fam_a", "root_cause": "oom"}
    rules = [
        {
            "rule_id": "r1",
            "condition": {"failed_family": "fam_a", "root_cause": "oom"},
            "target": "fam_c",
            "confidence": 0.8,
            "support": 3,
        }
    ]
    result = recommend_family_given_diagnosis(diagnosis, rules, {"fam_c"})
    assert result is not None
    assert result["target_family"] == "fam_c"
    assert result["confidence"] == 0.8


def test_recommend_family_target_not_available():
    diagnosis = {"failed_family": "fam_a", "root_cause": "oom"}
    rules = [
        {
            "rule_id": "r1",
            "condition": {"failed_family": "fam_a", "root_cause": "oom"},
            "target": "fam_c",
            "confidence": 0.8,
            "support": 3,
        }
    ]
    result = recommend_family_given_diagnosis(diagnosis, rules, {"fam_d"})
    assert result is None


def test_recommend_family_picks_highest_confidence():
    diagnosis = {"failed_family": "fam_a", "root_cause": "oom"}
    rules = [
        {
            "rule_id": "r1",
            "condition": {"failed_family": "fam_a", "root_cause": "oom"},
            "target": "fam_c",
            "confidence": 0.6,
            "support": 3,
        },
        {
            "rule_id": "r2",
            "condition": {"failed_family": "fam_a", "root_cause": "oom"},
            "target": "fam_d",
            "confidence": 0.9,
            "support": 5,
        },
    ]
    result = recommend_family_given_diagnosis(diagnosis, rules, {"fam_c", "fam_d"})
    assert result is not None
    assert result["target_family"] == "fam_d"


# -- score_family_for_context --


def test_score_family_blocked(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    # Add blocked outcome
    session.put_artifact(
        artifact_type="outcome",
        artifact_id="out1",
        payload={
            "pareto_memory": {
                "pareto_learning_events": [
                    {
                        "candidate_family": "fam_a",
                        "outcome": "blocked",
                        "decision": "doom_loop",
                    }
                ]
            }
        },
    )
    score = score_family_for_context(session, "fam_a", [])
    assert score == 0.0  # blocked gets -0.5, clamped to 0


def test_score_family_not_tried_bonus(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    score = score_family_for_context(session, "fam_new", [])
    # +0.2 (not blocked) +0.1 (not tried) = 0.3
    assert abs(score - 0.3) < 1e-9


def test_score_family_rule_bonus(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    rules = [
        {
            "rule_id": "r1",
            "condition": {"failed_family": "fam_a", "root_cause": "oom"},
            "target": "fam_b",
            "confidence": 0.8,
            "support": 3,
        }
    ]
    # Add a failed sandbox so diagnosis matches
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_family": "fam_a",
                "task1_total_score_proxy": 10.0,
            }
        },
    )
    # Add outcome artifact to provide root_cause for diagnosis
    session.put_artifact(
        artifact_type="outcome",
        artifact_id="out1",
        payload={
            "pareto_memory": {
                "pareto_learning_events": [
                    {
                        "candidate_family": "fam_a",
                        "outcome": "failed",
                        "decision": "oom",
                    }
                ]
            }
        },
    )
    score = score_family_for_context(session, "fam_b", rules)
    # +0.4 (high-confidence rule) +0.2 (not blocked) +0.1 (not tried) = 0.7
    assert abs(score - 0.7) < 1e-9


# -- _match_rule_condition --


def test_match_rule_condition_exact():
    rule = {"condition": {"failed_family": "a", "root_cause": "oom"}}
    diagnosis = {"failed_family": "a", "root_cause": "oom"}
    assert _match_rule_condition(rule, diagnosis) is True


def test_match_rule_condition_partial():
    rule = {"condition": {"failed_family": "a"}}
    diagnosis = {"failed_family": "a", "root_cause": "oom"}
    assert _match_rule_condition(rule, diagnosis) is True


def test_match_rule_condition_mismatch():
    rule = {"condition": {"failed_family": "a", "root_cause": "oom"}}
    diagnosis = {"failed_family": "a", "root_cause": "timeout"}
    assert _match_rule_condition(rule, diagnosis) is False


def test_match_rule_condition_empty():
    rule = {"condition": {}}
    diagnosis = {"failed_family": "a"}
    assert _match_rule_condition(rule, diagnosis) is True


# -- recommend_family_for_sandbox --


def test_recommend_family_for_sandbox_no_pending(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    result = recommend_family_for_sandbox(session, None)
    assert result["action"] == "report_results"
    assert "No pending blueprints" in result["reasoning"]


def test_recommend_family_for_sandbox_fallback(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={
            "blueprints": [
                {"blueprint_id": "b1", "candidate_family": "fam_a"},
                {"blueprint_id": "b2", "candidate_family": "fam_b"},
            ]
        },
    )
    result = recommend_family_for_sandbox(session, None)
    assert result["action"] == "run_blueprint_sandbox"
    assert result["recommended_index"] == 0
    assert result["recommended_family"] == "fam_a"
    assert result["confidence"] >= 0.0


def test_recommend_family_for_sandbox_with_rule(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    memory = HypothesisLabMemory(tmp_path / "mem.jsonl")

    # Record failure -> success rule
    memory.record_failure_pattern(
        "sess_001",
        "fam_a",
        {"root_cause": "early_shock_smearing", "worst_segment": "early"},
    )
    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "fam_b",
            "task1_total_score_proxy": 85.0,
        }
    })

    # Set up session with failed sandbox for fam_a
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_family": "fam_a",
                "task1_total_score_proxy": 10.0,
            }
        },
    )
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={
            "blueprints": [
                {"blueprint_id": "b1", "candidate_family": "fam_a"},
                {"blueprint_id": "b2", "candidate_family": "fam_b"},
            ]
        },
    )

    result = recommend_family_for_sandbox(session, memory)
    assert result["action"] == "run_blueprint_sandbox"
    assert result["recommended_family"] == "fam_b"
    assert result["recommended_index"] == 1
    assert result["confidence"] > 0.0


def test_recommend_family_for_sandbox_skips_sandboxed(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={
            "blueprints": [
                {"blueprint_id": "b1", "candidate_family": "fam_a"},
                {"blueprint_id": "b2", "candidate_family": "fam_b"},
            ]
        },
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={"blueprint_id": "b1"},
    )
    result = recommend_family_for_sandbox(session, None)
    assert result["action"] == "run_blueprint_sandbox"
    assert result["recommended_index"] == 1
    assert result["recommended_family"] == "fam_b"


# -- helpers --


def test_extract_session_diagnosis_empty(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    diag = _extract_session_diagnosis(session)
    assert diag == {}


def test_extract_session_diagnosis_from_sandbox(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_family": "fam_a",
                "task1_total_score_proxy": 10.0,
                "segment_score": {"early": 5.0, "mid": 15.0, "late": 10.0},
            }
        },
    )
    diag = _extract_session_diagnosis(session)
    assert diag["failed_family"] == "fam_a"
    assert diag["worst_segment"] == "early"


def test_family_has_historical_success(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_family": "fam_a",
                "task1_total_score_proxy": 85.0,
            }
        },
    )
    assert _family_has_historical_success(session, "fam_a") is True
    assert _family_has_historical_success(session, "fam_b") is False


def test_family_tried_in_session(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_family": "fam_a",
                "task1_total_score_proxy": 50.0,
            }
        },
    )
    assert _family_tried_in_session(session, "fam_a") is True
    assert _family_tried_in_session(session, "fam_b") is False


def test_get_blocked_families(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(
        artifact_type="outcome",
        artifact_id="out1",
        payload={
            "pareto_memory": {
                "pareto_learning_events": [
                    {
                        "candidate_family": "fam_a",
                        "outcome": "blocked",
                        "decision": "doom_loop",
                    }
                ]
            }
        },
    )
    blocked = _get_blocked_families(session)
    assert "fam_a" in blocked


def test_get_blocked_families_empty(tmp_path: Path):
    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    blocked = _get_blocked_families(session)
    assert blocked == set()
