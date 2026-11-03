"""Tests for HypothesisLabMemory cross-session memory layer."""

from __future__ import annotations

import json

import pytest

from pde_agent.core.hypothesis_lab_memory import (
    HypothesisLabMemory,
    _dominates,
    _lessons_recommendation,
    get_memory_path_from_env,
)


# -- CRUD operations --


def test_record_and_read_experiment(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    memory.record_experiment("sess_001", {"score": 42.0, "family": "fno"})

    records = memory._read_all()
    assert len(records) == 1
    assert records[0]["record_type"] == "experiment"
    assert records[0]["session_id"] == "sess_001"
    assert records[0]["outcome"]["score"] == 42.0


def test_record_failure_pattern(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    memory.record_failure_pattern(
        "sess_001",
        "task1_retrieval",
        {"reason": "context_mismatch", "count": 3},
    )

    failures = memory.get_similar_failures("task1_retrieval")
    assert len(failures) == 1
    assert failures[0]["family"] == "task1_retrieval"
    assert failures[0]["pattern"]["reason"] == "context_mismatch"


def test_record_candidate(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    memory.record_candidate(
        "sess_002",
        {
            "candidate_id": "c_01",
            "candidate_family": "spectral",
            "score": 88.5,
        },
    )

    records = memory._read_all()
    assert len(records) == 1
    assert records[0]["record_type"] == "candidate"
    assert records[0]["candidate"]["candidate_id"] == "c_01"


def test_record_hypothesis(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    memory.record_hypothesis(
        "sess_003",
        {"hypothesis_id": "h_01", "status": "proven", "mechanism": "residual"},
    )

    proven = memory.get_proven_hypotheses()
    assert len(proven) == 1
    assert proven[0]["hypothesis"]["hypothesis_id"] == "h_01"


# -- Cross-session aggregation --


def test_get_similar_failures_aggregates_across_sessions(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    # Session 1
    memory.record_failure_pattern(
        "sess_001", "family_a", {"reason": "oom", "count": 1}
    )
    # Session 2
    memory.record_failure_pattern(
        "sess_002", "family_a", {"reason": "timeout", "count": 1}
    )
    # Session 3 — different family
    memory.record_failure_pattern(
        "sess_003", "family_b", {"reason": "oom", "count": 1}
    )

    failures = memory.get_similar_failures("family_a")
    assert len(failures) == 2
    sessions = {f["session_id"] for f in failures}
    assert sessions == {"sess_001", "sess_002"}


def test_get_family_performance_history(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "fno",
            "candidate_id": "c1",
            "task1_total_score_proxy": 50.0,
            "segment_score": 25.0,
        }
    })
    memory.record_experiment("sess_002", {
        "best_candidate": {
            "candidate_family": "fno",
            "candidate_id": "c2",
            "task1_total_score_proxy": 75.0,
            "segment_score": 35.0,
        }
    })
    memory.record_experiment("sess_003", {
        "best_candidate": {
            "candidate_family": "unet",
            "candidate_id": "c3",
            "task1_total_score_proxy": 60.0,
        }
    })

    history = memory.get_family_performance_history("fno")
    assert len(history) == 2
    scores = [h["score"] for h in history]
    assert scores == [50.0, 75.0]


def test_get_proven_hypotheses_filters_by_status(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    memory.record_hypothesis("sess_001", {"status": "proven", "claim": "A"})
    memory.record_hypothesis("sess_002", {"status": "disproven", "claim": "B"})
    memory.record_hypothesis("sess_003", {"promoted": True, "claim": "C"})
    memory.record_hypothesis("sess_004", {"status": "pending", "claim": "D"})

    proven = memory.get_proven_hypotheses()
    assert len(proven) == 2
    claims = {p["hypothesis"]["claim"] for p in proven}
    assert claims == {"A", "C"}


# -- Global Pareto front --


def test_get_global_pareto_front_basic(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    # Candidate A: score=80, runtime=10 — dominated by B, not on front
    memory.record_candidate("sess_001", {
        "candidate_id": "c_a",
        "candidate_family": "fno",
        "score": 80.0,
        "inference_seconds": 10.0,
    })
    # Candidate B: score=90, runtime=5 — dominates A, on front
    memory.record_candidate("sess_002", {
        "candidate_id": "c_b",
        "candidate_family": "fno",
        "score": 90.0,
        "inference_seconds": 5.0,
    })
    # Candidate C: score=70, runtime=15 — dominated by A and B
    memory.record_candidate("sess_003", {
        "candidate_id": "c_c",
        "candidate_family": "unet",
        "score": 70.0,
        "inference_seconds": 15.0,
    })

    pareto = memory.get_global_pareto_front()
    ids = {p["candidate_id"] for p in pareto}
    assert ids == {"c_b"}  # only B is non-dominated
    assert pareto[0]["candidate_id"] == "c_b"


def test_get_global_pareto_front_from_experiments(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_id": "c1",
            "candidate_family": "fno",
            "task1_total_score_proxy": 100.0,
            "inference_seconds": 2.0,
        }
    })
    memory.record_experiment("sess_002", {
        "best_candidate": {
            "candidate_id": "c2",
            "candidate_family": "unet",
            "task1_total_score_proxy": 95.0,
            "inference_seconds": 1.0,
        }
    })

    pareto = memory.get_global_pareto_front()
    assert len(pareto) == 2


def test_get_global_pareto_front_dedupes_by_candidate_id(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    # Same candidate recorded twice
    memory.record_candidate("sess_001", {
        "candidate_id": "c1",
        "score": 80.0,
        "inference_seconds": 10.0,
    })
    memory.record_candidate("sess_002", {
        "candidate_id": "c1",
        "score": 85.0,
        "inference_seconds": 12.0,
    })

    pareto = memory.get_global_pareto_front()
    c1_entries = [p for p in pareto if p["candidate_id"] == "c1"]
    assert len(c1_entries) == 1


# -- Lessons learned --


def test_get_lessons_learned_summary(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")

    memory.record_experiment("sess_001", {
        "best_candidate": {
            "candidate_family": "fno",
            "task1_total_score_proxy": 50.0,
        }
    })
    memory.record_experiment("sess_002", {
        "best_candidate": {
            "candidate_family": "fno",
            "task1_total_score_proxy": 60.0,
        }
    })
    memory.record_failure_pattern(
        "sess_001", "fno", {"failure_reasons": {"oom": 2, "timeout": 1}}
    )
    memory.record_failure_pattern(
        "sess_002", "fno", {"failure_reasons": {"oom": 1}}
    )

    lessons = memory.get_lessons_learned(n=5)
    assert lessons["total_sessions"] == 2
    assert lessons["total_experiments"] == 2
    assert lessons["total_failures_recorded"] == 2

    fno_stats = lessons["family_performance_summary"]["fno"]
    assert fno_stats["count"] == 2
    assert fno_stats["best"] == 60.0

    top_patterns = lessons["top_failure_patterns"]
    assert len(top_patterns) == 2
    assert top_patterns[0]["reason"] == "oom"
    assert top_patterns[0]["count"] == 3


def test_get_lessons_learned_empty_memory(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    lessons = memory.get_lessons_learned()
    assert lessons["total_sessions"] == 0
    assert lessons["total_experiments"] == 0
    assert lessons["recommendation"] == "no_data"


# -- JSONL backend integrity --


def test_jsonl_append_creates_file(tmp_path):
    path = tmp_path / "new_memory.jsonl"
    memory = HypothesisLabMemory(path)
    memory.record_experiment("sess_001", {"score": 1.0})
    assert path.exists()


def test_jsonl_handles_corrupt_lines(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(
        '{"record_type": "experiment", "session_id": "s1"}\n'
        'this is not json\n'
        '{"record_type": "experiment", "session_id": "s2"}\n'
    )
    memory = HypothesisLabMemory(path)
    records = memory._read_all()
    assert len(records) == 2
    assert records[0]["session_id"] == "s1"
    assert records[1]["session_id"] == "s2"


def test_jsonl_handles_empty_file(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text("")
    memory = HypothesisLabMemory(path)
    records = memory._read_all()
    assert records == []


def test_jsonl_handles_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    memory = HypothesisLabMemory(path)
    records = memory._read_all()
    assert records == []


def test_multiple_record_types_in_one_file(tmp_path):
    memory = HypothesisLabMemory(tmp_path / "memory.jsonl")
    memory.record_experiment("s1", {"score": 10})
    memory.record_failure_pattern("s1", "fam", {"r": "x"})
    memory.record_candidate("s2", {"id": "c1"})
    memory.record_hypothesis("s3", {"claim": "h1"})

    records = memory._read_all()
    types = [r["record_type"] for r in records]
    assert types == ["experiment", "failure", "candidate", "hypothesis"]


# -- Helpers --


def test_dominates_better_score_and_runtime():
    q = {"score": 90.0, "runtime": 5.0}
    p = {"score": 80.0, "runtime": 10.0}
    assert _dominates(q, p) is True
    assert _dominates(p, q) is False


def test_dominates_no_runtime():
    q = {"score": 90.0, "runtime": None}
    p = {"score": 80.0, "runtime": 10.0}
    assert _dominates(q, p) is True


def test_dominates_equal_score():
    q = {"score": 80.0, "runtime": 5.0}
    p = {"score": 80.0, "runtime": 10.0}
    assert _dominates(q, p) is False  # not strictly better score


def test_lessons_recommendation_focus_on_best_family():
    stats = {"fno": {"count": 5, "best": 120.0, "mean": 100.0}}
    reasons = {}
    assert _lessons_recommendation(stats, reasons) == "focus_on_fno"


def test_lessons_recommendation_address_top_failure():
    stats = {"fno": {"count": 5, "best": 50.0, "mean": 40.0}}
    reasons = {"oom": 10, "timeout": 3}
    assert _lessons_recommendation(stats, reasons) == "address_oom"


def test_lessons_recommendation_continue_exploration():
    stats = {"fno": {"count": 5, "best": 50.0, "mean": 40.0}}
    reasons = {}
    assert _lessons_recommendation(stats, reasons) == "continue_exploration"


# -- Environment path --


def test_get_memory_path_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("PDE_AGENT_HYPOTHESIS_MEMORY_PATH", raising=False)
    assert get_memory_path_from_env() is None


def test_get_memory_path_from_env_reads_env(monkeypatch, tmp_path):
    path = str(tmp_path / "mem.jsonl")
    monkeypatch.setenv("PDE_AGENT_HYPOTHESIS_MEMORY_PATH", path)
    result = get_memory_path_from_env()
    assert result is not None
    assert str(result) == path


# -- Default path --


def test_default_memory_path():
    from pde_agent.core.hypothesis_lab_memory import DEFAULT_MEMORY_PATH
    assert ".pde_agent" in str(DEFAULT_MEMORY_PATH)
    assert "hypothesis_lab_memory.jsonl" in str(DEFAULT_MEMORY_PATH)
