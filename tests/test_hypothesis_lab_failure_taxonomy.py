"""Tests for hypothesis_lab_failure_taxonomy."""

from __future__ import annotations

from typing import Any

import pytest

from pde_agent.core.hypothesis_lab_failure_taxonomy import (
    FAILURE_TYPES,
    ROOT_CAUSE_TO_TYPE,
    classify_failure,
    diagnose_session_family,
    _infer_root_cause_hint_from_reasons,
)


class FakeSession:
    """Minimal mock session for diagnose_session_family tests."""

    def __init__(self, artifacts: list[dict[str, Any]] | None = None):
        self._artifacts = artifacts or []

    def list_artifacts(self, artifact_type: str) -> list[dict[str, Any]]:
        return [
            art for art in self._artifacts if art.get("artifact_type") == artifact_type
        ]


def _make_fake_session(artifacts: list[dict[str, Any]]) -> Any:
    """Return a session that passes isinstance check for HypothesisLabSession."""
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    class _FakeSession(HypothesisLabSession):
        def __init__(self, artifacts: list[dict[str, Any]]):
            # Bypass normal init to avoid filesystem setup
            object.__setattr__(self, "_artifacts", artifacts)
            object.__setattr__(self, "session_id", "fake_session_001")

        def list_artifacts(self, artifact_type: str) -> list[dict[str, Any]]:
            return [
                art for art in self._artifacts if art.get("artifact_type") == artifact_type
            ]

        def get_state_summary(self) -> dict[str, Any]:
            return {}

    return _FakeSession(artifacts)


def test_failure_types_structure() -> None:
    assert "PHYSICAL" in FAILURE_TYPES
    assert "NUMERICAL" in FAILURE_TYPES
    assert "STATISTICAL" in FAILURE_TYPES
    assert "OPERATIONAL" in FAILURE_TYPES
    for category, subtypes in FAILURE_TYPES.items():
        assert isinstance(subtypes, dict)
        for key, desc in subtypes.items():
            assert isinstance(key, str)
            assert isinstance(desc, str)


def test_root_cause_to_type_coverage() -> None:
    assert "early_shock_smearing" in ROOT_CAUSE_TO_TYPE
    assert "late_drift" in ROOT_CAUSE_TO_TYPE
    assert "context_corrupted" in ROOT_CAUSE_TO_TYPE
    assert "reference_mismatch" in ROOT_CAUSE_TO_TYPE
    assert "numerical_instability" in ROOT_CAUSE_TO_TYPE
    assert "timeout_risk" in ROOT_CAUSE_TO_TYPE
    for key, (cat, sub) in ROOT_CAUSE_TO_TYPE.items():
        assert cat in FAILURE_TYPES
        assert sub in FAILURE_TYPES[cat]


def test_classify_failure_promoted() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "promote_to_strategy_proposal",
            "metric_results": {},
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "NONE"
    assert result["primary_subtype"] == "ok"
    assert result["root_cause_hint"] == "ok"
    assert result["confidence"] == "high"


def test_classify_failure_timeout() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "reject_or_revise",
            "metric_results": {
                "inference_time_budget": {"passed": False, "value": 150.0, "threshold": 120.0}
            },
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {"reason_codes": ["no_pareto_axis_improved"]},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "OPERATIONAL"
    assert result["primary_subtype"] == "timeout_risk"
    assert result["root_cause_hint"] == "timeout_risk"


def test_classify_failure_context_corrupted() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "reject_or_revise",
            "metric_results": {
                "context_preserved": {"passed": False}
            },
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "OPERATIONAL"
    assert result["primary_subtype"] == "context_corruption"
    assert result["root_cause_hint"] == "context_corrupted"


def test_classify_failure_early_shock() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "reject_or_revise",
            "metric_results": {},
            "early_segment_diagnosis": {
                "reason_codes": ["early_score_low", "early_rel_mse_high"]
            },
            "pareto_promotion_gate": {},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "STATISTICAL"
    assert result["confidence"] == "high"
    assert result["root_cause_hint"] in ("early_score_low", "early_rel_mse_high")


def test_classify_failure_late_drift() -> None:
    sandbox_result = {
        "post_mortem": {"root_cause_hint": "late_drift"},
        "verifier": {
            "decision": "keep_as_evidence",
            "metric_results": {},
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "PHYSICAL"
    assert result["primary_subtype"] == "energy_dissipation_anomaly"
    assert result["root_cause_hint"] == "late_drift"


def test_classify_failure_numerical_instability() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "reject_or_revise",
            "metric_results": {},
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {
                "reason_codes": ["initial_derivative_mismatch"]
            },
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "NUMERICAL"
    assert result["primary_subtype"] == "time_step_sensitivity"
    assert result["root_cause_hint"] == "initial_derivative_mismatch"


def test_classify_failure_unknown_fallback() -> None:
    sandbox_result = {
        "verifier": {
            "decision": "some_unknown_decision",
            "metric_results": {},
            "early_segment_diagnosis": {},
            "pareto_promotion_gate": {},
        }
    }
    result = classify_failure(sandbox_result)
    assert result["primary_type"] == "OPERATIONAL"
    assert result["primary_subtype"] == "reference_mismatch"


def test_infer_root_cause_hint_from_reasons() -> None:
    assert _infer_root_cause_hint_from_reasons([]) == "unknown"
    assert _infer_root_cause_hint_from_reasons(["timeout_risk"]) == "timeout_risk"
    assert _infer_root_cause_hint_from_reasons(["early_shock_smearing", "timeout_risk"]) == "early_shock_smearing"


def test_diagnose_session_family_no_artifacts() -> None:
    session = _make_fake_session([])
    result = diagnose_session_family(session, "task1_retrieval_refinement")
    assert result["family"] == "task1_retrieval_refinement"
    assert result["failure_types"] == {}
    assert result["root_causes"] == []
    assert result["sandbox_count"] == 0


def test_diagnose_session_family_with_artifacts() -> None:
    artifacts = [
        {
            "artifact_type": "sandbox",
            "payload": {
                "best_candidate": {
                    "candidate_family": "task1_retrieval_refinement",
                    "task1_total_score_proxy": 20.0,
                },
                "verifier": {
                    "decision": "reject_or_revise",
                    "metric_results": {
                        "context_preserved": {"passed": False}
                    },
                    "early_segment_diagnosis": {},
                    "pareto_promotion_gate": {},
                },
            },
        },
        {
            "artifact_type": "sandbox",
            "payload": {
                "best_candidate": {
                    "candidate_family": "task1_retrieval_refinement",
                    "task1_total_score_proxy": 110.0,
                },
                "verifier": {
                    "decision": "promote_to_strategy_proposal",
                    "metric_results": {},
                    "early_segment_diagnosis": {},
                    "pareto_promotion_gate": {},
                },
            },
        },
    ]
    session = _make_fake_session(artifacts)
    result = diagnose_session_family(session, "task1_retrieval_refinement")
    assert result["sandbox_count"] == 2
    assert result["failure_types"].get("OPERATIONAL", 0) == 1
    assert "context_corrupted" in result["root_causes"]


def test_diagnose_session_family_different_family() -> None:
    artifacts = [
        {
            "artifact_type": "sandbox",
            "payload": {
                "best_candidate": {
                    "candidate_family": "task1_early_transition_expert",
                    "task1_total_score_proxy": 20.0,
                },
                "verifier": {
                    "decision": "reject_or_revise",
                    "metric_results": {},
                    "early_segment_diagnosis": {
                        "reason_codes": ["early_score_low"]
                    },
                    "pareto_promotion_gate": {},
                },
            },
        },
    ]
    session = _make_fake_session(artifacts)
    result = diagnose_session_family(session, "task1_retrieval_refinement")
    assert result["sandbox_count"] == 0
    assert result["failure_types"] == {}
