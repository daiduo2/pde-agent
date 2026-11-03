"""Tests for hypothesis_lab_creative_escalation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pde_agent.core.hypothesis_lab_creative_escalation import (
    build_agent_generated_blueprint,
    build_creative_escalation_context,
    parse_generated_executor,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


# -- Helpers --


def _make_session(tmp_path: Path) -> HypothesisLabSession:
    return HypothesisLabSession.create(workspace=tmp_path / "ws")


def _add_sandbox_artifact(
    session: HypothesisLabSession,
    family: str,
    score: float,
    artifact_id: str,
) -> None:
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id=artifact_id,
        payload={
            "best_candidate": {
                "candidate_id": f"c_{artifact_id}",
                "candidate_family": family,
                "task1_total_score_proxy": score,
            }
        },
    )


# -- build_creative_escalation_context --


def test_build_context_returns_expected_keys(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "task1_retrieval_prior", 50.0, "sb1")

    ctx = build_creative_escalation_context(session)

    assert "session_summary" in ctx
    assert "best_evidence" in ctx
    assert "failure_patterns" in ctx
    assert "blocked_families" in ctx
    assert "score_trajectory" in ctx
    assert "proven_strategies" in ctx
    assert "prompt_template" in ctx
    assert "memory_failures" in ctx


def test_build_context_session_summary(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "task1_retrieval_prior", 50.0, "sb1")

    ctx = build_creative_escalation_context(session)
    summary = ctx["session_summary"]

    assert summary["session_id"] == session.session_id
    assert summary["total_artifacts"] == 1
    assert summary["current_stage"] == "sandbox_complete"


def test_build_context_best_evidence(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "task1_retrieval_prior", 75.0, "sb1")
    _add_sandbox_artifact(session, "task1_retrieval_refinement", 60.0, "sb2")

    ctx = build_creative_escalation_context(session)
    best = ctx["best_evidence"]

    assert best is not None
    assert best["candidate_family"] == "task1_retrieval_prior"
    assert best["score"] == 75.0


def test_build_context_failure_patterns(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "task1_retrieval_prior", 10.0, "sb1")

    ctx = build_creative_escalation_context(session)
    # Low score may not register as failure via get_failure_pattern
    # but the structure should be present
    assert isinstance(ctx["failure_patterns"], dict)


def test_build_context_blocked_families_from_events(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.emit(
        event_type="doom_loop_detected",
        payload={"blocked_families": ["family_a", "family_b"]},
        tags=["doom_loop"],
    )

    ctx = build_creative_escalation_context(session)
    assert ctx["blocked_families"] == ["family_a", "family_b"]


def test_build_context_score_trajectory(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "fam_a", 50.0, "sb1")
    _add_sandbox_artifact(session, "fam_b", 55.0, "sb2")

    ctx = build_creative_escalation_context(session)
    traj = ctx["score_trajectory"]

    assert len(traj) == 2
    scores = [t["score"] for t in traj]
    assert 50.0 in scores
    assert 55.0 in scores
    # Iteration numbers are sequential
    assert sorted(t["iteration"] for t in traj) == [1, 2]


def test_build_context_with_memory(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "fam_a", 50.0, "sb1")

    memory_path = tmp_path / "memory.jsonl"
    from pde_agent.core.hypothesis_lab_memory import HypothesisLabMemory

    memory = HypothesisLabMemory(str(memory_path))
    memory.record_failure_pattern(
        session_id=session.session_id,
        family="fam_a",
        pattern={"failure_reasons": {"score_regressed": 1}},
    )
    memory.record_hypothesis(
        session_id=session.session_id,
        hypothesis={"name": "h1", "status": "proven", "promoted": True},
    )

    ctx = build_creative_escalation_context(session, memory=memory)

    assert len(ctx["proven_strategies"]) == 1
    assert ctx["memory_failures"]["fam_a"]["count"] == 1


def test_build_context_emits_event(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _add_sandbox_artifact(session, "fam_a", 50.0, "sb1")

    build_creative_escalation_context(session)

    events = session.get_events(event_type="creative_escalation_triggered")
    assert len(events) == 1
    assert events[0]["payload"]["session_id"] == session.session_id


def test_build_context_empty_session(tmp_path: Path) -> None:
    session = _make_session(tmp_path)

    ctx = build_creative_escalation_context(session)

    assert ctx["session_summary"]["total_artifacts"] == 0
    assert ctx["best_evidence"] is None
    assert ctx["score_trajectory"] == []
    assert ctx["blocked_families"] == []


# -- parse_generated_executor --


def test_parse_extracts_code_from_markdown() -> None:
    response = '''Here is your code:

```python
def predict(context, target=None, reference=None, **kwargs):
    import numpy as np
    return np.zeros_like(context[:, :1, :])
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is True
    assert result["issues"] == []
    assert "def predict" in result["executor_source"]


def test_parse_extracts_code_without_language() -> None:
    response = '''```
def predict(context, **kwargs):
    return context[:, :1, :]
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is True
    assert result["issues"] == []


def test_parse_rejects_missing_predict() -> None:
    response = '''```python
def foo(context, **kwargs):
    return context
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is False
    assert "missing_predict_function" in result["issues"]


def test_parse_rejects_bad_first_arg() -> None:
    response = '''```python
def predict(data, **kwargs):
    return data
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is False
    assert "predict_first_arg_must_be_context" in result["issues"]


def test_parse_rejects_forbidden_calls() -> None:
    response = '''```python
def predict(context, **kwargs):
    eval("1+1")
    return context
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is False
    assert any("forbidden_call:eval" in i for i in result["issues"])


def test_parse_rejects_no_code_block() -> None:
    result = parse_generated_executor("Just some text without code.")
    assert result["valid"] is False
    assert "no_python_code_block_found" in result["issues"]


def test_parse_rejects_empty_code_block() -> None:
    result = parse_generated_executor("```python\n\n```")
    assert result["valid"] is False
    assert "empty_code_block" in result["issues"]


def test_parse_detects_multiple_forbidden_calls() -> None:
    response = '''```python
def predict(context, **kwargs):
    eval("bad")
    exec("worse")
    open("file.txt")
    return context
```
'''
    result = parse_generated_executor(response)
    assert result["valid"] is False
    assert len(result["issues"]) == 3


# -- build_agent_generated_blueprint --


def test_build_blueprint_structure() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(source)

    assert blueprint["blueprint_id"].startswith("bp_agent_generated_")
    assert blueprint["hypothesis_id"] == "agent_generated_hypothesis_001"
    assert blueprint["candidate_family"] == "agent_generated_executor"
    assert "parameter_grid" in blueprint
    assert "sandbox" in blueprint


def test_build_blueprint_parameter_grid() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(source)

    params = blueprint["parameter_grid"]["candidate_params"]
    assert len(params) == 1
    assert params[0]["executor_source"] == source


def test_build_blueprint_sandbox_defaults() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(source)

    sandbox = blueprint["sandbox"]
    assert sandbox["allowed_executor"] == "agent_generated_executor"
    assert sandbox["primary_metric"] == "task1_total_score_proxy"
    assert sandbox["promotion_min_score_delta"] == 0.01
    assert sandbox["max_inference_seconds"] == 60.0


def test_build_blueprint_custom_hypothesis_id() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(
        source,
        hypothesis_id="custom_hyp_42",
    )
    assert blueprint["hypothesis_id"] == "custom_hyp_42"


def test_build_blueprint_custom_sandbox_kwargs() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(
        source,
        max_inference_seconds=120.0,
        promotion_min_score_delta=0.05,
    )
    assert blueprint["sandbox"]["max_inference_seconds"] == 120.0
    assert blueprint["sandbox"]["promotion_min_score_delta"] == 0.05


def test_build_blueprint_unique_ids() -> None:
    source = "def predict(context, **kwargs):\n    return context"
    bp1 = build_agent_generated_blueprint(source)
    bp2 = build_agent_generated_blueprint(source)
    assert bp1["blueprint_id"] != bp2["blueprint_id"]


# -- Integration: blueprint is compatible with sandbox runner --


def test_blueprint_compatible_with_sandbox_families() -> None:
    from pde_agent.core.hypothesis_lab_sandbox_runner import (
        SANDBOX_EXECUTABLE_FAMILIES,
    )

    source = "def predict(context, **kwargs):\n    return context"
    blueprint = build_agent_generated_blueprint(source)

    assert blueprint["candidate_family"] in SANDBOX_EXECUTABLE_FAMILIES
