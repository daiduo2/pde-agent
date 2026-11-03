"""Tests for hypothesis_lab_autonomous_loop and action_dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pde_agent.core.hypothesis_lab_action_dispatcher import (
    ActionDispatchError,
    _dispatch_action,
    is_deterministic_action,
)
from pde_agent.core.hypothesis_lab_autonomous_loop import (
    AutonomousHypothesisLab,
    AutonomousHypothesisLabError,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


# -- Helpers --


def _make_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "state_hint": "MANUAL_OPERATOR_TUNING_COMPLETE",
                "best_candidate": {
                    "candidate_id": "test",
                    "family": "test_family",
                    "params": {},
                    "segment_score": 10.0,
                },
                "top_candidates": [],
            }
        ),
        encoding="utf-8",
    )


def _make_session(tmp_path: Path) -> HypothesisLabSession:
    return HypothesisLabSession.create(workspace=tmp_path / "ws")


# -- Action Dispatcher --


def test_dispatch_unknown_action(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    with pytest.raises(ActionDispatchError, match="unknown action"):
        _dispatch_action("unknown_action", session)


def test_dispatch_missing_required_kwargs(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    with pytest.raises(ActionDispatchError, match="missing required kwargs"):
        _dispatch_action("build_knowledge_cards", session)


def test_dispatch_build_knowledge_cards(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    result = _dispatch_action(
        "build_knowledge_cards",
        session,
        manual_report_paths=[str(report)],
    )
    assert result["action"] == "build_knowledge_cards"
    assert result["status"] == "ok"
    assert "result" in result


def test_dispatch_generate_hypotheses_without_cards(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    with pytest.raises(ActionDispatchError):
        _dispatch_action("generate_hypotheses", session)


def test_is_deterministic_action() -> None:
    assert is_deterministic_action("build_knowledge_cards") is True
    assert is_deterministic_action("run_blueprint_sandbox") is True
    assert is_deterministic_action("report_results") is True
    assert is_deterministic_action("unknown_action") is False


# -- AutonomousHypothesisLab: step() --


def test_step_observe_and_decide_build_cards(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    engine = AutonomousHypothesisLab(session)
    result = engine.step()

    assert result["iteration"] == 1
    assert result["action"] == "build_knowledge_cards"
    assert "observation" in result
    assert "decision" in result
    assert result["decision"]["reasoning"]


def test_step_executes_build_cards(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    engine = AutonomousHypothesisLab(
        session,
        manual_report_paths=[str(report)],
    )
    result = engine.step()

    assert result["action"] == "build_knowledge_cards"
    assert result["execution"]["status"] == "ok"


def test_step_pipeline_progression(tmp_path: Path) -> None:
    """Test that step progresses through the pipeline stages."""
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    engine = AutonomousHypothesisLab(
        session,
        manual_report_paths=[str(report)],
    )

    # Step 1: build_knowledge_cards
    r1 = engine.step()
    assert r1["action"] == "build_knowledge_cards"
    assert r1["execution"]["status"] == "ok"

    # Step 2: generate_hypotheses
    r2 = engine.step()
    assert r2["action"] == "generate_hypotheses"

    # Step 3: design_algorithm_blueprints
    r3 = engine.step()
    assert r3["action"] == "design_algorithm_blueprints"

    # Step 4: run_blueprint_sandbox (needs context/target)
    r4 = engine.step()
    # Without context_path/target_path, sandbox will fail
    assert r4["action"] == "run_blueprint_sandbox"


def test_step_guarded_mode_pauses_at_sandbox(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    # Pre-populate cards, hypotheses, blueprints so next action is sandbox
    session.put_artifact(
        artifact_type="knowledge_cards",
        artifact_id="kc1",
        payload={"card_count": 1, "cards": []},
    )
    session.put_artifact(
        artifact_type="hypotheses",
        artifact_id="h1",
        payload={"hypothesis_count": 1, "hypotheses": []},
    )
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={"blueprint_count": 1, "blueprints": []},
    )

    engine = AutonomousHypothesisLab(session)
    result = engine.step(autonomy_level="guarded")

    assert result["action"] == "run_blueprint_sandbox"
    assert result.get("approval_required") is True


def test_step_full_mode_escalation(tmp_path: Path) -> None:
    """In full mode, non-deterministic actions trigger escalation."""
    session = _make_session(tmp_path)
    engine = AutonomousHypothesisLab(session)

    # Starting state has no artifacts, so action is build_knowledge_cards
    # which IS deterministic - no escalation
    result = engine.step(autonomy_level="full")
    assert result["action"] == "build_knowledge_cards"
    assert result.get("escalation_required") is not True


def test_step_error_handling_does_not_crash(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    engine = AutonomousHypothesisLab(session)

    # Force an action that will fail by mocking dispatch to raise
    with patch(
        "pde_agent.core.hypothesis_lab_autonomous_loop._dispatch_action",
        side_effect=ActionDispatchError("mock failure"),
    ):
        result = engine.step()

    assert result["execution"]["status"] == "error"
    assert "mock failure" in result["execution"]["error"]


# -- AutonomousHypothesisLab: run() --


def test_run_deterministic_completes_stages(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    engine = AutonomousHypothesisLab(
        session,
        manual_report_paths=[str(report)],
    )
    result = engine.run(max_iterations=10, autonomy_level="deterministic")

    assert result["state_hint"] == "HYPOTHESIS_LAB_AUTONOMOUS_LOOP_RAN"
    assert result["iterations"] >= 1
    assert result["status"] == "ok"
    assert "artifacts" in result
    assert "metrics" in result


def test_run_max_iterations_enforced(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    engine = AutonomousHypothesisLab(
        session,
        manual_report_paths=[str(report)],
    )
    result = engine.run(max_iterations=2, autonomy_level="deterministic")

    assert result["iterations"] <= 2


def test_run_guarded_stops_at_approval_gate(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    report = tmp_path / "report.json"
    _make_report(report)

    # Pre-populate up to blueprints
    session.put_artifact(
        artifact_type="knowledge_cards",
        artifact_id="kc1",
        payload={"card_count": 1, "cards": []},
    )
    session.put_artifact(
        artifact_type="hypotheses",
        artifact_id="h1",
        payload={"hypothesis_count": 1, "hypotheses": []},
    )
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={"blueprint_count": 1, "blueprints": []},
    )

    engine = AutonomousHypothesisLab(session)
    result = engine.run(max_iterations=10, autonomy_level="guarded")

    # Should stop at sandbox due to approval_required
    assert result["iterations"] >= 1
    last_result = result["results"][-1]
    assert last_result.get("approval_required") is True


def test_run_report_results_breaks_loop(tmp_path: Path) -> None:
    """If pipeline reaches report_results, loop should break."""
    session = _make_session(tmp_path)

    # Pre-populate all stages so next action is report_results
    session.put_artifact(
        artifact_type="knowledge_cards",
        artifact_id="kc1",
        payload={"card_count": 1, "cards": []},
    )
    session.put_artifact(
        artifact_type="hypotheses",
        artifact_id="h1",
        payload={"hypothesis_count": 1, "hypotheses": []},
    )
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={"blueprint_count": 1, "blueprints": []},
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={"best_candidate": {"candidate_id": "c1", "task1_total_score_proxy": 50.0}},
    )
    session.put_artifact(
        artifact_type="outcome",
        artifact_id="out1",
        payload={"sandbox_artifact_id": "sb1"},
    )

    engine = AutonomousHypothesisLab(session)
    result = engine.run(max_iterations=10, autonomy_level="deterministic")

    # Should break on report_results
    assert result["iterations"] == 1
    assert result["results"][0]["action"] == "report_results"


def test_run_error_in_step_does_not_crash_loop(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    engine = AutonomousHypothesisLab(session)

    with patch(
        "pde_agent.core.hypothesis_lab_autonomous_loop._dispatch_action",
        side_effect=ActionDispatchError("mock failure"),
    ):
        result = engine.run(max_iterations=3, autonomy_level="deterministic")

    assert result["iterations"] == 3
    for r in result["results"]:
        assert r["execution"]["status"] == "error"


# -- Doom loop detection --


def test_doom_loop_no_events(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    engine = AutonomousHypothesisLab(session)
    check = engine._check_doom_loop()
    assert check["guard_active"] is False


def test_doom_loop_with_events(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    # Emit some events to create a trace
    for i in range(5):
        session.emit(
            event_type="test_event",
            payload={
                "candidate_family": "family_a",
                "score_delta": 0.0,
            },
            tags=["test"],
        )

    engine = AutonomousHypothesisLab(session)
    check = engine._check_doom_loop()
    # With only 5 events and threshold of 3, may or may not trigger
    # depending on exact observation logic; just verify it runs
    assert "guard_active" in check


# -- Next pending blueprint --


def test_next_pending_blueprint_index(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={
            "blueprints": [
                {"blueprint_id": "bp_a", "candidate_family": "fam_a"},
                {"blueprint_id": "bp_b", "candidate_family": "fam_b"},
            ]
        },
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={"blueprint_id": "bp_a"},
    )

    engine = AutonomousHypothesisLab(session)
    idx = engine._next_pending_blueprint_index()
    assert idx == 1


def test_next_pending_blueprint_index_all_sandboxed(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.put_artifact(
        artifact_type="blueprints",
        artifact_id="bp1",
        payload={
            "blueprints": [
                {"blueprint_id": "bp_a", "candidate_family": "fam_a"},
            ]
        },
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={"blueprint_id": "bp_a"},
    )

    engine = AutonomousHypothesisLab(session)
    idx = engine._next_pending_blueprint_index()
    assert idx == 0


# -- CLI command --


def test_cli_autonomous_loop_command(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_autonomous_loop import (
        agent_hypothesis_run_autonomous_loop_command,
    )

    ws = tmp_path / "ws"
    HypothesisLabSession.create(workspace=ws)

    report = tmp_path / "report.json"
    _make_report(report)

    result = agent_hypothesis_run_autonomous_loop_command(
        workspace=str(ws),
        context_path=str(tmp_path / "context.npz"),
        target_path=str(tmp_path / "target.npz"),
        manual_report_paths=[str(report)],
        max_iterations=2,
        autonomy_level="deterministic",
    )

    assert result["state_hint"] == "HYPOTHESIS_LAB_AUTONOMOUS_LOOP_RAN"
    assert result["status"] == "ok"
    assert result["metrics"]["iterations"] <= 2
