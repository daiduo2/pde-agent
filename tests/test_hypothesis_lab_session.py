from __future__ import annotations

import json
from pathlib import Path

import pytest


# -- ArtifactStore --


def test_artifact_store_put_and_get(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    ref = store.put(artifact_type="test", artifact_id="a1", payload={"x": 1})

    assert ref.artifact_type == "test"
    assert ref.artifact_id == "a1"
    assert ref.version == 1

    payload = store.get("test", "a1")
    assert payload == {"x": 1}


def test_artifact_store_versioning(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    store.put(artifact_type="test", artifact_id="a1", payload={"x": 1})
    ref2 = store.put(artifact_type="test", artifact_id="a1", payload={"x": 2})

    assert ref2.version == 2
    assert store.get("test", "a1") == {"x": 2}


def test_artifact_store_get_missing_returns_none(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    assert store.get("test", "missing") is None


def test_artifact_store_list_by_type(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    store.put(artifact_type="cards", artifact_id="c1", payload={})
    store.put(artifact_type="cards", artifact_id="c2", payload={})
    store.put(artifact_type="hypotheses", artifact_id="h1", payload={})

    refs = store.list_by_type("cards")
    assert len(refs) == 2
    ids = {r.artifact_id for r in refs}
    assert ids == {"c1", "c2"}


def test_artifact_store_exists_and_delete(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    store.put(artifact_type="test", artifact_id="a1", payload={})

    assert store.exists("test", "a1") is True
    assert store.exists("test", "a2") is False

    assert store.delete("test", "a1") is True
    assert store.exists("test", "a1") is False
    assert store.delete("test", "a1") is False


def test_artifact_store_manifest(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import HypothesisLabArtifactStore

    store = HypothesisLabArtifactStore(tmp_path)
    store.put(artifact_type="test", artifact_id="a1", payload={"x": 1})

    manifest = store.get_manifest()
    assert manifest["schema_version"] == 1
    assert "test:a1" in manifest["artifacts"]


def test_artifact_store_invalid_input(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_artifact_store import (
        ArtifactStoreError,
        HypothesisLabArtifactStore,
    )

    store = HypothesisLabArtifactStore(tmp_path)

    with pytest.raises(ArtifactStoreError):
        store.put(artifact_type="", artifact_id="a1", payload={})

    with pytest.raises(ArtifactStoreError):
        store.put(artifact_type="test", artifact_id="", payload={})

    with pytest.raises(ArtifactStoreError):
        store.put(artifact_type="test", artifact_id="a1", payload="not_a_dict")


# -- EventStream --


def test_event_stream_emit_and_read(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    evt = stream.emit(event_type="stage_started", payload={"stage": "test"})

    assert evt.event_type == "stage_started"
    assert evt.payload == {"stage": "test"}
    assert evt.run_id is None

    events = stream.read_events()
    assert len(events) == 1
    assert events[0].event_type == "stage_started"


def test_event_stream_filter_by_type(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    stream.emit(event_type="stage_started", payload={})
    stream.emit(event_type="stage_completed", payload={})
    stream.emit(event_type="stage_started", payload={})

    started = stream.read_events(event_type="stage_started")
    assert len(started) == 2

    completed = stream.read_events(event_type="stage_completed")
    assert len(completed) == 1


def test_event_stream_filter_by_tag(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    stream.emit(event_type="a", payload={}, tags=["tag1"])
    stream.emit(event_type="b", payload={}, tags=["tag2"])
    stream.emit(event_type="c", payload={}, tags=["tag1", "tag2"])

    tag1 = stream.read_events(tag="tag1")
    assert len(tag1) == 2

    tag2 = stream.read_events(tag="tag2")
    assert len(tag2) == 2


def test_event_stream_limit(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    for i in range(5):
        stream.emit(event_type="test", payload={"i": i})

    events = stream.read_events(limit=2)
    assert len(events) == 2
    assert events[0].payload["i"] == 0
    assert events[1].payload["i"] == 1


def test_event_stream_count(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    stream.emit(event_type="a", payload={}, tags=["x"])
    stream.emit(event_type="a", payload={}, tags=["y"])
    stream.emit(event_type="b", payload={}, tags=["x"])

    assert stream.count() == 3
    assert stream.count(event_type="a") == 2
    assert stream.count(tag="x") == 2


def test_event_stream_read_latest(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import HypothesisLabEventStream

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")
    stream.emit(event_type="test", payload={"v": 1})
    stream.emit(event_type="test", payload={"v": 2})

    latest = stream.read_latest("test")
    assert latest is not None
    assert latest.payload["v"] == 2

    missing = stream.read_latest("other")
    assert missing is None


def test_event_stream_invalid_input(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_events import (
        HypothesisLabEventError,
        HypothesisLabEventStream,
    )

    stream = HypothesisLabEventStream(tmp_path / "events.jsonl")

    with pytest.raises(HypothesisLabEventError):
        stream.emit(event_type="", payload={})

    with pytest.raises(HypothesisLabEventError):
        stream.emit(event_type="test", payload="not_a_dict")


# -- Session --


def test_session_create_and_load(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    assert session.session_id.startswith("hls_")
    assert (session.workspace / "session.json").exists()

    loaded = HypothesisLabSession.load(workspace=session.workspace)
    assert loaded.session_id == session.session_id


def test_session_load_missing(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import (
        HypothesisLabSession,
        HypothesisLabSessionError,
    )

    with pytest.raises(HypothesisLabSessionError):
        HypothesisLabSession.load(workspace=tmp_path / "missing")


def test_session_artifact_lifecycle(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(artifact_type="cards", artifact_id="c1", payload={"x": 1})

    assert session.get_artifact("cards", "c1") == {"x": 1}
    assert session.get_artifact("cards", "missing") is None

    listed = session.list_artifacts("cards")
    assert len(listed) == 1
    assert listed[0]["artifact_id"] == "c1"


def test_session_latest_artifact(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(artifact_type="cards", artifact_id="c1", payload={"x": 1})
    session.put_artifact(artifact_type="cards", artifact_id="c2", payload={"x": 2})

    latest = session.get_latest_artifact("cards")
    assert latest is not None
    assert latest["artifact_id"] in {"c1", "c2"}

    missing = session.get_latest_artifact("missing")
    assert missing is None


def test_session_events(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.emit(event_type="test_event", payload={"k": "v"}, tags=["tag1"])

    events = session.get_events(event_type="test_event")
    assert len(events) == 1
    assert events[0]["event_type"] == "test_event"
    assert events[0]["payload"] == {"k": "v"}
    assert "tag1" in events[0]["tags"]


def test_session_checkpoint(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    assert session.load_checkpoint() is None

    cp = session.save_checkpoint(
        stage="sandbox", stage_data={"param_idx": 3}, completed_steps=["cards", "hypotheses"]
    )
    assert cp.stage == "sandbox"
    assert cp.completed_steps == ["cards", "hypotheses"]

    loaded = session.load_checkpoint()
    assert loaded is not None
    assert loaded.stage == "sandbox"

    session.clear_checkpoint()
    assert session.load_checkpoint() is None


def test_session_state_summary(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.put_artifact(artifact_type="cards", artifact_id="c1", payload={})
    session.put_artifact(artifact_type="hypotheses", artifact_id="h1", payload={})
    session.emit(event_type="test", payload={})

    summary = session.get_state_summary()
    assert summary["session_id"] == session.session_id
    assert summary["total_artifacts"] == 2
    assert summary["artifact_counts"]["cards"] == 1
    assert summary["artifact_counts"]["hypotheses"] == 1
    assert summary["event_count"] >= 1


# -- Query interface --


def _make_sandbox_artifact(session: Any, artifact_id: str, score: float) -> None:
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id=artifact_id,
        payload={
            "best_candidate": {
                "candidate_id": f"cand_{artifact_id}",
                "candidate_family": "family_a",
                "task1_total_score_proxy": score,
                "segment_score": score * 0.5,
            }
        },
    )


def test_query_list_candidates(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import list_candidates
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    _make_sandbox_artifact(session, "sb1", 50.0)
    _make_sandbox_artifact(session, "sb2", 70.0)

    candidates = list_candidates(session)
    assert len(candidates) == 2
    assert candidates[0]["score"] == 70.0
    assert candidates[1]["score"] == 50.0


def test_query_get_best_evidence(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import get_best_evidence
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    _make_sandbox_artifact(session, "sb1", 50.0)
    _make_sandbox_artifact(session, "sb2", 70.0)

    evidence = get_best_evidence(session)
    assert evidence is not None
    assert evidence["score"] == 70.0
    assert evidence["candidate_id"] == "cand_sb2"


def test_query_get_best_evidence_empty(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import get_best_evidence
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    assert get_best_evidence(session) is None


def test_query_recommend_next_action(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import recommend_next_action
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    rec = recommend_next_action(session)
    assert rec["action"] == "build_knowledge_cards"

    session.put_artifact(artifact_type="knowledge_cards", artifact_id="kc1", payload={})
    rec = recommend_next_action(session)
    assert rec["action"] == "generate_hypotheses"

    session.put_artifact(artifact_type="hypotheses", artifact_id="h1", payload={})
    rec = recommend_next_action(session)
    assert rec["action"] == "design_algorithm_blueprints"

    session.put_artifact(artifact_type="blueprints", artifact_id="bp1", payload={})
    rec = recommend_next_action(session)
    assert rec["action"] == "run_blueprint_sandbox"

    session.put_artifact(artifact_type="sandbox", artifact_id="sb1", payload={})
    rec = recommend_next_action(session)
    assert rec["action"] == "record_sandbox_outcome"


def test_query_list_pending_blueprints(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import list_pending_blueprints
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
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

    pending = list_pending_blueprints(session)
    assert len(pending) == 2
    sandboxed = [p for p in pending if p["sandboxed"]]
    not_sandboxed = [p for p in pending if not p["sandboxed"]]
    assert len(sandboxed) == 1
    assert len(not_sandboxed) == 1
    assert not_sandboxed[0]["blueprint_id"] == "bp_b"


def test_query_get_pareto_front(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import get_pareto_front
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    # When runtime is None, higher score dominates lower score
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb1",
        payload={
            "best_candidate": {
                "candidate_id": "c1",
                "task1_total_score_proxy": 80.0,
            }
        },
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb2",
        payload={
            "best_candidate": {
                "candidate_id": "c2",
                "task1_total_score_proxy": 90.0,
            }
        },
    )

    pareto = get_pareto_front(session)
    assert len(pareto) == 1
    assert pareto[0]["score"] == 90.0

    # With explicit runtimes, both can be non-dominated if tradeoff exists
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb3",
        payload={
            "best_candidate": {
                "candidate_id": "c3",
                "task1_total_score_proxy": 85.0,
                "inference_seconds": 1.0,
            }
        },
    )
    session.put_artifact(
        artifact_type="sandbox",
        artifact_id="sb4",
        payload={
            "best_candidate": {
                "candidate_id": "c4",
                "task1_total_score_proxy": 95.0,
                "inference_seconds": 10.0,
            }
        },
    )

    pareto = get_pareto_front(session)
    assert len(pareto) == 2
    scores = {p["score"] for p in pareto}
    assert scores == {85.0, 95.0}


def test_query_summarize_events(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab_queries import summarize_recent_events
    from pde_agent.core.hypothesis_lab_session import HypothesisLabSession

    session = HypothesisLabSession.create(workspace=tmp_path / "ws")
    session.emit(event_type="stage_started", payload={"stage": "a"})
    session.emit(event_type="stage_completed", payload={"stage": "a"})
    session.emit(event_type="stage_started", payload={"stage": "b"})

    summary = summarize_recent_events(session, n=10)
    assert summary["event_count"] == 3
    assert summary["event_counts"]["stage_started"] == 2
    assert summary["event_counts"]["stage_completed"] == 1


# -- CLI commands --


def test_cli_create_and_load_session(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_primitives import (
        agent_hypothesis_create_session_command,
        agent_hypothesis_load_session_command,
    )

    ws = str(tmp_path / "ws")
    result = agent_hypothesis_create_session_command(workspace=ws)
    assert result["state_hint"] == "HYPOTHESIS_LAB_SESSION_CREATED"
    assert "run_id" in result

    loaded = agent_hypothesis_load_session_command(workspace=ws)
    assert loaded["state_hint"] == "HYPOTHESIS_LAB_SESSION_LOADED"
    assert loaded["run_id"] == result["run_id"]


def test_cli_build_cards(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_primitives import (
        agent_hypothesis_build_cards_command,
        agent_hypothesis_create_session_command,
    )

    ws = tmp_path / "ws"
    agent_hypothesis_create_session_command(workspace=str(ws))

    report = tmp_path / "report.json"
    report.write_text(
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

    result = agent_hypothesis_build_cards_command(
        workspace=str(ws),
        manual_report_paths=[str(report)],
    )
    assert result["state_hint"] == "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
    assert result["metrics"]["card_count"] >= 0


def test_cli_query_state(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_primitives import (
        agent_hypothesis_create_session_command,
        agent_hypothesis_query_state_command,
    )

    ws = str(tmp_path / "ws")
    agent_hypothesis_create_session_command(workspace=ws)
    result = agent_hypothesis_query_state_command(workspace=ws)
    assert result["state_hint"] == "HYPOTHESIS_LAB_STATE_QUERIED"
    assert result["metrics"]["artifact_count"] == 0


def test_cli_list_candidates_and_get_best(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_primitives import (
        agent_hypothesis_create_session_command,
        agent_hypothesis_get_best_evidence_command,
        agent_hypothesis_list_candidates_command,
    )

    ws = str(tmp_path / "ws")
    agent_hypothesis_create_session_command(workspace=ws)

    candidates = agent_hypothesis_list_candidates_command(workspace=ws)
    assert candidates["state_hint"] == "HYPOTHESIS_LAB_CANDIDATES_LISTED"
    assert candidates["metrics"]["candidate_count"] == 0

    best = agent_hypothesis_get_best_evidence_command(workspace=ws)
    assert best["state_hint"] == "HYPOTHESIS_LAB_BEST_EVIDENCE_QUERIED"
    assert best["metrics"]["has_evidence"] is False
