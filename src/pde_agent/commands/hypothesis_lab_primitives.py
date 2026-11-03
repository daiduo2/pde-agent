"""CLI commands for Hypothesis Lab agent infrastructure primitives.

Each command is a thin wrapper around a composable primitive that operates
on a HypothesisLabSession. Agents can call these commands individually to
build, inspect, and iterate on scientific experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_primitives import (
    build_knowledge_cards_primitive,
    design_algorithm_blueprints_primitive,
    generate_hypotheses_primitive,
    record_sandbox_outcome_primitive,
    run_blueprint_sandbox_primitive,
    run_hypothesis_lab_on_session,
)
from pde_agent.core.hypothesis_lab_queries import (
    get_best_evidence,
    get_failure_pattern,
    get_pareto_front,
    list_candidates,
    list_pending_blueprints,
    recommend_next_action,
    summarize_recent_events,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.json_output import success_payload


# -- Session management commands --


def agent_hypothesis_create_session_command(
    *,
    workspace: str,
    session_id: str | None = None,
) -> dict[str, object]:
    session = HypothesisLabSession.create(
        workspace=workspace,
        session_id=session_id,
    )
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-create-session",
        state_hint="HYPOTHESIS_LAB_SESSION_CREATED",
        artifacts={
            "workspace": str(session.workspace),
            "session_path": str(session.workspace),
        },
        metrics={},
        next_hint="build-knowledge-cards or load existing inputs",
    )


def agent_hypothesis_load_session_command(
    *,
    workspace: str,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    summary = session.get_state_summary()
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-load-session",
        state_hint="HYPOTHESIS_LAB_SESSION_LOADED",
        artifacts={
            "workspace": str(session.workspace),
            "session_path": str(session.workspace),
        },
        metrics={
            "artifact_count": summary.get("total_artifacts", 0),
            "event_count": summary.get("event_count", 0),
        },
        next_hint=recommend_next_action(session)["action"],
    )


# -- Primitive stage commands --


def agent_hypothesis_build_cards_command(
    *,
    workspace: str,
    manual_report_paths: list[str],
    adaptive_report_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    checkpoint_paths: list[str] | None = None,
    literature_paths: list[str] | None = None,
    algorithm_candidate_paths: list[str] | None = None,
    failure_memory_paths: list[str] | None = None,
    artifact_id: str = "knowledge_cards",
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    result = build_knowledge_cards_primitive(
        session,
        manual_report_paths=manual_report_paths,
        adaptive_report_paths=adaptive_report_paths,
        log_paths=log_paths,
        checkpoint_paths=checkpoint_paths,
        literature_paths=literature_paths,
        algorithm_candidate_paths=algorithm_candidate_paths,
        failure_memory_paths=failure_memory_paths,
        artifact_id=artifact_id,
        force=force,
    )
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-build-cards",
        state_hint="HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT",
        artifacts={
            "workspace": str(session.workspace),
            "artifact_id": artifact_id,
        },
        metrics={"card_count": result.get("card_count", 0)},
        next_hint="generate-hypotheses",
    )


def agent_hypothesis_generate_hypotheses_command(
    *,
    workspace: str,
    cards_artifact_id: str = "knowledge_cards",
    artifact_id: str = "hypotheses",
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    result = generate_hypotheses_primitive(
        session,
        cards_artifact_id=cards_artifact_id,
        artifact_id=artifact_id,
        force=force,
    )
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-generate-hypotheses",
        state_hint="HYPOTHESIS_LAB_HYPOTHESES_GENERATED",
        artifacts={
            "workspace": str(session.workspace),
            "artifact_id": artifact_id,
        },
        metrics={"hypothesis_count": result.get("hypothesis_count", 0)},
        next_hint="design-algorithm-blueprints",
    )


def agent_hypothesis_design_blueprints_command(
    *,
    workspace: str,
    hypotheses_artifact_id: str = "hypotheses",
    artifact_id: str = "blueprints",
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    result = design_algorithm_blueprints_primitive(
        session,
        hypotheses_artifact_id=hypotheses_artifact_id,
        artifact_id=artifact_id,
        force=force,
    )
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-design-blueprints",
        state_hint="HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED",
        artifacts={
            "workspace": str(session.workspace),
            "artifact_id": artifact_id,
        },
        metrics={"blueprint_count": result.get("blueprint_count", 0)},
        next_hint="run-blueprint-sandbox",
    )


def agent_hypothesis_run_sandbox_command(
    *,
    workspace: str,
    context_path: str,
    target_path: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    submission_id: str,
    blueprints_artifact_id: str = "blueprints",
    blueprint_index: int = 0,
    artifact_id: str = "sandbox",
    limit: int | None = None,
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    result = run_blueprint_sandbox_primitive(
        session,
        blueprints_artifact_id=blueprints_artifact_id,
        blueprint_index=blueprint_index,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        artifact_id=artifact_id,
        force=force,
    )
    best = result.get("best_candidate", {})
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-run-sandbox",
        state_hint="HYPOTHESIS_LAB_SANDBOX_VERIFIED",
        artifacts={
            "workspace": str(session.workspace),
            "artifact_id": artifact_id,
            "sandbox_report_path": result["artifacts"]["sandbox_report_path"],
        },
        metrics={
            "best_score": best.get("task1_total_score_proxy"),
            "segment_score": best.get("segment_score"),
            "candidate_budget": result.get("verifier", {}).get("candidate_budget"),
        },
        next_hint="record-sandbox-outcome or run-next-blueprint",
    )


def agent_hypothesis_record_outcome_command(
    *,
    workspace: str,
    cards_artifact_id: str = "knowledge_cards",
    hypotheses_artifact_id: str = "hypotheses",
    blueprints_artifact_id: str = "blueprints",
    sandbox_artifact_id: str = "sandbox",
    failure_memory_output_path: str | None = None,
    artifact_id: str = "outcome",
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    result = record_sandbox_outcome_primitive(
        session,
        cards_artifact_id=cards_artifact_id,
        hypotheses_artifact_id=hypotheses_artifact_id,
        blueprints_artifact_id=blueprints_artifact_id,
        sandbox_artifact_id=sandbox_artifact_id,
        failure_memory_output_path=failure_memory_output_path,
        artifact_id=artifact_id,
        force=force,
    )
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-record-outcome",
        state_hint="HYPOTHESIS_LAB_OUTCOME_RECORDED",
        artifacts={
            "workspace": str(session.workspace),
            "artifact_id": artifact_id,
        },
        metrics={
            "has_candidate_outcome": "candidate_outcome" in result,
            "has_pareto_memory": "pareto_memory" in result,
        },
        next_hint="build-next-policy or run-next-blueprint",
    )


def agent_hypothesis_run_pipeline_command(
    *,
    workspace: str,
    manual_report_paths: list[str],
    context_path: str,
    target_path: str,
    adaptive_report_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    checkpoint_paths: list[str] | None = None,
    literature_paths: list[str] | None = None,
    algorithm_candidate_paths: list[str] | None = None,
    failure_memory_paths: list[str] | None = None,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    submission_id: str = "session_pipeline",
    failure_memory_output_path: str | None = None,
    force: bool = True,
) -> dict[str, object]:
    session = HypothesisLabSession.create(workspace=workspace)
    result = run_hypothesis_lab_on_session(
        session,
        manual_report_paths=manual_report_paths,
        adaptive_report_paths=adaptive_report_paths,
        log_paths=log_paths,
        checkpoint_paths=checkpoint_paths,
        literature_paths=literature_paths,
        algorithm_candidate_paths=algorithm_candidate_paths,
        failure_memory_paths=failure_memory_paths,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        failure_memory_output_path=failure_memory_output_path,
        force=force,
    )
    sandbox = result.get("sandbox", {})
    best = sandbox.get("best_candidate", {})
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-run-pipeline",
        state_hint="HYPOTHESIS_LAB_RAN",
        artifacts={
            "workspace": str(session.workspace),
            "report_artifact_id": "hypothesis_lab_report",
        },
        metrics={
            "cards_count": result.get("cards_count", 0),
            "hypotheses_count": result.get("hypotheses_count", 0),
            "blueprints_count": result.get("blueprints_count", 0),
            "best_score": best.get("task1_total_score_proxy"),
        },
        next_hint="query-state for next action",
    )


# -- Query commands --


def agent_hypothesis_query_state_command(
    *,
    workspace: str,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    summary = session.get_state_summary()
    recommendation = recommend_next_action(session)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-query-state",
        state_hint="HYPOTHESIS_LAB_STATE_QUERIED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "artifact_count": summary.get("total_artifacts", 0),
            "event_count": summary.get("event_count", 0),
        },
        next_hint=recommendation["action"],
    )


def agent_hypothesis_list_candidates_command(
    *,
    workspace: str,
    family: str | None = None,
    status: str | None = None,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    candidates = list_candidates(session, family=family, status=status)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-list-candidates",
        state_hint="HYPOTHESIS_LAB_CANDIDATES_LISTED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "candidate_count": len(candidates),
            "family_filter": family,
            "status_filter": status,
        },
        next_hint="get-best-evidence or run-sandbox for a specific candidate",
    )


def agent_hypothesis_get_best_evidence_command(
    *,
    workspace: str,
    metric: str = "score",
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    evidence = get_best_evidence(session, metric=metric)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-get-best-evidence",
        state_hint="HYPOTHESIS_LAB_BEST_EVIDENCE_QUERIED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "has_evidence": evidence is not None,
            "best_score": evidence.get("score") if evidence else None,
            "best_family": evidence.get("candidate_family") if evidence else None,
        },
        next_hint="run-sandbox or compare with current best",
    )


def agent_hypothesis_get_failure_pattern_command(
    *,
    workspace: str,
    family: str,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    pattern = get_failure_pattern(session, family=family)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-get-failure-pattern",
        state_hint="HYPOTHESIS_LAB_FAILURE_PATTERN_QUERIED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "family": family,
            "failure_count": pattern.get("failure_count", 0),
            "recommendation": pattern.get("recommendation"),
        },
        next_hint="avoid this family or apply recommended fix",
    )


def agent_hypothesis_list_pending_blueprints_command(
    *,
    workspace: str,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    pending = list_pending_blueprints(session)
    not_sandboxed = [p for p in pending if not p.get("sandboxed")]
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-list-pending-blueprints",
        state_hint="HYPOTHESIS_LAB_PENDING_BLUEPRINTS_LISTED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "total_blueprints": len(pending),
            "pending_count": len(not_sandboxed),
            "sandboxed_count": len(pending) - len(not_sandboxed),
        },
        next_hint="run-sandbox for next pending blueprint",
    )


def agent_hypothesis_get_pareto_front_command(
    *,
    workspace: str,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    pareto = get_pareto_front(session)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-get-pareto-front",
        state_hint="HYPOTHESIS_LAB_PARETO_FRONT_QUERIED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "pareto_size": len(pareto),
            "best_score": pareto[0].get("score") if pareto else None,
        },
        next_hint="explore tradeoffs or refine best candidates",
    )


def agent_hypothesis_summarize_events_command(
    *,
    workspace: str,
    n: int = 20,
    event_type: str | None = None,
) -> dict[str, object]:
    session = HypothesisLabSession.load(workspace=workspace)
    summary = summarize_recent_events(session, n=n, event_type=event_type)
    return success_payload(
        run_id=session.session_id,
        command="agent-hypothesis-summarize-events",
        state_hint="HYPOTHESIS_LAB_EVENTS_SUMMARIZED",
        artifacts={
            "workspace": str(session.workspace),
        },
        metrics={
            "event_count": summary.get("event_count", 0),
            "event_type_filter": event_type,
        },
        next_hint="review recent activity or take next action",
    )
