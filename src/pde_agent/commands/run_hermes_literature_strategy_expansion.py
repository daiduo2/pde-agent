from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_literature_expansion import (
    BLOCKED_EXECUTOR_TASKS_STATE_HINT,
    FAILURE_MEMORY_STATE_HINT,
    LOCAL_ASSET_CANDIDATES_STATE_HINT,
    QUEUE_EVALUATION_STATE_HINT,
    QUEUE_STATE_HINT,
    REFINEMENT_STATE_HINT,
    REGISTRY_SUMMARY_STATE_HINT,
    STATE_HINT,
    build_local_asset_algorithm_candidates,
    convert_candidate_promotion_to_strategy_evidence,
    evaluate_expansion_sandbox_queue,
    plan_candidate_evidence_accumulation,
    prepare_expansion_blocked_executor_authoring_tasks,
    refine_algorithm_candidate,
    record_promoted_algorithm_strategy_proposal,
    record_hermes_literature_candidate_outcome,
    run_candidate_expansion_handoff,
    run_expansion_sandbox_queue,
    run_hermes_literature_strategy_expansion,
    summarize_literature_registry_for_hermes,
)
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
from pde_agent.core.json_output import success_payload


def run_hermes_literature_strategy_expansion_command(
    *,
    algorithm_candidates_path: str,
    workspace_dir: str,
    checkpoint_paths: list[str] | None,
    failure_memory_path: str | None,
    registry_summary_path: str | None,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        workspace_dir=Path(workspace_dir),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        registry_summary_path=Path(registry_summary_path)
        if registry_summary_path is not None
        else None,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        first_action = report.get("actions", [{}])[0] if report.get("actions") else {}
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hermes_literature_strategy_expansion",
            payload={
                "algorithm_candidate_id": first_action.get(
                    "algorithm_candidate_id"
                )
                if isinstance(first_action, dict)
                else None,
                "candidate_family": first_action.get("candidate_family")
                if isinstance(first_action, dict)
                else None,
                "outcome": "expanded" if report["action_count"] else "skipped",
                "action_count": report["action_count"],
                "executable_count": report["executable_count"],
                "proposal_count": report["proposal_count"],
                "skipped_count": report["skipped_count"],
            },
            run_id="local",
            source_artifact_path=report["expansion_report_path"],
            tags=["hermes", "literature"],
        )
    return success_payload(
        run_id="local",
        command="run-hermes-literature-strategy-expansion",
        state_hint=STATE_HINT,
        artifacts={
            "expansion_report_path": report["expansion_report_path"],
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            "action_count": report["action_count"],
            "executable_count": report["executable_count"],
            "proposal_count": report["proposal_count"],
            "skipped_count": report["skipped_count"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )


def run_candidate_expansion_handoff_command(
    *,
    candidate_plan_path: str,
    workspace_dir: str,
    local_strategy_library_path: str | None,
    checkpoint_paths: list[str] | None,
    max_strategy_candidates: int | None,
    failure_memory_path: str | None,
    registry_summary_path: str | None,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = run_candidate_expansion_handoff(
        candidate_plan_path=Path(candidate_plan_path),
        workspace_dir=Path(workspace_dir),
        local_strategy_library_path=Path(local_strategy_library_path)
        if local_strategy_library_path is not None
        else None,
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])]
        if checkpoint_paths
        else None,
        max_strategy_candidates=max_strategy_candidates,
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        registry_summary_path=Path(registry_summary_path)
        if registry_summary_path is not None
        else None,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hermes_candidate_expansion_handoff",
            payload={
                "candidate_plan_path": candidate_plan_path,
                "expansion_required": True,
                "algorithm_candidate_count": report["metrics"][
                    "algorithm_candidate_count"
                ],
                "expansion_action_count": report["metrics"][
                    "expansion_action_count"
                ],
                "next_command": report["next_command"],
            },
            run_id="local",
            source_artifact_path=report["handoff_report_path"],
            tags=["hermes", "literature", "handoff"],
        )
    return success_payload(
        run_id="local",
        command="run-candidate-expansion-handoff",
        state_hint="HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN",
        artifacts={
            "handoff_report_path": report["handoff_report_path"],
            "algorithm_candidates_path": report["artifacts"][
                "algorithm_candidates_path"
            ],
            "expansion_report_path": report["artifacts"]["expansion_report_path"],
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            **report["metrics"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )


def build_local_asset_algorithm_candidates_command(
    *,
    local_strategy_library_path: str,
    checkpoint_paths: list[str] | None,
    output_path: str,
    max_strategy_candidates: int,
    force: bool,
) -> dict[str, object]:
    report = build_local_asset_algorithm_candidates(
        local_strategy_library_path=Path(local_strategy_library_path),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        output_path=Path(output_path),
        max_strategy_candidates=max_strategy_candidates,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-local-asset-algorithm-candidates",
        state_hint="ALGORITHM_CANDIDATES_COMPILED",
        artifacts={"algorithm_candidates_path": report["algorithm_candidates_path"]},
        metrics={
            "source_state_hint": LOCAL_ASSET_CANDIDATES_STATE_HINT,
            "algorithm_candidate_count": report["algorithm_candidate_count"],
        },
        next_hint=report["next_command"],
    )


def run_expansion_sandbox_queue_command(
    *,
    expansion_report_path: str,
    context_path: str,
    target_path: str,
    initial_submission_dir: str | None,
    output_dir: str,
    submission_id: str,
    max_items: int | None,
    failure_memory_path: str | None,
    reference_path: str | None,
    reference_sample_limit: int | None,
    timing_profile: str | None,
    timing_factor: float | None,
    selected_algorithm_candidate_id: str | None = None,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = run_expansion_sandbox_queue(
        expansion_report_path=Path(expansion_report_path),
        context_path=Path(context_path),
        target_path=Path(target_path),
        initial_submission_dir=Path(initial_submission_dir)
        if initial_submission_dir is not None
        else None,
        output_dir=Path(output_dir),
        submission_id=submission_id,
        max_items=max_items,
        failure_memory_path=Path(failure_memory_path)
        if failure_memory_path is not None
        else None,
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        selected_algorithm_candidate_id=selected_algorithm_candidate_id,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hermes_expansion_sandbox_queue",
            payload={
                **report["metrics"],
                "expansion_report_path": expansion_report_path,
                "next_command": report["next_command"],
            },
            run_id="local",
            source_artifact_path=report["queue_report_path"],
            tags=["hermes", "literature", "sandbox_queue"],
        )
    return success_payload(
        run_id="local",
        command="run-expansion-sandbox-queue",
        state_hint=QUEUE_STATE_HINT,
        artifacts={
            "queue_report_path": report["queue_report_path"],
            **({"failure_memory_path": report["failure_memory_path"]}
               if report.get("failure_memory_path")
               else {}),
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            **report["metrics"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )


def evaluate_expansion_sandbox_queue_command(
    *,
    queue_report_path: str,
    output_dir: str,
    registry_path: str,
    current_best_score: float,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = evaluate_expansion_sandbox_queue(
        queue_report_path=Path(queue_report_path),
        output_dir=Path(output_dir),
        registry_path=Path(registry_path),
        current_best_score=current_best_score,
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="hermes_expansion_sandbox_queue_evaluated",
            payload={
                **report["metrics"],
                "queue_report_path": queue_report_path,
                "registry_path": report["artifacts"]["registry_path"],
                "next_command": report["next_command"],
            },
            run_id="local",
            source_artifact_path=report["queue_evaluation_report_path"],
            tags=["hermes", "literature", "promotion", "registry"],
        )
    return success_payload(
        run_id="local",
        command="evaluate-expansion-sandbox-queue",
        state_hint=QUEUE_EVALUATION_STATE_HINT,
        artifacts={
            "queue_evaluation_report_path": report[
                "queue_evaluation_report_path"
            ],
            "registry_path": report["artifacts"]["registry_path"],
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            **report["metrics"],
            "trace_event_recorded": trace_event is not None,
        },
        next_hint=report["next_command"],
    )


def prepare_expansion_blocked_executor_authoring_tasks_command(
    *,
    queue_report_path: str,
    output_dir: str,
    context_path: str,
    target_path: str,
    submission_id: str,
    max_tasks: int | None,
    force: bool,
) -> dict[str, object]:
    report = prepare_expansion_blocked_executor_authoring_tasks(
        queue_report_path=Path(queue_report_path),
        output_dir=Path(output_dir),
        context_path=Path(context_path),
        target_path=Path(target_path),
        submission_id=submission_id,
        max_tasks=max_tasks,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-expansion-blocked-executor-authoring-tasks",
        state_hint=BLOCKED_EXECUTOR_TASKS_STATE_HINT,
        artifacts={
            "blocked_executor_tasks_path": report[
                "blocked_executor_tasks_path"
            ],
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def record_hermes_literature_candidate_outcome_command(
    *,
    algorithm_candidate_id: str,
    candidate_key: str,
    outcome: str,
    reason_codes: list[str],
    output_path: str,
    force: bool,
    trace_path: str | None = None,
) -> dict[str, object]:
    report = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id=algorithm_candidate_id,
        candidate_key=candidate_key,
        outcome=outcome,
        reason_codes=reason_codes,
        output_path=Path(output_path),
        force=force,
    )
    trace_event = None
    if trace_path is not None:
        last_outcome = (
            report.get("outcomes", [{}])[-1] if report.get("outcomes") else {}
        )
        trace_event = HypothesisLabTraceStore(trace_path).append_event(
            event_type="literature_candidate_outcome",
            payload={
                "algorithm_candidate_id": algorithm_candidate_id,
                "candidate_key": candidate_key,
                "candidate_family": _family_from_candidate_key(candidate_key),
                "outcome": outcome,
                "reason_codes": reason_codes,
                "recorded_outcome": last_outcome.get("outcome")
                if isinstance(last_outcome, dict)
                else None,
            },
            run_id="local",
            source_artifact_path=output_path,
            tags=["hermes", "literature", "outcome"],
        )
    return success_payload(
        run_id="local",
        command="record-hermes-literature-candidate-outcome",
        state_hint=FAILURE_MEMORY_STATE_HINT,
        artifacts={
            "failure_memory_path": str(output_path),
            **({"trace_path": trace_path, "trace_event_id": trace_event["event_id"]}
               if trace_event is not None
               else {}),
        },
        metrics={
            "failed_candidate_count": len(
                report.get("failed_algorithm_candidate_ids", [])
            ),
            "outcome_count": len(report.get("outcomes", [])),
            "trace_event_recorded": trace_event is not None,
        },
        next_hint="run-hermes-literature-strategy-expansion",
    )


def _family_from_candidate_key(candidate_key: str) -> str | None:
    if not candidate_key:
        return None
    return candidate_key.split(":", 1)[0].split("/", 1)[0]


def summarize_literature_registry_for_hermes_command(
    *,
    registry_path: str,
    output_path: str,
    current_best_score: float,
    force: bool,
) -> dict[str, object]:
    report = summarize_literature_registry_for_hermes(
        registry_path=Path(registry_path),
        output_path=Path(output_path),
        current_best_score=current_best_score,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="summarize-literature-registry-for-hermes",
        state_hint=REGISTRY_SUMMARY_STATE_HINT,
        artifacts={"hermes_registry_summary_path": report["summary_path"]},
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def refine_algorithm_candidate_command(
    *,
    registry_summary_path: str,
    output_path: str,
    algorithm_candidate_id: str | None,
    force: bool,
) -> dict[str, object]:
    report = refine_algorithm_candidate(
        registry_summary_path=Path(registry_summary_path),
        output_path=Path(output_path),
        algorithm_candidate_id=algorithm_candidate_id,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="refine-algorithm-candidate",
        state_hint=REFINEMENT_STATE_HINT,
        artifacts={"candidate_refinement_path": report["candidate_refinement_path"]},
        metrics={
            "algorithm_candidate_id": report["algorithm_candidate_id"],
            "candidate_family": report["candidate_family"],
            "refinement_type": report["refinement_type"],
            "diagnostic_score": report["diagnostic_score"],
            "score_delta_vs_current_best": report[
                "score_delta_vs_current_best"
            ],
        },
        next_hint=report["next_command"],
    )


def record_promoted_algorithm_strategy_proposal_command(
    *,
    candidate_promotion_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = record_promoted_algorithm_strategy_proposal(
        candidate_promotion_path=Path(candidate_promotion_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-promoted-algorithm-strategy-proposal",
        state_hint="STRATEGY_PROPOSAL_RECORDED",
        artifacts={"strategy_proposal_path": report["strategy_proposal_path"]},
        metrics={
            "strategy_id": report["strategy_id"],
            "source_type": report["source_type"],
            "observable_count": len(report["expected_observables"]),
            "risk_count": len(report["known_risks"]),
        },
        next_hint=report["next_command"],
    )


def convert_candidate_promotion_to_strategy_evidence_command(
    *,
    candidate_promotion_path: str,
    extra_candidate_promotion_paths: list[str] | None = None,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    evidence = convert_candidate_promotion_to_strategy_evidence(
        candidate_promotion_path=Path(candidate_promotion_path),
        extra_candidate_promotion_paths=[
            Path(path) for path in (extra_candidate_promotion_paths or [])
        ],
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="convert-candidate-promotion-to-strategy-evidence",
        state_hint="STRATEGY_EVIDENCE_SUMMARY_BUILT",
        artifacts={
            "strategy_evidence_summary_path": evidence["evidence_summary_path"],
            "source_candidate_promotion_path": evidence[
                "source_candidate_promotion_path"
            ],
        },
        metrics={
            "evaluation_window_count": evidence["evaluation_window_count"],
            "mean_score_delta": evidence["mean_score_delta"],
            "long_window_score_delta": evidence["long_window_score_delta"],
            "failure_rate": evidence["failure_rate"],
        },
        next_hint=evidence["next_command"],
    )


def plan_candidate_evidence_accumulation_command(
    *,
    strategy_promotion_report_path: str,
    strategy_evidence_summary_path: str | None,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    plan = plan_candidate_evidence_accumulation(
        strategy_promotion_report_path=Path(strategy_promotion_report_path),
        strategy_evidence_summary_path=Path(strategy_evidence_summary_path)
        if strategy_evidence_summary_path is not None
        else None,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-candidate-evidence-accumulation",
        state_hint="CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED",
        artifacts={"evidence_accumulation_plan_path": plan["evidence_accumulation_plan_path"]},
        metrics={
            "decision": plan["decision"],
            "current_window_count": plan["current_window_count"],
            "required_window_count": plan["required_window_count"],
            "missing_window_count": plan["missing_window_count"],
        },
        next_hint=plan["next_command"],
    )
