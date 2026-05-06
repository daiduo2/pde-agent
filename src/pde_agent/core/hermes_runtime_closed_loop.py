"""Closed-loop runtime bridge from Hypothesis Lab trace to Hermes heartbeat."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat
from pde_agent.core.hermes_local_judge import judge_hermes_submission
from pde_agent.core.hermes_literature_expansion import (
    convert_candidate_promotion_to_strategy_evidence,
    evaluate_expansion_sandbox_queue,
    plan_candidate_evidence_accumulation,
    prepare_expansion_blocked_executor_authoring_tasks,
    refine_algorithm_candidate,
    record_promoted_algorithm_strategy_proposal,
    run_candidate_expansion_handoff,
    run_expansion_sandbox_queue,
    summarize_literature_registry_for_hermes,
)
from pde_agent.core.strategy_growth import (
    build_strategy_rubric,
    evaluate_strategy_promotion,
)
from pde_agent.core.literature_algorithm_lab import (
    audit_bounded_executor_authoring_task,
)
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
from pde_agent.core.hermes_repair_policy_summary import build_repair_policy_summary
from pde_agent.core.hermes_research_directive import (
    apply_research_directive_overrides,
    load_hermes_research_directive,
)
from pde_agent.core.hypothesis_lab_trace import (
    HypothesisLabTraceStore,
    summarize_hypothesis_lab_trace,
)
from pde_agent.core.research_chronicle import build_research_chronicle
from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog


STATE_HINT = "HERMES_RUNTIME_CLOSED_LOOP_RAN"
HeartbeatRunner = Callable[..., dict[str, Any]]
ExpansionHandoffRunner = Callable[..., dict[str, Any]]
SandboxQueueRunner = Callable[..., dict[str, Any]]
QueueEvaluationRunner = Callable[..., dict[str, Any]]
RegistrySummaryRunner = Callable[..., dict[str, Any]]
RefinementRunner = Callable[..., dict[str, Any]]
StrategyProposalRunner = Callable[..., dict[str, Any]]
StrategyEvidenceRunner = Callable[..., dict[str, Any]]
StrategyRubricRunner = Callable[..., dict[str, Any]]
StrategyPromotionRunner = Callable[..., dict[str, Any]]
EvidenceAccumulationRunner = Callable[..., dict[str, Any]]
BlockedExecutorAuthoringRunner = Callable[..., dict[str, Any]]
AuthoringAuditRunner = Callable[..., dict[str, Any]]


class HermesRuntimeClosedLoopError(RuntimeError):
    pass


def run_hermes_runtime_closed_loop(
    *,
    output_dir: str | Path,
    trace_path: str | Path,
    initial_submission_dir: str | Path,
    candidate_generator: str,
    candidate_generator_fallback: str = "auto",
    max_batches: int = 1,
    recent_window: int = 50,
    repeated_family_threshold: int = 3,
    failed_family_threshold: int = 2,
    min_score_delta: float = 0.1,
    max_rounds_per_batch: int = 1,
    batch_deadline_seconds: float = 300.0,
    submit_score_threshold: float = 120.0,
    learning_objective: str = "score",
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task1_reference_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    hypothesis_candidate_options: dict[str, Any] | None = None,
    research_directive_path: str | Path | None = None,
    evidence_accumulation_plan_path: str | Path | None = None,
    expansion_registry_summary_path: str | Path | None = None,
    expansion_handoff_report_path: str | Path | None = None,
    audit_repair_tasks_path: str | Path | None = None,
    input_research_chronicle_path: str | Path | None = None,
    heartbeat_runner: HeartbeatRunner | None = None,
    expansion_handoff_runner: ExpansionHandoffRunner | None = None,
    sandbox_queue_runner: SandboxQueueRunner | None = None,
    queue_evaluation_runner: QueueEvaluationRunner | None = None,
    registry_summary_runner: RegistrySummaryRunner | None = None,
    refinement_runner: RefinementRunner | None = None,
    strategy_proposal_runner: StrategyProposalRunner | None = None,
    strategy_evidence_runner: StrategyEvidenceRunner | None = None,
    strategy_rubric_runner: StrategyRubricRunner | None = None,
    strategy_promotion_runner: StrategyPromotionRunner | None = None,
    evidence_accumulation_runner: EvidenceAccumulationRunner | None = None,
    blocked_executor_authoring_runner: BlockedExecutorAuthoringRunner | None = None,
    authoring_audit_runner: AuthoringAuditRunner | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "hermes_runtime_closed_loop_report.json"
    if report_path.exists() and not force:
        raise HermesRuntimeClosedLoopError(f"runtime loop report exists: {report_path}")
    root.mkdir(parents=True, exist_ok=True)
    trace_file = Path(trace_path)
    trace_bootstrapped = False
    if not trace_file.exists():
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text("", encoding="utf-8")
        trace_bootstrapped = True
    _preflight_task_pair(
        task_id=1,
        test_path=Path(task1_test_path) if task1_test_path is not None else None,
        target_path=Path(task1_target_path) if task1_target_path is not None else None,
    )
    _preflight_task1_reference(
        reference_path=Path(task1_reference_path)
        if task1_reference_path is not None
        else None,
    )
    _preflight_task_pair(
        task_id=2,
        test_path=Path(task2_test_path) if task2_test_path is not None else None,
        target_path=Path(task2_target_path) if task2_target_path is not None else None,
    )
    evidence_accumulation_plan = _load_evidence_accumulation_plan(
        Path(evidence_accumulation_plan_path)
        if evidence_accumulation_plan_path is not None
        else None
    )
    expansion_registry_summary = _load_expansion_registry_summary(
        Path(expansion_registry_summary_path)
        if expansion_registry_summary_path is not None
        else None
    )
    expansion_handoff_report = _load_expansion_handoff_report(
        Path(expansion_handoff_report_path)
        if expansion_handoff_report_path is not None
        else None
    )
    audit_repair_tasks = _load_audit_repair_tasks(
        Path(audit_repair_tasks_path) if audit_repair_tasks_path is not None else None
    )
    _preflight_audit_repair_inputs(
        audit_repair_tasks=audit_repair_tasks,
        expansion_registry_summary=expansion_registry_summary,
        expansion_handoff_report=expansion_handoff_report,
    )
    research_directive = load_hermes_research_directive(research_directive_path)
    directive_overrides = apply_research_directive_overrides(
        directive_report=research_directive,
        candidate_generator=candidate_generator,
        candidate_generator_fallback=candidate_generator_fallback,
        learning_objective=learning_objective,
        hypothesis_candidate_options=hypothesis_candidate_options,
    )
    effective_candidate_generator = str(directive_overrides["candidate_generator"])
    effective_candidate_generator_fallback = str(
        directive_overrides["candidate_generator_fallback"]
    )
    effective_learning_objective = str(directive_overrides["learning_objective"])
    effective_hypothesis_candidate_options = directive_overrides[
        "hypothesis_candidate_options"
    ]
    effective_candidate_generator_guard_overrides = [
        item
        for item in directive_overrides.get(
            "candidate_generator_guard_overrides",
            [],
        )
        if isinstance(item, str) and item
    ]

    tool_catalog = build_scientific_tool_catalog(
        output_path=root / "scientific_tool_catalog.json",
        force=True,
    )
    trace_summary = summarize_hypothesis_lab_trace(
        trace_path=trace_file,
        output_path=root / "trace_summary.json",
        force=True,
    )
    kpi = build_hypothesis_kpi_report(
        trace_path=trace_file,
        output_path=root / "hypothesis_kpi_report.json",
        recent_window=recent_window,
        force=True,
    )
    repair_policy = build_repair_policy_summary(
        trace_path=trace_file,
        output_path=root / "repair_policy_summary.json",
        recent_window=recent_window,
        force=True,
    )
    doom_loop_path = root / "doom_loop_report.json"
    doom = check_hypothesis_doom_loop(
        trace_path=trace_file,
        output_path=doom_loop_path,
        recent_window=recent_window,
        repeated_family_threshold=repeated_family_threshold,
        failed_family_threshold=failed_family_threshold,
        min_score_delta=min_score_delta,
        force=True,
    )
    candidate_plan = build_hypothesis_candidate_plan(
        output_path=root / "candidate_plan.json",
        candidate_generator=effective_candidate_generator,
        doom_loop_report_path=doom_loop_path,
        hypothesis_kpi_report_path=root / "hypothesis_kpi_report.json",
        repair_policy_summary_path=root / "repair_policy_summary.json",
        research_chronicle_path=Path(input_research_chronicle_path)
        if input_research_chronicle_path is not None
        else None,
        expansion_handoff_report_path=Path(expansion_handoff_report_path)
        if expansion_handoff_report_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        preferred_error_modes=_preferred_error_modes(effective_learning_objective),
        guard_override_candidate_generators=(
            effective_candidate_generator_guard_overrides
        ),
        max_schedule_length=max_batches,
        min_viable_candidate_count=2,
        force=True,
    )
    candidate_generator_schedule = (
        None
        if effective_candidate_generator_fallback == "none"
        else candidate_plan["candidate_generator_schedule"]
    )
    if _candidate_schedule_empty(
        candidate_generator_schedule=candidate_generator_schedule,
        candidate_plan=candidate_plan,
    ):
        heartbeat = _write_empty_candidate_schedule_heartbeat(
            output_dir=root / "heartbeat",
            initial_submission_dir=Path(initial_submission_dir),
            candidate_generator=effective_candidate_generator,
            candidate_generator_fallback=effective_candidate_generator_fallback,
            candidate_generator_schedule=candidate_generator_schedule or [],
            learning_objective=effective_learning_objective,
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            submit_score_threshold=submit_score_threshold,
            doom_loop_report_path=doom_loop_path,
            hypothesis_kpi_report_path=root / "hypothesis_kpi_report.json",
            repair_policy_summary_path=root / "repair_policy_summary.json",
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            task1_test_path=Path(task1_test_path)
            if task1_test_path is not None
            else None,
            task1_target_path=Path(task1_target_path)
            if task1_target_path is not None
            else None,
            task2_test_path=Path(task2_test_path)
            if task2_test_path is not None
            else None,
            task2_target_path=Path(task2_target_path)
            if task2_target_path is not None
            else None,
            force=True,
        )
    else:
        run_heartbeat = heartbeat_runner or run_hermes_heartbeat
        heartbeat = run_heartbeat(
            output_dir=root / "heartbeat",
            initial_submission_dir=Path(initial_submission_dir),
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            submit_score_threshold=submit_score_threshold,
            learning_objective=effective_learning_objective,
            task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
            task1_target_path=Path(task1_target_path)
            if task1_target_path is not None
            else None,
            task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
            task2_target_path=Path(task2_target_path)
            if task2_target_path is not None
            else None,
            candidate_generator=effective_candidate_generator,
            candidate_generator_schedule=candidate_generator_schedule,
            candidate_generator_fallback=effective_candidate_generator_fallback,
            candidate_generator_guard_overrides=(
                effective_candidate_generator_guard_overrides
            ),
            doom_loop_report_path=doom_loop_path,
            hypothesis_kpi_report_path=root / "hypothesis_kpi_report.json",
            repair_policy_summary_path=root / "repair_policy_summary.json",
            hypothesis_reference_path=Path(task1_reference_path)
            if task1_reference_path is not None
            else None,
            hypothesis_candidate_options=effective_hypothesis_candidate_options
            if isinstance(effective_hypothesis_candidate_options, dict)
            else None,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=True,
        )
    cluster_saturation_signals = candidate_plan.get("cluster_saturation_signals", [])
    if not isinstance(cluster_saturation_signals, list):
        cluster_saturation_signals = []
    saturated_candidate_families = candidate_plan.get(
        "learning_constraints", {}
    ).get("saturated_candidate_families", [])
    if not isinstance(saturated_candidate_families, list):
        saturated_candidate_families = []
    persistent_search_guidance = heartbeat.get("persistent_search_guidance", [])
    if not isinstance(persistent_search_guidance, list):
        persistent_search_guidance = []
    candidate_plan_repair_tasks = candidate_plan.get("audit_repair_tasks", [])
    if not isinstance(candidate_plan_repair_tasks, list):
        candidate_plan_repair_tasks = []
    learning_constraints = candidate_plan.get("learning_constraints", {})
    if not isinstance(learning_constraints, dict):
        learning_constraints = {}
    repair_nonconvergence_errors = _string_list(
        learning_constraints.get("repair_nonconvergence_errors")
    )
    repair_convergence_errors = _string_list(
        learning_constraints.get("repair_convergence_errors")
    )
    repair_policy_recommended_actions = [
        dict(item)
        for item in learning_constraints.get("repair_policy_recommended_actions", [])
        if isinstance(item, dict)
    ]
    audit_repair_converged_error_counts = (
        learning_constraints.get("audit_repair_converged_error_counts")
    )
    if not isinstance(audit_repair_converged_error_counts, dict):
        audit_repair_converged_error_counts = {}
    audit_repair_policies = candidate_plan.get("audit_repair_policies")
    if not isinstance(audit_repair_policies, list):
        audit_repair_policies = learning_constraints.get("audit_repair_policies", [])
    if not isinstance(audit_repair_policies, list):
        audit_repair_policies = []
    trace_store = HypothesisLabTraceStore(trace_file)
    recorded_candidate_attempts = _record_heartbeat_candidate_attempts(
        trace_store=trace_store,
        heartbeat=heartbeat,
    )
    trace_event = trace_store.append_event(
        event_type="hermes_runtime_closed_loop",
        payload={
            "algorithm_candidate_id": heartbeat.get("best_candidate", {}).get(
                "candidate_id"
            )
            if isinstance(heartbeat.get("best_candidate"), dict)
            else None,
            "candidate_family": heartbeat.get("best_candidate", {}).get("family")
            if isinstance(heartbeat.get("best_candidate"), dict)
            else None,
            "outcome": heartbeat.get("decision"),
            "best_score": heartbeat.get("best_score"),
            "score_delta": None,
            "guard_active": doom["guard_active"],
            "blocked_candidate_families": doom["blocked_candidate_families"],
            "learning_constraints": learning_constraints,
            "cluster_saturation_signals": cluster_saturation_signals,
            "saturated_candidate_families": saturated_candidate_families,
            "persistent_search_guidance": persistent_search_guidance,
            "audit_repair_tasks": candidate_plan_repair_tasks,
            "audit_repair_policies": audit_repair_policies,
            "repair_nonconvergence_errors": repair_nonconvergence_errors,
            "repair_convergence_errors": repair_convergence_errors,
            "audit_repair_converged_error_counts": audit_repair_converged_error_counts,
            "repair_policy_recommended_actions": repair_policy_recommended_actions,
        },
        run_id="local",
        source_artifact_path=heartbeat.get("heartbeat_report_path"),
        tags=["hermes", "runtime_closed_loop"],
    )
    artifacts = {
        "runtime_loop_report_path": str(report_path),
        "trace_path": str(trace_file),
        "tool_catalog_path": str(root / "scientific_tool_catalog.json"),
        "trace_summary_path": str(root / "trace_summary.json"),
        "hypothesis_kpi_report_path": str(root / "hypothesis_kpi_report.json"),
        "repair_policy_summary_path": str(root / "repair_policy_summary.json"),
        "doom_loop_report_path": str(root / "doom_loop_report.json"),
        "candidate_plan_path": str(root / "candidate_plan.json"),
        "heartbeat_report_path": heartbeat["heartbeat_report_path"],
        "trace_event_id": trace_event["event_id"],
    }
    smoke_report_path = _discover_learning_constraint_smoke_report(root)
    if smoke_report_path is not None:
        artifacts["learning_constraint_smoke_report_path"] = str(smoke_report_path)
    if evidence_accumulation_plan_path is not None:
        artifacts["input_evidence_accumulation_plan_path"] = str(
            evidence_accumulation_plan_path
        )
    if expansion_registry_summary_path is not None:
        artifacts["input_expansion_registry_summary_path"] = str(
            expansion_registry_summary_path
        )
    if expansion_handoff_report_path is not None:
        artifacts["input_expansion_handoff_report_path"] = str(
            expansion_handoff_report_path
        )
    if audit_repair_tasks_path is not None:
        artifacts["input_audit_repair_tasks_path"] = str(audit_repair_tasks_path)
    if input_research_chronicle_path is not None:
        artifacts["input_research_chronicle_path"] = str(input_research_chronicle_path)
    if research_directive_path is not None:
        artifacts["research_directive_path"] = str(research_directive_path)
    metrics = {
        "tool_count": tool_catalog["tool_count"],
        "trace_bootstrapped": trace_bootstrapped,
        "evidence_accumulation_plan_loaded": evidence_accumulation_plan is not None,
        "expansion_registry_summary_loaded": expansion_registry_summary is not None,
        "expansion_handoff_report_loaded": expansion_handoff_report is not None,
        "audit_repair_tasks_loaded": audit_repair_tasks is not None,
        "trace_event_count": trace_summary["event_count"],
        "guard_active": doom["guard_active"],
        "blocked_candidate_families": doom["blocked_candidate_families"],
        "hypothesis_candidate_count": kpi["metrics"]["candidate_count"],
        "hypothesis_promotion_rate": kpi["metrics"]["promotion_rate"],
        "candidate_plan_length": len(candidate_plan["candidate_generator_schedule"]),
        "candidate_space_exhausted": candidate_plan["expansion_required"],
        "cluster_saturation_signal_count": len(cluster_saturation_signals),
        "saturated_candidate_families": saturated_candidate_families,
        "persistent_search_guidance_count": len(persistent_search_guidance),
        "audit_repair_task_count": len(candidate_plan_repair_tasks),
        "audit_repair_policy_count": len(audit_repair_policies),
        "repair_nonconvergence_error_count": len(repair_nonconvergence_errors),
        "repair_nonconvergence_errors": repair_nonconvergence_errors,
        "repair_convergence_error_count": len(repair_convergence_errors),
        "repair_convergence_errors": repair_convergence_errors,
        "repair_policy_recommended_action_count": len(
            repair_policy_recommended_actions
        ),
        "repair_policy_converged_repair_error_count": repair_policy["metrics"][
            "converged_repair_error_count"
        ],
        "repair_policy_nonconverged_repair_error_count": repair_policy["metrics"][
            "nonconverged_repair_error_count"
        ],
        "heartbeat_best_score": heartbeat.get("best_score"),
        "heartbeat_decision": heartbeat.get("decision"),
        "recorded_candidate_attempt_count": recorded_candidate_attempts,
        "research_directive_loaded": research_directive is not None,
        "research_directive_guard_override_count": len(
            effective_candidate_generator_guard_overrides
        ),
    }
    expansion_pipeline = _maybe_run_expansion_pipeline(
        root=root,
        candidate_plan=candidate_plan,
        trace_file=trace_file,
        heartbeat=heartbeat,
        initial_submission_dir=Path(initial_submission_dir),
        candidate_generator=effective_candidate_generator,
        candidate_generator_fallback=effective_candidate_generator_fallback,
        max_batches=max_batches,
        batch_deadline_seconds=batch_deadline_seconds,
        max_rounds_per_batch=max_rounds_per_batch,
        submit_score_threshold=submit_score_threshold,
        learning_objective=effective_learning_objective,
        doom_loop_report_path=doom_loop_path,
        hypothesis_kpi_report_path=root / "hypothesis_kpi_report.json",
        repair_policy_summary_path=root / "repair_policy_summary.json",
        input_research_chronicle_path=Path(input_research_chronicle_path)
        if input_research_chronicle_path is not None
        else None,
        task1_test_path=Path(task1_test_path) if task1_test_path is not None else None,
        task1_target_path=Path(task1_target_path)
        if task1_target_path is not None
        else None,
        task1_reference_path=Path(task1_reference_path)
        if task1_reference_path is not None
        else None,
        task2_test_path=Path(task2_test_path) if task2_test_path is not None else None,
        task2_target_path=Path(task2_target_path)
        if task2_target_path is not None
        else None,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        heartbeat_runner=heartbeat_runner or run_hermes_heartbeat,
        expansion_handoff_runner=expansion_handoff_runner
        or run_candidate_expansion_handoff,
        sandbox_queue_runner=sandbox_queue_runner or run_expansion_sandbox_queue,
        queue_evaluation_runner=queue_evaluation_runner
        or evaluate_expansion_sandbox_queue,
        registry_summary_runner=registry_summary_runner
        or summarize_literature_registry_for_hermes,
        refinement_runner=refinement_runner or refine_algorithm_candidate,
        strategy_proposal_runner=strategy_proposal_runner
        or record_promoted_algorithm_strategy_proposal,
        strategy_evidence_runner=strategy_evidence_runner
        or convert_candidate_promotion_to_strategy_evidence,
        strategy_rubric_runner=strategy_rubric_runner or build_strategy_rubric,
        strategy_promotion_runner=strategy_promotion_runner
        or evaluate_strategy_promotion,
        evidence_accumulation_runner=evidence_accumulation_runner
        or plan_candidate_evidence_accumulation,
        evidence_accumulation_plan=evidence_accumulation_plan,
        expansion_registry_summary_path=Path(expansion_registry_summary_path)
        if expansion_registry_summary_path is not None
        else None,
        blocked_executor_authoring_runner=blocked_executor_authoring_runner
        or prepare_expansion_blocked_executor_authoring_tasks,
        authoring_audit_runner=authoring_audit_runner
        or audit_bounded_executor_authoring_task,
        force=True,
    )
    if expansion_pipeline is not None:
        artifacts.update(expansion_pipeline["artifacts"])
        metrics.update(expansion_pipeline["metrics"])
    audit_command = _audit_command(
        report_path=report_path,
        output_path=root / "hermes_runtime_closed_loop_audit.json",
        require_learning_constraint_smoke=(
            "learning_constraint_smoke_report_path" in artifacts
        ),
    )
    next_command = (
        str(expansion_pipeline["next_command"])
        if expansion_pipeline is not None
        else (
            candidate_plan["next_command"]
            if candidate_plan["expansion_required"]
            else audit_command["command"]
        )
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "artifacts": artifacts,
        "metrics": metrics,
        "guard": {
            "guard_active": doom["guard_active"],
            "blocked_candidate_families": doom["blocked_candidate_families"],
            "decision": doom["decision"],
        },
        "candidate_plan": {
            "candidate_generator_schedule": candidate_plan[
                "candidate_generator_schedule"
            ],
            "applied_candidate_generator_schedule": candidate_generator_schedule,
            "eligible_candidate_count": candidate_plan["eligible_candidate_count"],
            "decision": candidate_plan["decision"],
            "expansion_required": candidate_plan["expansion_required"],
            "learning_constraints": learning_constraints,
            "prior_expansion_exhaustion": candidate_plan.get(
                "prior_expansion_exhaustion",
                {"exhausted": False},
            ),
            "cluster_saturation_signals": cluster_saturation_signals,
            "directive_guard_overrides": candidate_plan.get(
                "directive_guard_overrides",
                [],
            ),
            "audit_repair_tasks": candidate_plan_repair_tasks,
            "audit_repair_policies": audit_repair_policies,
            "repair_nonconvergence_errors": repair_nonconvergence_errors,
            "repair_convergence_errors": repair_convergence_errors,
            "audit_repair_converged_error_counts": audit_repair_converged_error_counts,
            "repair_policy_recommended_actions": repair_policy_recommended_actions,
            "expansion_handoff": candidate_plan.get("expansion_handoff", {}),
            "next_command": candidate_plan["next_command"],
        },
        "heartbeat": {
            "best_candidate": heartbeat.get("best_candidate"),
            "best_score": heartbeat.get("best_score"),
            "decision": heartbeat.get("decision"),
            "stop_reason": heartbeat.get("stop_reason"),
        },
        "research_directive": research_directive or {},
        "expansion_pipeline": expansion_pipeline or {},
        "audit_command": audit_command,
        "next_command": next_command,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    chronicle = _build_runtime_research_chronicle(
        root=root,
        report_path=report_path,
        artifacts=artifacts,
        heartbeat=heartbeat,
        initial_submission_dir=Path(initial_submission_dir),
    )
    report["artifacts"]["research_chronicle_path"] = chronicle["artifacts"][
        "chronicle_path"
    ]
    report["artifacts"]["research_chronicle_updated_log_paths"] = chronicle[
        "artifacts"
    ]["updated_log_paths"]
    report["metrics"]["research_chronicle_built"] = True
    report["metrics"]["research_chronicle_updated_log_count"] = chronicle[
        "metrics"
    ]["updated_log_count"]
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _audit_command(
    *,
    report_path: Path,
    output_path: Path,
    require_learning_constraint_smoke: bool,
) -> dict[str, Any]:
    return {
        "command": "audit-hermes-runtime-closed-loop",
        "runtime_report_path": str(report_path),
        "output_path": str(output_path),
        "require_learning_constraint_smoke": require_learning_constraint_smoke,
    }


def _build_runtime_research_chronicle(
    *,
    root: Path,
    report_path: Path,
    artifacts: dict[str, Any],
    heartbeat: dict[str, Any],
    initial_submission_dir: Path,
) -> dict[str, Any]:
    artifact_paths = _runtime_chronicle_artifact_paths(
        report_path=report_path,
        artifacts=artifacts,
    )
    return build_research_chronicle(
        artifact_paths=artifact_paths,
        output_path=root / "research_chronicle.md",
        submission_dir=_heartbeat_best_submission_dir(
            heartbeat=heartbeat,
            fallback=initial_submission_dir,
        ),
        task_ids=[1, 2],
        force=True,
    )


def _runtime_chronicle_artifact_paths(
    *,
    report_path: Path,
    artifacts: dict[str, Any],
) -> list[Path]:
    paths = [report_path]
    for key in ("heartbeat_report_path", "post_expansion_heartbeat_report_path"):
        value = artifacts.get(key)
        if value is None:
            continue
        path = Path(str(value))
        if path.exists() and path not in paths:
            paths.append(path)
    return paths


def _candidate_schedule_empty(
    *,
    candidate_generator_schedule: list[str] | None,
    candidate_plan: dict[str, Any],
) -> bool:
    return (
        candidate_plan.get("expansion_required") is True
        and isinstance(candidate_generator_schedule, list)
        and not candidate_generator_schedule
    )


def _write_empty_candidate_schedule_heartbeat(
    *,
    output_dir: Path,
    initial_submission_dir: Path,
    candidate_generator: str,
    candidate_generator_fallback: str,
    candidate_generator_schedule: list[str],
    learning_objective: str,
    max_batches: int,
    batch_deadline_seconds: float,
    max_rounds_per_batch: int,
    submit_score_threshold: float,
    doom_loop_report_path: Path,
    hypothesis_kpi_report_path: Path,
    repair_policy_summary_path: Path,
    timing_profile: str | None,
    timing_factor: float | None,
    task1_test_path: Path | None,
    task1_target_path: Path | None,
    task2_test_path: Path | None,
    task2_target_path: Path | None,
    force: bool,
) -> dict[str, Any]:
    report_path = output_dir / "hermes_heartbeat_report.json"
    if report_path.exists() and not force:
        raise HermesRuntimeClosedLoopError(
            f"heartbeat report already exists: {report_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_score = _score_initial_submission(
        submission_dir=initial_submission_dir,
        output_path=output_dir / "initial_submission_judge.json",
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        task2_test_path=task2_test_path,
        task2_target_path=task2_target_path,
    )
    best_score = initial_score["score"]
    report = {
        "schema_version": 1,
        "state_hint": "HERMES_HEARTBEAT_RAN",
        "heartbeat_report_path": str(report_path),
        "parameters": {
            "max_batches": max_batches,
            "batch_deadline_seconds": batch_deadline_seconds,
            "max_rounds_per_batch": max_rounds_per_batch,
            "submit_score_threshold": submit_score_threshold,
            "learning_objective": learning_objective,
            "learning_objective_schedule": [],
            "candidate_generator": candidate_generator,
            "candidate_generator_schedule": candidate_generator_schedule,
            "doom_loop_report_path": str(doom_loop_report_path),
            "candidate_generator_fallback": candidate_generator_fallback,
            "hypothesis_kpi_report_path": str(hypothesis_kpi_report_path),
            "repair_policy_summary_path": str(repair_policy_summary_path),
            "repair_policy_recommended_action_count": 0,
            "timing_profile": timing_profile,
            "timing_factor": timing_factor,
        },
        "batch_count": 0,
        "batches": [],
        "best_candidate": {
            "candidate_id": "initial",
            "family": candidate_generator,
            "score": best_score,
            "submission_dir": str(initial_submission_dir),
            "objective": {
                "selected_objective": learning_objective,
                "objective_score": best_score,
            },
        },
        "best_score": best_score,
        "best_objective": learning_objective,
        "best_objective_score": best_score,
        "stop_reason": "candidate_schedule_empty",
        "decision": "candidate_space_exhausted",
        "next_command": "run-hermes-literature-strategy-expansion",
        "artifacts": initial_score["artifacts"],
        "repair_policy_recommended_actions": [],
    }
    doom = _read_optional_json(doom_loop_report_path)
    if doom:
        report["guard"] = {
            "state_hint": doom.get("state_hint"),
            "guard_active": bool(doom.get("guard_active")),
            "blocked_candidate_families": [
                item
                for item in doom.get("blocked_candidate_families", [])
                if isinstance(item, str) and item
            ],
            "corrective_prompt": doom.get("corrective_prompt", ""),
        }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _score_initial_submission(
    *,
    submission_dir: Path,
    output_path: Path,
    task1_test_path: Path | None,
    task1_target_path: Path | None,
    task2_test_path: Path | None,
    task2_target_path: Path | None,
) -> dict[str, Any]:
    if (
        task1_test_path is None
        and task1_target_path is None
        and task2_test_path is None
        and task2_target_path is None
    ):
        return {"score": 0.0, "artifacts": {}}
    judge = judge_hermes_submission(
        submission_dir=submission_dir,
        output_path=output_path,
        task1_test_path=task1_test_path,
        task1_target_path=task1_target_path,
        task2_test_path=task2_test_path,
        task2_target_path=task2_target_path,
    )
    return {
        "score": float(judge["total_score_proxy"]),
        "artifacts": {"initial_judge_report_path": str(output_path)},
    }


def _maybe_run_expansion_pipeline(
    *,
    root: Path,
    candidate_plan: dict[str, Any],
    trace_file: Path,
    heartbeat: dict[str, Any],
    initial_submission_dir: Path,
    candidate_generator: str,
    candidate_generator_fallback: str,
    max_batches: int,
    batch_deadline_seconds: float,
    max_rounds_per_batch: int,
    submit_score_threshold: float,
    learning_objective: str,
    doom_loop_report_path: Path,
    hypothesis_kpi_report_path: Path,
    repair_policy_summary_path: Path,
    input_research_chronicle_path: Path | None,
    task1_test_path: Path | None,
    task1_target_path: Path | None,
    task1_reference_path: Path | None,
    task2_test_path: Path | None,
    task2_target_path: Path | None,
    timing_profile: str | None,
    timing_factor: float | None,
    heartbeat_runner: HeartbeatRunner,
    expansion_handoff_runner: ExpansionHandoffRunner,
    sandbox_queue_runner: SandboxQueueRunner,
    queue_evaluation_runner: QueueEvaluationRunner,
    registry_summary_runner: RegistrySummaryRunner,
    refinement_runner: RefinementRunner,
    strategy_proposal_runner: StrategyProposalRunner,
    strategy_evidence_runner: StrategyEvidenceRunner,
    strategy_rubric_runner: StrategyRubricRunner,
    strategy_promotion_runner: StrategyPromotionRunner,
    evidence_accumulation_runner: EvidenceAccumulationRunner,
    evidence_accumulation_plan: dict[str, Any] | None,
    expansion_registry_summary_path: Path | None,
    blocked_executor_authoring_runner: BlockedExecutorAuthoringRunner,
    authoring_audit_runner: AuthoringAuditRunner,
    force: bool,
) -> dict[str, Any] | None:
    if candidate_plan.get("expansion_required") is not True:
        return None
    if task1_test_path is None or task1_target_path is None:
        return None
    workspace = root / "expansion_handoff"
    current_best_score = _heartbeat_best_score(heartbeat)
    handoff = expansion_handoff_runner(
        candidate_plan_path=root / "candidate_plan.json",
        workspace_dir=workspace,
        local_strategy_library_path=None,
        checkpoint_paths=None,
        max_strategy_candidates=None,
        failure_memory_path=_existing_failure_memory_path(workspace),
        registry_summary_path=expansion_registry_summary_path,
        force=force,
    )
    if _handoff_reports_expansion_exhaustion(handoff):
        return _run_post_expansion_replan_heartbeat(
            root=root,
            workspace=workspace,
            handoff=handoff,
            trace_file=trace_file,
            initial_submission_dir=initial_submission_dir,
            heartbeat=heartbeat,
            candidate_generator=candidate_generator,
            candidate_generator_fallback=candidate_generator_fallback,
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            submit_score_threshold=submit_score_threshold,
            learning_objective=learning_objective,
            doom_loop_report_path=doom_loop_report_path,
            hypothesis_kpi_report_path=hypothesis_kpi_report_path,
            repair_policy_summary_path=repair_policy_summary_path,
            input_research_chronicle_path=input_research_chronicle_path,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            task2_test_path=task2_test_path,
            task2_target_path=task2_target_path,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            heartbeat_runner=heartbeat_runner,
            force=force,
        )
    expansion_report_path = Path(handoff["artifacts"]["expansion_report_path"])
    queue = sandbox_queue_runner(
        expansion_report_path=expansion_report_path,
        context_path=task1_target_path,
        target_path=task1_target_path,
        initial_submission_dir=_heartbeat_best_submission_dir(
            heartbeat=heartbeat,
            fallback=initial_submission_dir,
        ),
        output_dir=workspace / "sandbox_queue",
        submission_id="team_hermes_runtime_expansion",
        max_items=3,
        failure_memory_path=workspace / "sandbox_queue" / "failure_memory.json",
        reference_path=None,
        reference_sample_limit=None,
        timing_profile=None,
        timing_factor=None,
        force=force,
    )
    evaluation = queue_evaluation_runner(
        queue_report_path=Path(queue["queue_report_path"]),
        output_dir=workspace / "sandbox_queue_evaluation",
        registry_path=workspace / "literature_registry.json",
        current_best_score=current_best_score,
        force=force,
    )
    summary = registry_summary_runner(
        registry_path=Path(evaluation["artifacts"]["registry_path"]),
        output_path=workspace / "hermes_registry_summary.json",
        current_best_score=current_best_score,
        force=force,
    )
    refinement_retest = _maybe_run_refinement_retest(
        summary=summary,
        workspace=workspace,
        expansion_report_path=expansion_report_path,
        current_best_score=current_best_score,
        task1_target_path=task1_target_path,
        task1_reference_path=task1_reference_path,
        initial_submission_dir=_heartbeat_best_submission_dir(
            heartbeat=heartbeat,
            fallback=initial_submission_dir,
        ),
        sandbox_queue_runner=sandbox_queue_runner,
        queue_evaluation_runner=queue_evaluation_runner,
        registry_summary_runner=registry_summary_runner,
        refinement_runner=refinement_runner,
        evidence_accumulation_plan=evidence_accumulation_plan,
        force=force,
    )
    blocked_authoring = _maybe_prepare_blocked_executor_authoring(
        summary=refinement_retest.get("summary") or summary,
        queue_report_path=Path(queue["queue_report_path"]),
        workspace=workspace,
        trace_file=trace_file,
        task1_target_path=task1_target_path,
        algorithm_candidates_path=Path(handoff["artifacts"]["algorithm_candidates_path"]),
        registry_path=Path(evaluation["artifacts"]["registry_path"]),
        current_best_score=current_best_score,
        blocked_executor_authoring_runner=blocked_executor_authoring_runner,
        authoring_audit_runner=authoring_audit_runner,
        force=force,
    )
    strategy_growth_bridge = _maybe_run_strategy_growth_bridge(
        summary=refinement_retest.get("summary") or summary,
        workspace=workspace,
        strategy_proposal_runner=strategy_proposal_runner,
        strategy_evidence_runner=strategy_evidence_runner,
        strategy_rubric_runner=strategy_rubric_runner,
        strategy_promotion_runner=strategy_promotion_runner,
        evidence_accumulation_runner=evidence_accumulation_runner,
        evidence_accumulation_plan=evidence_accumulation_plan,
        force=force,
    )
    trace_store = HypothesisLabTraceStore(trace_file)
    recorded_expansion_candidate_count = _record_expansion_candidate_events(
        trace_store=trace_store,
        evaluation=evaluation,
        implementation_source="hermes_runtime_expansion_pipeline",
        event_tag="expansion_candidate",
    )
    recorded_refinement_candidate_count = 0
    if isinstance(refinement_retest.get("evaluation"), dict):
        recorded_refinement_candidate_count = _record_expansion_candidate_events(
            trace_store=trace_store,
            evaluation=refinement_retest["evaluation"],
            implementation_source="hermes_runtime_refinement_retest",
            event_tag="refinement_candidate",
        )
    trace_store.append_event(
        event_type="hermes_runtime_expansion_pipeline",
        payload={
            "current_best_score": current_best_score,
            "handoff_action_count": handoff["metrics"].get(
                "expansion_action_count"
            ),
            "sandbox_executed_count": queue["metrics"].get("executed_count"),
            "promotion_count": evaluation["metrics"].get("promotion_count"),
            "learning_signal_count": summary["metrics"].get(
                "learning_signal_count"
            ),
            "recorded_expansion_candidate_count": recorded_expansion_candidate_count,
            "recorded_refinement_candidate_count": (
                recorded_refinement_candidate_count
            ),
            "blocked_executor_authoring_task_count": blocked_authoring["metrics"].get(
                "blocked_executor_authoring_task_count",
                0,
            ),
            "refinement_retest_executed": refinement_retest["metrics"].get(
                "refinement_retest_executed",
                False,
            ),
            "strategy_growth_bridge_executed": strategy_growth_bridge["metrics"].get(
                "strategy_growth_bridge_executed",
                False,
            ),
            "blocked_executor_authoring_approved_count": blocked_authoring[
                "metrics"
            ].get("blocked_executor_authoring_approved_count", 0),
            "next_command": blocked_authoring.get("next_command")
            or strategy_growth_bridge.get("next_command")
            or refinement_retest.get("next_command")
            or summary["next_command"],
        },
        run_id="local",
        source_artifact_path=evaluation["queue_evaluation_report_path"],
        tags=["hermes", "runtime_closed_loop", "expansion_pipeline"],
    )
    return {
        "state_hint": "HERMES_RUNTIME_EXPANSION_PIPELINE_RAN",
        "artifacts": {
            "expansion_handoff_report_path": handoff["handoff_report_path"],
            "expansion_algorithm_candidates_path": handoff["artifacts"][
                "algorithm_candidates_path"
            ],
            "expansion_report_path": handoff["artifacts"]["expansion_report_path"],
            "expansion_sandbox_queue_report_path": queue["queue_report_path"],
            "expansion_queue_evaluation_report_path": evaluation[
                "queue_evaluation_report_path"
            ],
            "expansion_literature_registry_path": evaluation["artifacts"][
                "registry_path"
            ],
            "expansion_registry_summary_path": summary["summary_path"],
            **refinement_retest["artifacts"],
            **strategy_growth_bridge["artifacts"],
            **blocked_authoring["artifacts"],
        },
        "metrics": {
            "expansion_candidate_count": handoff["metrics"].get(
                "algorithm_candidate_count"
            ),
            "expansion_action_count": handoff["metrics"].get(
                "expansion_action_count"
            ),
            "expansion_sandbox_executed_count": queue["metrics"].get(
                "executed_count"
            ),
            "expansion_sandbox_skipped_count": queue["metrics"].get("skipped_count"),
            "expansion_promotion_count": evaluation["metrics"].get(
                "promotion_count"
            ),
            "expansion_registry_event_count": evaluation["metrics"].get(
                "registry_event_count"
            ),
            "expansion_learning_signal_count": summary["metrics"].get(
                "learning_signal_count"
            ),
            **refinement_retest["metrics"],
            **strategy_growth_bridge["metrics"],
            "recorded_expansion_candidate_count": recorded_expansion_candidate_count,
            "recorded_refinement_candidate_count": (
                recorded_refinement_candidate_count
            ),
            **blocked_authoring["metrics"],
        },
        "handoff": {
            "next_command": handoff["next_command"],
        },
        "sandbox_queue": {
            "next_command": queue["next_command"],
        },
        "evaluation": {
            "next_command": evaluation["next_command"],
        },
        "registry_summary": {
            "next_command": summary["next_command"],
        },
        "refinement_retest": refinement_retest,
        "strategy_growth_bridge": strategy_growth_bridge,
        "blocked_executor_authoring": blocked_authoring,
        "next_command": (
            blocked_authoring.get("next_command")
            or strategy_growth_bridge.get("next_command")
            or refinement_retest.get("next_command")
            or summary["next_command"]
        ),
}


def _handoff_reports_expansion_exhaustion(handoff: dict[str, Any]) -> bool:
    exhaustion = handoff.get("expansion_exhaustion")
    if isinstance(exhaustion, dict) and exhaustion.get("exhausted") is True:
        metrics = handoff.get("metrics")
        if isinstance(metrics, dict):
            action_count = metrics.get("expansion_action_count")
            executable_count = metrics.get("expansion_executable_count")
            proposal_count = metrics.get("expansion_proposal_count")
            viable_count = sum(
                int(value)
                for value in [action_count, executable_count, proposal_count]
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
            return viable_count == 0
        return True
    if handoff.get("next_command") != "plan-hypothesis-candidates":
        return False
    metrics = handoff.get("metrics")
    if not isinstance(metrics, dict):
        return True
    action_count = metrics.get("expansion_action_count")
    executable_count = metrics.get("expansion_executable_count")
    proposal_count = metrics.get("expansion_proposal_count")
    viable_count = sum(
        int(value)
        for value in [action_count, executable_count, proposal_count]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    )
    return viable_count == 0


def _run_post_expansion_replan_heartbeat(
    *,
    root: Path,
    workspace: Path,
    handoff: dict[str, Any],
    trace_file: Path,
    initial_submission_dir: Path,
    heartbeat: dict[str, Any],
    candidate_generator: str,
    candidate_generator_fallback: str,
    max_batches: int,
    batch_deadline_seconds: float,
    max_rounds_per_batch: int,
    submit_score_threshold: float,
    learning_objective: str,
    doom_loop_report_path: Path,
    hypothesis_kpi_report_path: Path,
    repair_policy_summary_path: Path,
    input_research_chronicle_path: Path | None,
    task1_test_path: Path | None,
    task1_target_path: Path | None,
    task2_test_path: Path | None,
    task2_target_path: Path | None,
    timing_profile: str | None,
    timing_factor: float | None,
    heartbeat_runner: HeartbeatRunner,
    force: bool,
) -> dict[str, Any]:
    handoff_report_path = Path(str(handoff["handoff_report_path"]))
    post_plan_path = workspace / "post_expansion_candidate_plan.json"
    post_plan = build_hypothesis_candidate_plan(
        output_path=post_plan_path,
        candidate_generator=candidate_generator,
        doom_loop_report_path=doom_loop_report_path,
        hypothesis_kpi_report_path=hypothesis_kpi_report_path,
        repair_policy_summary_path=repair_policy_summary_path,
        research_chronicle_path=input_research_chronicle_path,
        expansion_handoff_report_path=handoff_report_path,
        preferred_error_modes=_preferred_error_modes(learning_objective),
        max_schedule_length=max_batches,
        min_viable_candidate_count=1,
        force=force,
    )
    post_schedule = (
        None
        if candidate_generator_fallback == "none"
        else post_plan["candidate_generator_schedule"]
    )
    post_output_dir = workspace / "post_expansion_heartbeat"
    if _candidate_schedule_empty(
        candidate_generator_schedule=post_schedule,
        candidate_plan=post_plan,
    ):
        post_heartbeat = _write_empty_candidate_schedule_heartbeat(
            output_dir=post_output_dir,
            initial_submission_dir=_heartbeat_best_submission_dir(
                heartbeat=heartbeat,
                fallback=initial_submission_dir,
            ),
            candidate_generator=candidate_generator,
            candidate_generator_fallback=candidate_generator_fallback,
            candidate_generator_schedule=post_schedule or [],
            learning_objective=learning_objective,
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            submit_score_threshold=submit_score_threshold,
            doom_loop_report_path=doom_loop_report_path,
            hypothesis_kpi_report_path=hypothesis_kpi_report_path,
            repair_policy_summary_path=repair_policy_summary_path,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            task2_test_path=task2_test_path,
            task2_target_path=task2_target_path,
            force=force,
        )
    else:
        post_heartbeat = heartbeat_runner(
            output_dir=post_output_dir,
            initial_submission_dir=_heartbeat_best_submission_dir(
                heartbeat=heartbeat,
                fallback=initial_submission_dir,
            ),
            max_batches=max_batches,
            batch_deadline_seconds=batch_deadline_seconds,
            max_rounds_per_batch=max_rounds_per_batch,
            submit_score_threshold=submit_score_threshold,
            learning_objective=learning_objective,
            task1_test_path=task1_test_path,
            task1_target_path=task1_target_path,
            task2_test_path=task2_test_path,
            task2_target_path=task2_target_path,
            candidate_generator=candidate_generator,
            candidate_generator_schedule=post_schedule,
            candidate_generator_fallback=candidate_generator_fallback,
            doom_loop_report_path=doom_loop_report_path,
            hypothesis_kpi_report_path=hypothesis_kpi_report_path,
            repair_policy_summary_path=repair_policy_summary_path,
            timing_profile=timing_profile,
            timing_factor=timing_factor,
            force=force,
        )
    trace_store = HypothesisLabTraceStore(trace_file)
    recorded_post_attempt_count = _record_heartbeat_candidate_attempts(
        trace_store=trace_store,
        heartbeat=post_heartbeat,
    )
    expansion_exhaustion = handoff.get("expansion_exhaustion")
    if not isinstance(expansion_exhaustion, dict):
        expansion_exhaustion = {
            "exhausted": True,
            "reason": "handoff_requested_replan",
        }
    trace_store.append_event(
        event_type="hermes_runtime_expansion_exhaustion_replan",
        payload={
            "current_best_score": _heartbeat_best_score(heartbeat),
            "expansion_exhaustion": expansion_exhaustion,
            "post_expansion_candidate_plan_length": len(
                post_plan["candidate_generator_schedule"]
            ),
            "post_expansion_heartbeat_best_score": post_heartbeat.get("best_score"),
            "post_expansion_heartbeat_decision": post_heartbeat.get("decision"),
            "recorded_post_expansion_attempt_count": recorded_post_attempt_count,
            "next_command": "audit-hermes-runtime-closed-loop",
        },
        run_id="local",
        source_artifact_path=post_heartbeat.get("heartbeat_report_path")
        or str(post_plan_path),
        tags=["hermes", "runtime_closed_loop", "expansion_exhaustion_replan"],
    )
    return {
        "state_hint": "HERMES_RUNTIME_EXPANSION_PIPELINE_RAN",
        "artifacts": {
            "expansion_handoff_report_path": handoff["handoff_report_path"],
            "expansion_algorithm_candidates_path": handoff["artifacts"][
                "algorithm_candidates_path"
            ],
            "expansion_report_path": handoff["artifacts"]["expansion_report_path"],
            "post_expansion_candidate_plan_path": str(post_plan_path),
            "post_expansion_heartbeat_report_path": post_heartbeat[
                "heartbeat_report_path"
            ],
        },
        "metrics": {
            "expansion_candidate_count": handoff["metrics"].get(
                "algorithm_candidate_count"
            ),
            "expansion_action_count": handoff["metrics"].get(
                "expansion_action_count"
            ),
            "expansion_exhausted": True,
            "post_expansion_candidate_plan_length": len(
                post_plan["candidate_generator_schedule"]
            ),
            "post_expansion_heartbeat_best_score": post_heartbeat.get("best_score"),
            "post_expansion_heartbeat_decision": post_heartbeat.get("decision"),
            "recorded_post_expansion_attempt_count": recorded_post_attempt_count,
        },
        "handoff": {
            "next_command": handoff["next_command"],
        },
        "expansion_exhaustion": expansion_exhaustion,
        "post_expansion_candidate_plan": {
            "candidate_generator_schedule": post_plan[
                "candidate_generator_schedule"
            ],
            "applied_candidate_generator_schedule": post_schedule,
            "eligible_candidate_count": post_plan["eligible_candidate_count"],
            "decision": post_plan["decision"],
            "expansion_required": post_plan["expansion_required"],
            "learning_constraints": post_plan.get("learning_constraints", {}),
            "next_command": post_plan["next_command"],
        },
        "post_expansion_heartbeat": {
            "best_candidate": post_heartbeat.get("best_candidate"),
            "best_score": post_heartbeat.get("best_score"),
            "decision": post_heartbeat.get("decision"),
            "stop_reason": post_heartbeat.get("stop_reason"),
        },
        "next_command": "audit-hermes-runtime-closed-loop",
    }


def _maybe_run_refinement_retest(
    *,
    summary: dict[str, Any],
    workspace: Path,
    expansion_report_path: Path,
    current_best_score: float,
    task1_target_path: Path,
    task1_reference_path: Path | None,
    initial_submission_dir: Path,
    sandbox_queue_runner: SandboxQueueRunner,
    queue_evaluation_runner: QueueEvaluationRunner,
    registry_summary_runner: RegistrySummaryRunner,
    refinement_runner: RefinementRunner,
    evidence_accumulation_plan: dict[str, Any] | None,
    force: bool,
) -> dict[str, Any]:
    empty = {
        "state_hint": "HERMES_RUNTIME_REFINEMENT_RETEST_SKIPPED",
        "artifacts": {},
        "metrics": {
            "refinement_retest_executed": False,
            "refinement_sandbox_executed_count": 0,
            "refinement_promotion_count": 0,
        },
        "summary": None,
        "next_command": None,
    }
    if not _summary_requests_refinement_retest(summary):
        return empty
    refinement = refinement_runner(
        registry_summary_path=Path(str(summary["summary_path"])),
        output_path=workspace / "candidate_refinement.json",
        algorithm_candidate_id=None,
        force=force,
    )
    if task1_reference_path is None:
        return {
            "state_hint": "HERMES_RUNTIME_REFINEMENT_RETEST_AWAITING_REFERENCE",
            "artifacts": {
                "candidate_refinement_path": refinement["candidate_refinement_path"],
            },
            "metrics": {
                "refinement_retest_executed": False,
                "refinement_sandbox_executed_count": 0,
                "refinement_promotion_count": 0,
            },
            "refinement": refinement,
            "summary": None,
            "next_command": "run-expansion-sandbox-queue",
        }
    queue = sandbox_queue_runner(
        expansion_report_path=expansion_report_path,
        context_path=task1_target_path,
        target_path=task1_target_path,
        initial_submission_dir=initial_submission_dir,
        output_dir=workspace / "refinement_sandbox_queue",
        submission_id="team_hermes_runtime_refinement",
        max_items=1,
        failure_memory_path=workspace / "refinement_sandbox_queue" / "failure_memory.json",
        reference_path=task1_reference_path,
        reference_sample_limit=(
            _accumulation_reference_sample_limit(evidence_accumulation_plan)
            or _refinement_reference_sample_limit(refinement)
        ),
        selected_algorithm_candidate_id=_refinement_algorithm_candidate_id(refinement),
        timing_profile=None,
        timing_factor=None,
        force=force,
    )
    evaluation = queue_evaluation_runner(
        queue_report_path=Path(queue["queue_report_path"]),
        output_dir=workspace / "refinement_sandbox_queue_evaluation",
        registry_path=workspace / "refinement_literature_registry.json",
        current_best_score=current_best_score,
        force=force,
    )
    refined_summary = registry_summary_runner(
        registry_path=Path(evaluation["artifacts"]["registry_path"]),
        output_path=workspace / "refinement_hermes_registry_summary.json",
        current_best_score=current_best_score,
        force=force,
    )
    return {
        "state_hint": "HERMES_RUNTIME_REFINEMENT_RETEST_RAN",
        "artifacts": {
            "candidate_refinement_path": refinement["candidate_refinement_path"],
            "refinement_sandbox_queue_report_path": queue["queue_report_path"],
            "refinement_queue_evaluation_report_path": evaluation[
                "queue_evaluation_report_path"
            ],
            "refinement_literature_registry_path": evaluation["artifacts"][
                "registry_path"
            ],
            "refinement_registry_summary_path": refined_summary["summary_path"],
        },
        "metrics": {
            "refinement_retest_executed": True,
            "refinement_sandbox_executed_count": queue["metrics"].get(
                "executed_count"
            ),
            "refinement_sandbox_skipped_count": queue["metrics"].get("skipped_count"),
            "refinement_promotion_count": evaluation["metrics"].get(
                "promotion_count"
            ),
            "refinement_registry_event_count": evaluation["metrics"].get(
                "registry_event_count"
            ),
            "refinement_learning_signal_count": refined_summary["metrics"].get(
                "learning_signal_count"
            ),
        },
        "refinement": refinement,
        "sandbox_queue": queue,
        "evaluation": evaluation,
        "summary": refined_summary,
        "next_command": refined_summary["next_command"],
    }


def _maybe_run_strategy_growth_bridge(
    *,
    summary: dict[str, Any],
    workspace: Path,
    strategy_proposal_runner: StrategyProposalRunner,
    strategy_evidence_runner: StrategyEvidenceRunner,
    strategy_rubric_runner: StrategyRubricRunner,
    strategy_promotion_runner: StrategyPromotionRunner,
    evidence_accumulation_runner: EvidenceAccumulationRunner,
    evidence_accumulation_plan: dict[str, Any] | None,
    force: bool,
) -> dict[str, Any]:
    action = _strategy_growth_action(summary)
    empty = {
        "state_hint": "HERMES_RUNTIME_STRATEGY_GROWTH_BRIDGE_SKIPPED",
        "artifacts": {},
        "metrics": {
            "strategy_growth_bridge_executed": False,
            "evidence_accumulation_missing_window_count": 0,
        },
        "next_command": None,
    }
    if action is None:
        return empty
    candidate_promotion_path = Path(
        _nonempty_runtime_text(
            action.get("candidate_promotion_path"),
            "candidate_promotion_path",
        )
    )
    bridge_dir = workspace / "strategy_growth_bridge"
    proposal = strategy_proposal_runner(
        candidate_promotion_path=candidate_promotion_path,
        output_path=bridge_dir / "strategy_proposal.json",
        force=force,
    )
    evidence = strategy_evidence_runner(
        candidate_promotion_path=candidate_promotion_path,
        extra_candidate_promotion_paths=_accumulation_candidate_promotion_paths(
            evidence_accumulation_plan,
            current_candidate_promotion_path=candidate_promotion_path,
        ),
        output_path=bridge_dir / "strategy_evidence_summary.json",
        force=force,
    )
    rubric = strategy_rubric_runner(
        proposal_path=Path(proposal["strategy_proposal_path"]),
        output_path=bridge_dir / "strategy_rubric.json",
        min_evaluation_windows=2,
        max_inference_time_regression=0.1,
    )
    promotion = strategy_promotion_runner(
        proposal_path=Path(proposal["strategy_proposal_path"]),
        rubric_path=Path(rubric["experiment_rubric_path"]),
        evidence_summary_path=Path(evidence["evidence_summary_path"]),
        output_path=bridge_dir / "strategy_promotion.json",
    )
    accumulation = None
    next_command = promotion.get("next_command")
    if "insufficient_evaluation_windows" in _string_items(
        promotion.get("reason_codes")
    ):
        accumulation = evidence_accumulation_runner(
            strategy_promotion_report_path=Path(promotion["promotion_report_path"]),
            strategy_evidence_summary_path=Path(evidence["evidence_summary_path"]),
            output_path=bridge_dir / "evidence_accumulation_plan.json",
            force=force,
        )
        next_command = accumulation.get("next_command")
    artifacts = {
        "strategy_growth_proposal_path": proposal["strategy_proposal_path"],
        "strategy_growth_evidence_summary_path": evidence["evidence_summary_path"],
        "strategy_growth_rubric_path": rubric["experiment_rubric_path"],
        "strategy_growth_promotion_report_path": promotion["promotion_report_path"],
    }
    if accumulation is not None:
        artifacts["evidence_accumulation_plan_path"] = accumulation[
            "evidence_accumulation_plan_path"
        ]
    return {
        "state_hint": "HERMES_RUNTIME_STRATEGY_GROWTH_BRIDGE_RAN",
        "artifacts": artifacts,
        "metrics": {
            "strategy_growth_bridge_executed": True,
            "strategy_growth_promotion_decision": promotion.get("decision"),
            "strategy_growth_promote_to_skill_candidate": bool(
                promotion.get("promote_to_skill_candidate")
            ),
            "strategy_growth_evaluation_window_count": evidence.get(
                "evaluation_window_count"
            ),
            "evidence_accumulation_missing_window_count": (
                accumulation.get("missing_window_count") if accumulation else 0
            ),
        },
        "action": action,
        "proposal": proposal,
        "evidence": evidence,
        "rubric": rubric,
        "promotion": promotion,
        "evidence_accumulation": accumulation,
        "next_command": next_command,
    }


def _load_evidence_accumulation_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeClosedLoopError(
            f"evidence accumulation plan not found: {path}"
        )
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeClosedLoopError(
            f"evidence accumulation plan is not valid JSON: {path}"
        ) from exc
    if plan.get("state_hint") != "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED":
        raise HermesRuntimeClosedLoopError(
            "evidence accumulation plan has invalid state_hint"
        )
    return plan


def _load_expansion_registry_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeClosedLoopError(
            f"expansion registry summary not found: {path}"
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeClosedLoopError(
            f"expansion registry summary is not valid JSON: {path}"
        ) from exc
    if summary.get("state_hint") != "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT":
        raise HermesRuntimeClosedLoopError(
            "expansion registry summary has invalid state_hint"
        )
    return summary


def _load_expansion_handoff_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeClosedLoopError(
            f"expansion handoff report not found: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeClosedLoopError(
            f"expansion handoff report is not valid JSON: {path}"
        ) from exc
    if report.get("state_hint") != "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN":
        raise HermesRuntimeClosedLoopError(
            "expansion handoff report has invalid state_hint"
        )
    return report


def _load_audit_repair_tasks(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise HermesRuntimeClosedLoopError(f"audit repair tasks not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeClosedLoopError(
            f"audit repair tasks are not valid JSON: {path}"
        ) from exc
    if report.get("state_hint") != "HERMES_RUNTIME_AUDIT_REPAIR_TASKS_PREPARED":
        raise HermesRuntimeClosedLoopError("audit repair tasks have invalid state_hint")
    return report


def _preflight_audit_repair_inputs(
    *,
    audit_repair_tasks: dict[str, Any] | None,
    expansion_registry_summary: dict[str, Any] | None,
    expansion_handoff_report: dict[str, Any] | None,
) -> None:
    if not _repair_tasks_include(audit_repair_tasks, "memory_bridge_inputs"):
        if not _repair_tasks_include(
            audit_repair_tasks,
            "cluster_saturation_consistency",
        ):
            return
        label = "cluster_saturation_consistency"
    else:
        label = "memory_bridge_inputs"
    if expansion_registry_summary is not None or expansion_handoff_report is not None:
        return
    raise HermesRuntimeClosedLoopError(
        f"{label} repair requires expansion_handoff_report_path "
        "or expansion_registry_summary_path"
    )


def _repair_tasks_include(
    audit_repair_tasks: dict[str, Any] | None,
    audit_error: str,
) -> bool:
    if not isinstance(audit_repair_tasks, dict):
        return False
    tasks = audit_repair_tasks.get("repair_tasks")
    if not isinstance(tasks, list):
        return False
    return any(
        isinstance(task, dict) and task.get("audit_error") == audit_error
        for task in tasks
    )


def _accumulation_reference_sample_limit(plan: dict[str, Any] | None) -> int | None:
    if plan is None:
        return None
    for action in _object_items(plan.get("recommended_actions")):
        if action.get("action") != "run_legal_reference_retest":
            continue
        args = action.get("suggested_sandbox_args")
        if not isinstance(args, dict):
            continue
        value = args.get("reference_sample_limit")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _accumulation_candidate_promotion_paths(
    plan: dict[str, Any] | None,
    *,
    current_candidate_promotion_path: Path,
) -> list[Path]:
    if plan is None:
        return []
    current = str(current_candidate_promotion_path)
    paths: list[Path] = []
    seen = {current}
    for value in _string_items(plan.get("source_candidate_promotion_paths")):
        if value in seen:
            continue
        seen.add(value)
        paths.append(Path(value))
    return paths


def _strategy_growth_action(summary: dict[str, Any]) -> dict[str, Any] | None:
    for action in _object_items(summary.get("candidate_action_queue")):
        if (
            action.get("recommended_action") == "record_strategy_proposal"
            and isinstance(action.get("candidate_promotion_path"), str)
            and action.get("candidate_promotion_path")
        ):
            return action
    return None


def _summary_requests_refinement_retest(summary: dict[str, Any]) -> bool:
    if summary.get("next_command") == "refine-algorithm-candidate":
        return True
    for action in _object_items(summary.get("candidate_action_queue")):
        if action.get("recommended_action") == "refine_with_legal_reference":
            return True
    return False


def _refinement_reference_sample_limit(refinement: dict[str, Any]) -> int | None:
    args = refinement.get("suggested_sandbox_args")
    if not isinstance(args, dict):
        return None
    value = args.get("reference_sample_limit")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _refinement_algorithm_candidate_id(refinement: dict[str, Any]) -> str:
    value = refinement.get("algorithm_candidate_id")
    if not isinstance(value, str) or not value.strip():
        raise HermesRuntimeClosedLoopError(
            "candidate refinement is missing algorithm_candidate_id"
        )
    return value.strip()


def _record_expansion_candidate_events(
    *,
    trace_store: HypothesisLabTraceStore,
    evaluation: dict[str, Any],
    implementation_source: str,
    event_tag: str,
) -> int:
    count = 0
    for item in _object_items(evaluation.get("items")):
        if item.get("status") != "evaluated":
            continue
        candidate_id = item.get("algorithm_candidate_id")
        family = item.get("candidate_family")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if not isinstance(family, str) or not family:
            continue
        payload = {
            "algorithm_candidate_id": candidate_id,
            "candidate_family": family,
            "implementation_source": implementation_source,
            "outcome": _expansion_candidate_outcome(item.get("decision")),
            "score": _optional_float(item.get("score")),
            "score_delta": _optional_float(item.get("score_delta_vs_current_best")),
            "evidence_path": item.get("evidence_path"),
            "promotion_path": item.get("promotion_path"),
            "registry_event_path": item.get("registry_event_path"),
        }
        reason_codes = _string_items(item.get("reason_codes"))
        if reason_codes:
            payload["reason_codes"] = reason_codes
        trace_store.append_event(
            event_type="hypothesis_lab_run",
            payload=payload,
            run_id="local",
            source_artifact_path=item.get("promotion_path")
            or item.get("evidence_path")
            or evaluation.get("queue_evaluation_report_path"),
            tags=["hermes", "runtime_closed_loop", event_tag],
        )
        count += 1
    return count


def _expansion_candidate_outcome(decision: Any) -> str:
    if decision == "promote":
        return "promoted"
    if decision in {"deprioritize", "reject", "blocked", "refine"}:
        return str(decision)
    return "observed"


def _existing_failure_memory_path(workspace: Path) -> Path | None:
    path = workspace / "failure_memory.json"
    return path if path.is_file() else None


def _maybe_prepare_blocked_executor_authoring(
    *,
    summary: dict[str, Any],
    queue_report_path: Path,
    workspace: Path,
    trace_file: Path,
    task1_target_path: Path,
    algorithm_candidates_path: Path,
    registry_path: Path,
    current_best_score: float,
    blocked_executor_authoring_runner: BlockedExecutorAuthoringRunner,
    authoring_audit_runner: AuthoringAuditRunner,
    force: bool,
) -> dict[str, Any]:
    empty = {
        "state_hint": "HERMES_RUNTIME_BLOCKED_EXECUTOR_AUTHORING_SKIPPED",
        "artifacts": {},
        "metrics": {
            "blocked_executor_authoring_task_count": 0,
            "blocked_executor_authoring_approved_count": 0,
            "blocked_executor_authoring_rejected_count": 0,
        },
        "audits": [],
        "next_command": None,
    }
    if not _summary_requests_blocked_executor_authoring(summary):
        return empty
    prepared = blocked_executor_authoring_runner(
        queue_report_path=queue_report_path,
        output_dir=workspace / "blocked_executor_authoring",
        context_path=task1_target_path,
        target_path=task1_target_path,
        submission_id="team_hermes_runtime_blocked_executor",
        max_tasks=1,
        force=force,
    )
    audits = []
    for task in _object_items(prepared.get("tasks")):
        authoring_path = task.get("authoring_task_path")
        if not isinstance(authoring_path, str) or not authoring_path.strip():
            continue
        audit = authoring_audit_runner(
            authoring_task_path=Path(authoring_path),
            skill_root=Path("skills"),
            output_path=Path(authoring_path).with_name(
                "bounded_executor_authoring_audit.json"
            ),
            force=force,
        )
        audits.append(audit)
    approved_count = sum(1 for audit in audits if audit.get("decision") == "approved")
    rejected_count = sum(1 for audit in audits if audit.get("decision") != "approved")
    completion_tasks = _write_bounded_executor_completion_tasks(
        tasks=_object_items(prepared.get("tasks")),
        audits=audits,
        algorithm_candidates_path=algorithm_candidates_path,
        registry_path=registry_path,
        current_best_score=current_best_score,
        trace_file=trace_file,
        force=force,
    )
    next_command = (
        "agent-author-bounded-executor"
        if approved_count > 0
        else prepared.get("next_command")
    )
    return {
        "state_hint": "HERMES_RUNTIME_BLOCKED_EXECUTOR_AUTHORING_PREPARED",
        "artifacts": {
            "blocked_executor_tasks_path": prepared.get(
                "blocked_executor_tasks_path"
            ),
            "blocked_executor_authoring_audit_paths": [
                audit.get("authoring_audit_path")
                for audit in audits
                if audit.get("authoring_audit_path")
            ],
            "blocked_executor_completion_task_paths": [
                task["completion_task_path"] for task in completion_tasks
            ],
        },
        "metrics": {
            "blocked_executor_authoring_task_count": int(
                prepared.get("metrics", {}).get("task_count", len(audits))
            ),
            "blocked_executor_authoring_approved_count": approved_count,
            "blocked_executor_authoring_rejected_count": rejected_count,
            "blocked_executor_completion_task_count": len(completion_tasks),
        },
        "prepared": prepared,
        "audits": audits,
        "completion_tasks": completion_tasks,
        "next_command": next_command,
    }


def _write_bounded_executor_completion_tasks(
    *,
    tasks: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    algorithm_candidates_path: Path,
    registry_path: Path,
    current_best_score: float,
    trace_file: Path,
    force: bool,
) -> list[dict[str, Any]]:
    audits_by_authoring = {
        str(audit.get("authoring_task_path")): audit
        for audit in audits
        if isinstance(audit.get("authoring_task_path"), str)
    }
    completion_tasks = []
    for task in tasks:
        authoring_task_path = task.get("authoring_task_path")
        if not isinstance(authoring_task_path, str) or not authoring_task_path.strip():
            continue
        audit = audits_by_authoring.get(authoring_task_path)
        if not isinstance(audit, dict) or audit.get("decision") != "approved":
            continue
        output_path = Path(authoring_task_path).with_name(
            "bounded_executor_completion_task.json"
        )
        if output_path.exists() and not force:
            raise HermesRuntimeClosedLoopError(
                f"bounded executor completion task exists: {output_path}"
            )
        command = (
            "uv run --frozen pde-agent complete-bounded-executor-authoring-task "
            f"--authoring-task-path {Path(authoring_task_path)} "
            f"--algorithm-candidates-path {algorithm_candidates_path} "
            f"--registry-path {registry_path} "
            f"--current-best-score {float(current_best_score)} "
            f"--trace-path {trace_file} "
            "--skill-root skills --force"
        )
        payload = {
            "schema_version": 1,
            "state_hint": "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED",
            "completion_task_path": str(output_path),
            "algorithm_candidate_id": task.get("algorithm_candidate_id"),
            "authoring_task_path": str(Path(authoring_task_path)),
            "authoring_audit_path": audit.get("authoring_audit_path"),
            "executor_path": task.get("executor_path"),
            "algorithm_candidates_path": str(algorithm_candidates_path),
            "registry_path": str(registry_path),
            "current_best_score": float(current_best_score),
            "trace_path": str(trace_file),
            "command": command,
            "next_command": "complete-bounded-executor-authoring-task",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        completion_tasks.append(payload)
    return completion_tasks


def _summary_requests_blocked_executor_authoring(summary: dict[str, Any]) -> bool:
    if summary.get("next_command") in {
        "create-executor-design-request",
        "prepare-expansion-blocked-executor-authoring-tasks",
        "audit-bounded-executor-authoring-task",
    }:
        return True
    for signal in _object_items(summary.get("learning_signals")):
        if signal.get("signal") == "executor_gap_blocks_algorithm_family":
            return True
    return False


def _heartbeat_best_score(heartbeat: dict[str, Any]) -> float:
    value = heartbeat.get("best_score")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    best = heartbeat.get("best_candidate")
    if isinstance(best, dict):
        score = best.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool):
            return float(score)
    return 0.0


def _heartbeat_best_submission_dir(
    *,
    heartbeat: dict[str, Any],
    fallback: Path,
) -> Path:
    best = heartbeat.get("best_candidate")
    if isinstance(best, dict):
        value = best.get("submission_dir")
        if isinstance(value, str) and value.strip():
            return Path(value)
    return fallback


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _nonempty_runtime_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HermesRuntimeClosedLoopError(f"{label} must be a non-empty string")
    return value.strip()


def _preferred_error_modes(learning_objective: str) -> list[str]:
    if learning_objective in {"early_accuracy", "early_segment", "score"}:
        return ["early_segment", "transition_window"]
    if learning_objective in {"runtime", "time"}:
        return ["runtime", "baseline_control"]
    if learning_objective in {"accuracy", "late_accuracy"}:
        return ["late_stability", "global_score"]
    return []


def _discover_learning_constraint_smoke_report(root: Path) -> Path | None:
    candidates = [
        root / "learning_constraint_smoke" / "hypothesis_learning_constraint_smoke.json",
        root / "hypothesis_learning_constraint_smoke.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = _read_optional_json(path)
        if (
            payload.get("state_hint") == "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"
            and payload.get("decision") == "passed"
        ):
            return path
    return None


def _record_heartbeat_candidate_attempts(
    *,
    trace_store: HypothesisLabTraceStore,
    heartbeat: dict[str, Any],
) -> int:
    count = 0
    for batch in _iter_heartbeat_batches(heartbeat):
        batch_report = _read_optional_json(batch.get("batch_report_path"))
        trajectory = batch_report.get("trajectory")
        if not isinstance(trajectory, list):
            continue
        for item in trajectory:
            if not isinstance(item, dict) or int(item.get("round_index", 0)) <= 0:
                continue
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id == "initial":
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            score_delta = _optional_float(item.get("score_delta_vs_best"))
            family = _candidate_family_from_attempt(
                batch=batch,
            )
            implementation_source = metadata.get("source")
            trace_store.append_event(
                event_type="hypothesis_lab_run",
                payload={
                    "algorithm_candidate_id": candidate_id,
                    "candidate_family": family,
                    "implementation_source": implementation_source
                    if isinstance(implementation_source, str)
                    else None,
                    "outcome": _candidate_attempt_outcome(
                        rules_passed=item.get("rules_passed"),
                        score_delta=score_delta,
                    ),
                    "score": _optional_float(item.get("score")),
                    "score_delta": score_delta,
                    "submission_dir": item.get("submission_dir"),
                    "judge_report_path": item.get("judge_report_path"),
                    "batch_report_path": batch.get("batch_report_path"),
                },
                run_id="local",
                source_artifact_path=item.get("judge_report_path")
                or batch.get("batch_report_path"),
                tags=["hermes", "candidate_attempt"],
            )
            count += 1
    return count


def _iter_heartbeat_batches(heartbeat: dict[str, Any]) -> list[dict[str, Any]]:
    batches = heartbeat.get("batches")
    if not isinstance(batches, list):
        return []
    return [batch for batch in batches if isinstance(batch, dict)]


def _read_optional_json(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str | Path):
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_family_from_attempt(
    *,
    batch: dict[str, Any],
) -> str:
    generator = batch.get("candidate_generator")
    if isinstance(generator, str) and generator:
        return generator
    return "unknown_candidate_family"


def _candidate_attempt_outcome(
    *,
    rules_passed: Any,
    score_delta: float | None,
) -> str:
    if rules_passed is False:
        return "failed"
    if score_delta is not None and score_delta > 0.0:
        return "promoted"
    return "failed"


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, str) and item})


def _preflight_task_pair(
    *,
    task_id: int,
    test_path: Path | None,
    target_path: Path | None,
) -> None:
    if test_path is None or target_path is None:
        return
    if not test_path.exists() or not target_path.exists():
        return
    test_shape = _hdf5_tensor_shape(test_path)
    target_shape = _hdf5_tensor_shape(target_path)
    if test_shape[0] != target_shape[0]:
        raise HermesRuntimeClosedLoopError(
            f"Task {task_id} sample count mismatch: "
            f"test N={test_shape[0]} ({test_path}) != "
            f"target N={target_shape[0]} ({target_path})"
        )


def _preflight_task1_reference(*, reference_path: Path | None) -> None:
    if reference_path is None:
        return
    if not reference_path.exists():
        raise HermesRuntimeClosedLoopError(
            f"Task 1 reference file not found: {reference_path}"
        )
    shape = _hdf5_tensor_shape(reference_path)
    if shape[1] < 200:
        raise HermesRuntimeClosedLoopError(
            f"Task 1 reference must contain at least 200 time steps, got {shape}"
        )
    if shape[2] != 256:
        raise HermesRuntimeClosedLoopError(
            f"Task 1 reference spatial width must be 256, got {shape}"
        )


def _hdf5_tensor_shape(path: Path, *, dataset_key: str = "tensor") -> tuple[int, ...]:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise HermesRuntimeClosedLoopError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        shape = tuple(int(dim) for dim in handle[dataset_key].shape)
    if len(shape) != 3:
        raise HermesRuntimeClosedLoopError(
            f"expected rank-3 tensor in {path}, got shape {shape}"
        )
    return shape
