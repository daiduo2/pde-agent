from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_doom_loop import (
    STATE_HINT as DOOM_LOOP_STATE_HINT,
)
from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
from pde_agent.core.hypothesis_candidate_planner import (
    STATE_HINT as CANDIDATE_PLAN_STATE_HINT,
)
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_algorithm_designer import (
    DESIGN_STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGN_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer import (
    VALIDATION_STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGN_VALIDATION_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer import (
    adapt_hypothesis_algorithm_sandbox_item_to_blueprint,
    build_hypothesis_algorithm_design,
    plan_hypothesis_algorithm_sandbox,
    record_hypothesis_algorithm_evidence,
    validate_hypothesis_algorithm_design,
)
from pde_agent.core.hypothesis_algorithm_designer_stability import (
    STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer_stability import (
    run_hypothesis_algorithm_designer_stability_batch,
)
from pde_agent.core.hypothesis_kpi import STATE_HINT as KPI_STATE_HINT
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
from pde_agent.core.hermes_learning_constraint_smoke import (
    STATE_HINT as LEARNING_CONSTRAINT_SMOKE_STATE_HINT,
)
from pde_agent.core.hermes_learning_constraint_smoke import (
    run_hypothesis_learning_constraint_smoke,
)
from pde_agent.core.hermes_runtime_memory_bridge_smoke import (
    STATE_HINT as HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_STATE_HINT,
)
from pde_agent.core.hermes_runtime_memory_bridge_smoke import (
    run_hermes_runtime_memory_bridge_smoke,
)
from pde_agent.core.hermes_auto_repair_memory_acceptance_smoke import (
    STATE_HINT as HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_STATE_HINT,
)
from pde_agent.core.hermes_auto_repair_memory_acceptance_smoke import (
    run_hermes_auto_repair_memory_acceptance_smoke,
)
from pde_agent.core.hermes_hypothesis_lab_infrastructure_audit import (
    STATE_HINT as HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDIT_STATE_HINT,
)
from pde_agent.core.hermes_hypothesis_lab_infrastructure_audit import (
    audit_hermes_hypothesis_lab_infrastructure,
)
from pde_agent.core.hermes_availability_audit import (
    STATE_HINT as HERMES_AVAILABILITY_AUDIT_STATE_HINT,
)
from pde_agent.core.hermes_availability_audit import (
    audit_agent_runtime_availability,
    audit_hermes_availability,
)
from pde_agent.core.hypothesis_lab_trace import (
    HypothesisLabTraceStore,
    SUMMARY_STATE_HINT,
    summarize_hypothesis_lab_trace,
)
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox
from pde_agent.core.json_output import success_payload
from pde_agent.core.hermes_runtime_closed_loop import (
    STATE_HINT as HERMES_RUNTIME_CLOSED_LOOP_STATE_HINT,
)
from pde_agent.core.hermes_runtime_closed_loop import run_hermes_runtime_closed_loop
from pde_agent.core.hermes_runtime_closed_loop_audit import (
    STATE_HINT as HERMES_RUNTIME_CLOSED_LOOP_AUDIT_STATE_HINT,
)
from pde_agent.core.hermes_runtime_closed_loop_audit import (
    audit_hermes_runtime_closed_loop,
)
from pde_agent.core.hermes_runtime_audit_chain import (
    REPAIR_TASK_RESOLUTION_STATE_HINT,
    REPAIR_TASK_RUN_STATE_HINT,
    STATE_HINT as HERMES_RUNTIME_AUDIT_CHAIN_STATE_HINT,
)
from pde_agent.core.hermes_runtime_audit_chain import (
    resolve_audit_repair_task,
    run_resolved_audit_repair_task,
    run_hermes_runtime_audit_chain,
)
from pde_agent.core.hermes_repair_policy_summary import (
    STATE_HINT as HERMES_REPAIR_POLICY_SUMMARY_STATE_HINT,
)
from pde_agent.core.hermes_repair_policy_summary import (
    build_repair_policy_summary,
)
from pde_agent.core.hermes_task1_agent_capability_benchmark import (
    STATE_HINT as HERMES_TASK1_AGENT_CAPABILITY_BENCHMARK_STATE_HINT,
)
from pde_agent.core.hermes_task1_agent_capability_benchmark import (
    evaluate_hermes_task1_agent_capability,
)
from pde_agent.core.pde_literature_extraction_skill import (
    STATE_HINT as PDE_LITERATURE_EXTRACTION_SKILL_STATE_HINT,
)
from pde_agent.core.pde_literature_extraction_skill import (
    prepare_pde_literature_extraction_skill,
)
from pde_agent.core.scientific_tool_router import (
    STATE_HINT as TOOL_CATALOG_STATE_HINT,
)
from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog
from pde_agent.core.task1_short_window_training import (
    FAMILY_SELECTION_STATE_HINT as TASK1_SHORT_WINDOW_FAMILY_SELECTION_STATE_HINT,
    STATE_HINT as TASK1_SHORT_WINDOW_TRAINING_STATE_HINT,
)
from pde_agent.core.task1_short_window_training import (
    build_fno_training_harness_contract,
    plan_fno_training_candidates,
    record_fno_training_optimization_evidence,
    run_fno_training_optimization_loop,
    run_task1_short_window_family_selection_loop,
    run_task1_short_window_training_loop,
)


TRACE_EVENT_STATE_HINT = "HYPOTHESIS_LAB_TRACE_EVENT_RECORDED"


def build_fno_training_harness_contract_command(
    *,
    output_path: str,
    workspace_dir: str,
    source_path: str,
    previous_evidence_paths: list[str],
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_candidates: int,
    modes: list[int],
    widths: list[int],
    depths: list[int],
    loss_types: list[str],
    forecast_steps_values: list[int],
    train_steps_values: list[int],
    force: bool,
) -> dict[str, object]:
    report = build_fno_training_harness_contract(
        output_path=Path(output_path),
        workspace_dir=Path(workspace_dir),
        source_path=Path(source_path),
        previous_evidence_paths=[Path(path) for path in previous_evidence_paths],
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        max_candidates=max_candidates,
        modes=modes,
        widths=widths,
        depths=depths,
        loss_types=loss_types,
        forecast_steps_values=forecast_steps_values,
        train_steps_values=train_steps_values,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="build-fno-training-harness-contract",
        state_hint="FNO_TRAINING_HARNESS_CONTRACT_BUILT",
        artifacts={
            "harness_contract_path": output_path,
            "available_actions": report["available_actions"],
            "budget_contract": report["budget_contract"],
            "data_contract": report["data_contract"],
        },
        metrics={
            "action_count": len(report["available_actions"]),
            "max_candidates": report["budget_contract"]["max_candidates"],
            "train_sample_limit": report["budget_contract"]["train_sample_limit"],
            "val_sample_limit": report["budget_contract"]["val_sample_limit"],
        },
        next_hint="choose-action-from-harness-contract",
    )


def plan_fno_training_candidates_command(
    *,
    evidence_paths: list[str],
    output_path: str,
    max_candidates: int,
    modes: list[int],
    widths: list[int],
    depths: list[int],
    loss_types: list[str],
    forecast_steps_values: list[int],
    train_steps_values: list[int],
    force: bool,
) -> dict[str, object]:
    report = plan_fno_training_candidates(
        evidence_paths=[Path(path) for path in evidence_paths],
        output_path=Path(output_path),
        max_candidates=max_candidates,
        modes=modes,
        widths=widths,
        depths=depths,
        loss_types=loss_types,
        forecast_steps_values=forecast_steps_values,
        train_steps_values=train_steps_values,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="plan-fno-training-candidates",
        state_hint="FNO_TRAINING_CANDIDATES_PLANNED",
        artifacts={"candidate_plan_path": output_path},
        metrics={
            "candidate_count": len(report["candidate_configs"]),
            "avoided_config_count": report["avoided_config_count"],
        },
        next_hint=report["next_command"],
    )


def run_fno_training_optimization_loop_command(
    *,
    candidate_plan_path: str,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_fno_training_optimization_loop(
        candidate_plan_path=Path(candidate_plan_path),
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-fno-training-optimization-loop",
        state_hint="FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
        artifacts={
            "optimization_report_path": report["artifacts"][
                "optimization_report_path"
            ],
        },
        metrics={
            "candidate_count": len(report["candidate_records"]),
            "selected_candidate_id": report["selected_candidate"]["candidate_id"],
            "directive_action": report["training_directive"]["action"],
            "best_early_score": report["selected_candidate"]["early_score"],
        },
        next_hint=report["next_command"],
    )


def record_fno_training_optimization_evidence_command(
    *,
    optimization_report_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=Path(optimization_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="record-fno-training-optimization-evidence",
        state_hint="FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
        artifacts={"fno_training_optimization_evidence_path": output_path},
        metrics={
            "selected_candidate_id": evidence["selected_candidate"]["candidate_id"],
            "directive_action": evidence["training_directive"]["action"],
        },
        next_hint=evidence["next_command"],
    )


def run_task1_short_window_training_loop_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    model_family: str,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_task1_short_window_training_loop(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        model_family=model_family,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    loss_curve = report["training"]["loss_curve"]
    early = report["metrics"]["early"]
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-short-window-training-loop",
        state_hint=TASK1_SHORT_WINDOW_TRAINING_STATE_HINT,
        artifacts={
            "report_path": report["artifacts"]["report_path"],
            "checkpoint_path": report["artifacts"]["checkpoint_path"],
            "config_path": report["artifacts"]["config_path"],
            "training_log_path": report["artifacts"]["training_log_path"],
            "prediction_path": report["artifacts"]["prediction_path"],
            "evidence_path": report["artifacts"]["evidence_path"],
        },
        metrics={
            "model_family": report["model"]["model_family"],
            "parameter_count": report["model"]["parameter_count"],
            "train_steps": report["training"]["train_steps"],
            "final_train_loss": loss_curve["train"][-1],
            "final_val_loss": loss_curve["val"][-1],
            "early_rel_mse": early["rel_mse"],
            "early_score": early["score"],
            "decision": report["evidence"]["decision"],
        },
        next_hint=report["next_command"],
    )


def run_task1_short_window_family_selection_loop_command(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    model_families: list[str],
    train_sample_limit: int,
    val_sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    force: bool,
) -> dict[str, object]:
    report = run_task1_short_window_family_selection_loop(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        model_families=model_families,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="run-task1-short-window-family-selection-loop",
        state_hint=TASK1_SHORT_WINDOW_FAMILY_SELECTION_STATE_HINT,
        artifacts={
            "comparison_report_path": report["artifacts"]["comparison_report_path"],
            "comparison_evidence_path": report["artifacts"][
                "comparison_evidence_path"
            ],
        },
        metrics={
            "selected_family": report["selected_family"],
            "family_count": len(report["family_records"]),
            "directive_action": report["training_directive"]["action"],
            "best_early_score": max(
                float(item["early_score"]) for item in report["family_records"]
            ),
        },
        next_hint=report["next_command"],
    )


def record_hypothesis_lab_trace_event_command(
    *,
    trace_path: str,
    event_type: str,
    payload_json: str,
    run_id: str | None,
    source_artifact_path: str | None,
    tags: list[str] | None,
) -> dict[str, object]:
    payload = _parse_json_object(payload_json, "payload_json")
    event = HypothesisLabTraceStore(trace_path).append_event(
        event_type=event_type,
        payload=payload,
        run_id=run_id,
        source_artifact_path=source_artifact_path,
        tags=tags or [],
    )
    return success_payload(
        run_id=run_id or "local",
        command="record-hypothesis-lab-trace-event",
        state_hint=TRACE_EVENT_STATE_HINT,
        artifacts={
            "trace_path": str(trace_path),
            "event_id": event["event_id"],
        },
        metrics={
            "event_type": event["event_type"],
            "candidate_family": payload.get("candidate_family"),
            "outcome": payload.get("outcome"),
        },
        next_hint="summarize-hypothesis-lab-trace",
    )


def summarize_hypothesis_lab_trace_command(
    *,
    trace_path: str,
    output_path: str,
    force: bool = False,
) -> dict[str, object]:
    summary = summarize_hypothesis_lab_trace(
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="summarize-hypothesis-lab-trace",
        state_hint=SUMMARY_STATE_HINT,
        artifacts={
            "trace_path": trace_path,
            "trace_summary_json": output_path,
            "summary_path": output_path,
        },
        metrics={
            "event_count": summary["event_count"],
            "promoted_candidate_count": len(summary["promoted_candidate_ids"]),
            "failed_candidate_family_count": len(summary["failed_candidate_families"]),
        },
        next_hint=summary["next_command"],
    )


def detect_hypothesis_doom_loop_command(
    *,
    trace_path: str,
    output_path: str,
    recent_window: int,
    repeated_family_threshold: int,
    failed_family_threshold: int,
    min_score_delta: float,
    force: bool,
) -> dict[str, object]:
    report = check_hypothesis_doom_loop(
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        recent_window=recent_window,
        repeated_family_threshold=repeated_family_threshold,
        failed_family_threshold=failed_family_threshold,
        min_score_delta=min_score_delta,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="detect-hypothesis-doom-loop",
        state_hint=DOOM_LOOP_STATE_HINT,
        artifacts={
            "trace_path": trace_path,
            "doom_loop_report_json": output_path,
            "doom_loop_report_path": output_path,
        },
        metrics={
            "guard_active": report["guard_active"],
            "decision": report["decision"],
            "blocked_candidate_families": report["blocked_candidate_families"],
        },
        next_hint=report["next_command"],
    )


def build_scientific_tool_catalog_command(
    *,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    catalog = build_scientific_tool_catalog(
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-scientific-tool-catalog",
        state_hint=TOOL_CATALOG_STATE_HINT,
        artifacts={"tool_catalog_path": output_path},
        metrics={
            "tool_count": catalog["tool_count"],
            "risk_levels": catalog["risk_levels"],
        },
        next_hint=catalog["next_command"],
    )


def audit_hermes_availability_command(
    *,
    output_path: str,
    tool_catalog_path: str,
    canary_report_paths: list[str],
    cli_commands: list[str],
    required_commands: list[str],
    force: bool,
) -> dict[str, object]:
    report = audit_hermes_availability(
        output_path=Path(output_path),
        tool_catalog_path=Path(tool_catalog_path),
        canary_report_paths=[Path(path) for path in canary_report_paths],
        cli_commands=cli_commands,
        required_commands=required_commands,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="audit-hermes-availability",
        state_hint=HERMES_AVAILABILITY_AUDIT_STATE_HINT,
        artifacts={
            "availability_audit_path": output_path,
            "tool_catalog_path": tool_catalog_path,
            "canary_report_paths": canary_report_paths,
            "missing_required_commands": report["gaps"][
                "missing_required_commands"
            ],
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def audit_agent_runtime_availability_command(
    *,
    output_path: str,
    runtime_name: str,
    tool_catalog_path: str,
    canary_report_paths: list[str],
    cli_commands: list[str],
    required_commands: list[str],
    force: bool,
) -> dict[str, object]:
    report = audit_agent_runtime_availability(
        runtime_name=runtime_name,
        output_path=Path(output_path),
        tool_catalog_path=Path(tool_catalog_path),
        canary_report_paths=[Path(path) for path in canary_report_paths],
        cli_commands=cli_commands,
        required_commands=required_commands,
        force=force,
    )
    return success_payload(
        run_id=Path(output_path).stem or "local",
        command="audit-agent-runtime-availability",
        state_hint=HERMES_AVAILABILITY_AUDIT_STATE_HINT,
        artifacts={
            "availability_audit_path": output_path,
            "runtime_name": runtime_name,
            "tool_catalog_path": tool_catalog_path,
            "canary_report_paths": canary_report_paths,
            "missing_required_commands": report["gaps"][
                "missing_required_commands"
            ],
        },
        metrics={
            **report["metrics"],
            "runtime_primary": report["runtime"]["primary"],
        },
        next_hint=report["next_command"],
    )


def build_hypothesis_kpi_report_command(
    *,
    trace_path: str,
    output_path: str,
    recent_window: int,
    force: bool,
) -> dict[str, object]:
    report = build_hypothesis_kpi_report(
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        recent_window=recent_window,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hypothesis-kpi-report",
        state_hint=KPI_STATE_HINT,
        artifacts={
            "hypothesis_kpi_report_json": output_path,
            "hypothesis_kpi_report_path": output_path,
            "active_audit_repair_errors": report.get(
                "active_audit_repair_errors",
                [],
            ),
            "repair_nonconvergence_errors": report.get(
                "repair_nonconvergence_errors",
                [],
            ),
            "repair_convergence_errors": report.get(
                "repair_convergence_errors",
                [],
            ),
            "audit_repair_error_counts": report.get("audit_repair_error_counts", {}),
            "audit_repair_consumed_error_counts": report.get(
                "audit_repair_consumed_error_counts",
                {},
            ),
            "audit_repair_converged_error_counts": report.get(
                "audit_repair_converged_error_counts",
                {},
            ),
            "persistent_search_guidance": report.get(
                "persistent_search_guidance",
                [],
            ),
            "search_guidance_action_counts": report.get(
                "search_guidance_action_counts",
                {},
            ),
            "cluster_saturation_signals": report.get(
                "cluster_saturation_signals",
                [],
            ),
            "saturated_candidate_families": report.get(
                "saturated_candidate_families",
                [],
            ),
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def evaluate_hermes_task1_agent_capability_command(
    *,
    chain_report_path: str,
    trace_path: str,
    output_path: str,
    baseline_score: float | None,
    min_score_delta: float,
    min_unique_candidate_families: int,
    min_sandbox_evidence_events: int,
    require_independent_slice_evidence: bool,
    memory_json_path: str | None,
    exploration_schedule_path: str | None,
    require_control_layer_evidence: bool,
    force: bool,
) -> dict[str, object]:
    report = evaluate_hermes_task1_agent_capability(
        chain_report_path=Path(chain_report_path),
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        baseline_score=baseline_score,
        min_score_delta=min_score_delta,
        min_unique_candidate_families=min_unique_candidate_families,
        min_sandbox_evidence_events=min_sandbox_evidence_events,
        require_independent_slice_evidence=require_independent_slice_evidence,
        memory_json_path=Path(memory_json_path) if memory_json_path else None,
        exploration_schedule_path=(
            Path(exploration_schedule_path) if exploration_schedule_path else None
        ),
        require_control_layer_evidence=require_control_layer_evidence,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="evaluate-hermes-task1-agent-capability",
        state_hint=HERMES_TASK1_AGENT_CAPABILITY_BENCHMARK_STATE_HINT,
        artifacts={
            "task1_agent_capability_benchmark_json": output_path,
            "task1_agent_capability_benchmark_path": output_path,
        },
        metrics={
            "decision": report["decision"],
            "score_delta": report["score_progress"]["score_delta"],
            "best_score": report["score_progress"]["best_score"],
            "unique_candidate_family_count": report["candidate_generation"][
                "unique_candidate_family_count"
            ],
            "evidence_event_count": report["sandbox_validation"][
                "evidence_event_count"
            ],
            "invalid_promotion_count": report["promotion_quality"][
                "invalid_promotion_count"
            ],
            "control_layer_memory_loaded": report["control_layer"]["memory_loaded"],
            "control_layer_schedule_loaded": report["control_layer"][
                "schedule_loaded"
            ],
        },
        next_hint=report["next_command"],
    )


def build_repair_policy_summary_command(
    *,
    trace_path: str,
    output_path: str,
    recent_window: int,
    force: bool,
) -> dict[str, object]:
    report = build_repair_policy_summary(
        trace_path=Path(trace_path),
        output_path=Path(output_path),
        recent_window=recent_window,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-repair-policy-summary",
        state_hint=HERMES_REPAIR_POLICY_SUMMARY_STATE_HINT,
        artifacts={
            "repair_policy_summary_json": output_path,
            "repair_policy_summary_path": output_path,
            "converged_repair_errors": report.get("converged_repair_errors", []),
            "nonconverged_repair_errors": report.get(
                "nonconverged_repair_errors",
                [],
            ),
            "repair_error_stats": report.get("repair_error_stats", {}),
            "recommended_actions": report.get("recommended_actions", []),
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def plan_hypothesis_candidates_command(
    *,
    output_path: str,
    candidate_generator: str,
    doom_loop_report_path: str | None,
    hypothesis_kpi_report_path: str | None,
    repair_policy_summary_path: str | None = None,
    research_chronicle_path: str | None = None,
    expansion_handoff_report_path: str | None = None,
    audit_repair_tasks_path: str | None = None,
    preferred_error_modes: list[str] | None = None,
    max_schedule_length: int = 5,
    min_viable_candidate_count: int = 1,
    force: bool = False,
) -> dict[str, object]:
    report = build_hypothesis_candidate_plan(
        output_path=Path(output_path),
        candidate_generator=candidate_generator,
        doom_loop_report_path=Path(doom_loop_report_path)
        if doom_loop_report_path is not None
        else None,
        hypothesis_kpi_report_path=Path(hypothesis_kpi_report_path)
        if hypothesis_kpi_report_path is not None
        else None,
        repair_policy_summary_path=Path(repair_policy_summary_path)
        if repair_policy_summary_path is not None
        else None,
        research_chronicle_path=Path(research_chronicle_path)
        if research_chronicle_path is not None
        else None,
        expansion_handoff_report_path=Path(expansion_handoff_report_path)
        if expansion_handoff_report_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        preferred_error_modes=preferred_error_modes or [],
        max_schedule_length=max_schedule_length,
        min_viable_candidate_count=min_viable_candidate_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-hypothesis-candidates",
        state_hint=CANDIDATE_PLAN_STATE_HINT,
        artifacts={
            "candidate_plan_json": output_path,
            "candidate_plan_path": output_path,
            "candidate_generator_schedule": report.get(
                "candidate_generator_schedule",
                [],
            ),
            "audit_repair_tasks": report.get("audit_repair_tasks", []),
            "audit_repair_policies": report.get("audit_repair_policies", []),
            "repair_nonconvergence_errors": (
                report.get("learning_constraints", {}).get(
                    "repair_nonconvergence_errors",
                    [],
                )
                if isinstance(report.get("learning_constraints"), dict)
                else []
            ),
            "repair_convergence_errors": (
                report.get("learning_constraints", {}).get(
                    "repair_convergence_errors",
                    [],
                )
                if isinstance(report.get("learning_constraints"), dict)
                else []
            ),
            "audit_repair_converged_error_counts": (
                report.get("learning_constraints", {}).get(
                    "audit_repair_converged_error_counts",
                    {},
                )
                if isinstance(report.get("learning_constraints"), dict)
                else {}
            ),
            "repair_policy_recommended_actions": (
                report.get("learning_constraints", {}).get(
                    "repair_policy_recommended_actions",
                    [],
                )
                if isinstance(report.get("learning_constraints"), dict)
                else []
            ),
            "research_chronicle_repair_policy_memory": (
                report.get("learning_constraints", {}).get(
                    "research_chronicle_repair_policy_memory",
                    {},
                )
                if isinstance(report.get("learning_constraints"), dict)
                else {}
            ),
        },
        metrics={
            "candidate_plan_length": len(report["candidate_generator_schedule"]),
            "eligible_candidate_count": report["eligible_candidate_count"],
            "expansion_required": report["expansion_required"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )


def build_hypothesis_algorithm_design_command(
    *,
    output_path: str,
    failure_memory_paths: list[str],
    evidence_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    strategy_library_paths: list[str] | None = None,
    literature_card_paths: list[str] | None = None,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, object]:
    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[Path(path) for path in failure_memory_paths],
        evidence_paths=[Path(path) for path in evidence_paths or []],
        log_paths=[Path(path) for path in log_paths or []],
        strategy_library_paths=[Path(path) for path in strategy_library_paths or []],
        literature_card_paths=[Path(path) for path in literature_card_paths or []],
        output_path=Path(output_path),
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hypothesis-algorithm-design",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGN_STATE_HINT,
        artifacts={
            "algorithm_design_json": output_path,
            "algorithm_design_path": output_path,
            "mechanism_hypotheses": report.get("mechanism_hypotheses", []),
            "algorithm_blueprints": report.get("algorithm_blueprints", []),
            "diversity_audit": report.get("diversity_audit", {}),
        },
        metrics={
            "hypothesis_count": len(report.get("mechanism_hypotheses", [])),
            "blueprint_count": len(report.get("algorithm_blueprints", [])),
            "mechanism_family_count": report.get("designer_rubric", {}).get(
                "mechanism_family_count",
                0,
            )
            if isinstance(report.get("designer_rubric"), dict)
            else 0,
            "decision": report.get("designer_rubric", {}).get("decision")
            if isinstance(report.get("designer_rubric"), dict)
            else None,
            "training_evidence_count": report.get("diagnosis", {}).get(
                "training_evidence_count",
                0,
            )
            if isinstance(report.get("diagnosis"), dict)
            else 0,
            "explicit_evidence_count": report.get("diagnosis", {}).get(
                "explicit_evidence_count",
                0,
            )
            if isinstance(report.get("diagnosis"), dict)
            else 0,
            "source_evidence_types": sorted(
                {
                    str(item.get("source_type"))
                    for item in report.get("diagnosis", {}).get(
                        "evidence_summaries",
                        [],
                    )
                    if isinstance(item, dict) and item.get("source_type")
                }
            )
            if isinstance(report.get("diagnosis"), dict)
            else [],
        },
        next_hint=report["next_command"],
    )


def validate_hypothesis_algorithm_design_command(
    *,
    design_path: str,
    output_path: str,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, object]:
    report = validate_hypothesis_algorithm_design(
        design_path=Path(design_path),
        output_path=Path(output_path),
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="validate-hypothesis-algorithm-design",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGN_VALIDATION_STATE_HINT,
        artifacts={
            "algorithm_design_audit_json": output_path,
            "algorithm_design_audit_path": output_path,
            "design_path": design_path,
            "violations": report.get("violations", []),
        },
        metrics={
            "decision": report["decision"],
            "hypothesis_count": report["hypothesis_count"],
            "mechanism_family_count": report["mechanism_family_count"],
            "blueprint_count": report["blueprint_count"],
            "violation_count": len(report.get("violations", [])),
        },
        next_hint=report["next_command"],
    )


def record_hypothesis_algorithm_evidence_command(
    *,
    design_path: str,
    sandbox_report_path: str,
    output_path: str,
    force: bool = False,
) -> dict[str, object]:
    report = record_hypothesis_algorithm_evidence(
        design_path=Path(design_path),
        sandbox_report_path=Path(sandbox_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-hypothesis-algorithm-evidence",
        state_hint="HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
        artifacts={
            "algorithm_evidence_json": output_path,
            "algorithm_evidence_path": output_path,
            "algorithm_card": report["algorithm_card"],
            "evidence": report["evidence"],
            "failure_taxonomy": report["failure_taxonomy"],
            "next_generation_constraints": report["next_generation_constraints"],
        },
        metrics={
            "decision": report["decision"],
            "failure_code_count": len(report["failure_taxonomy"]["failure_codes"]),
            "avoid_mechanism_family_count": len(
                report["next_generation_constraints"]["avoid_mechanism_families"]
            ),
        },
        next_hint=report["next_command"],
    )


def plan_hypothesis_algorithm_sandbox_command(
    *,
    design_path: str,
    output_path: str,
    max_ready_items: int = 5,
    force: bool = False,
) -> dict[str, object]:
    report = plan_hypothesis_algorithm_sandbox(
        design_path=Path(design_path),
        output_path=Path(output_path),
        max_ready_items=max_ready_items,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-hypothesis-algorithm-sandbox",
        state_hint="HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED",
        artifacts={
            "algorithm_sandbox_plan_json": output_path,
            "algorithm_sandbox_plan_path": output_path,
            "sandbox_items": report["sandbox_items"],
            "blocked_items": report["blocked_items"],
        },
        metrics={
            "ready_count": report["ready_count"],
            "blocked_count": report["blocked_count"],
        },
        next_hint=report["next_command"],
    )


def adapt_hypothesis_algorithm_sandbox_item_command(
    *,
    sandbox_plan_path: str,
    output_path: str,
    blueprint_id: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    plan = json.loads(Path(sandbox_plan_path).read_text(encoding="utf-8"))
    items = plan.get("sandbox_items") if isinstance(plan, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("sandbox plan missing sandbox_items")
    selected = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if blueprint_id is None or item.get("blueprint_id") == blueprint_id:
            selected = item
            break
    if selected is None:
        raise RuntimeError("sandbox item not found")
    blueprint = adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
        sandbox_item=selected,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="adapt-hypothesis-algorithm-sandbox-item",
        state_hint="HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED",
        artifacts={
            "adapted_blueprint_json": output_path,
            "adapted_blueprint_path": output_path,
            "source_algorithm_blueprint_id": blueprint["source_algorithm_blueprint_id"],
            "candidate_family": blueprint["candidate_family"],
        },
        metrics={
            "candidate_family": blueprint["candidate_family"],
            "mechanism_family": blueprint["algorithm_designer_metadata"][
                "mechanism_family"
            ],
        },
        next_hint="run-blueprint-sandbox",
    )


def run_hypothesis_algorithm_sandbox_command(
    *,
    adapted_blueprint_path: str,
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
) -> dict[str, object]:
    blueprint = json.loads(Path(adapted_blueprint_path).read_text(encoding="utf-8"))
    if blueprint.get("state_hint") != "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED":
        raise RuntimeError("adapted blueprint has invalid state_hint")
    report = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        output_dir=Path(output_dir),
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    report["adapted_blueprint"] = {
        "adapted_blueprint_path": adapted_blueprint_path,
        "source_algorithm_blueprint_id": blueprint.get("source_algorithm_blueprint_id"),
        "source_hypothesis_id": blueprint.get("source_hypothesis_id"),
        "mechanism_family": (
            blueprint.get("algorithm_designer_metadata", {}).get("mechanism_family")
            if isinstance(blueprint.get("algorithm_designer_metadata"), dict)
            else None
        ),
    }
    artifacts_for_writeback = (
        report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    )
    sandbox_report_path = artifacts_for_writeback.get("sandbox_report_path")
    if isinstance(sandbox_report_path, str):
        Path(sandbox_report_path).write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    verifier = report.get("verifier") if isinstance(report.get("verifier"), dict) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    return success_payload(
        run_id="local",
        command="run-hypothesis-algorithm-sandbox",
        state_hint="HYPOTHESIS_ALGORITHM_SANDBOX_RAN",
        artifacts={
            "sandbox_report_path": artifacts.get("sandbox_report_path"),
            "judge_report_path": artifacts.get("judge_report_path"),
            "best_submission_dir": artifacts.get("best_submission_dir"),
            "adapted_blueprint_path": adapted_blueprint_path,
        },
        metrics={
            "decision": verifier.get("decision"),
            "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
            "best_candidate_id": report.get("best_candidate", {}).get("candidate_id")
            if isinstance(report.get("best_candidate"), dict)
            else None,
        },
        next_hint="record-hypothesis-algorithm-evidence",
    )


def run_hypothesis_algorithm_designer_stability_batch_command(
    *,
    memory_set_json: str,
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    initial_submission_dir: str | None = None,
    max_sandbox_items_per_set: int = 1,
    force: bool = False,
) -> dict[str, object]:
    memory_sets = json.loads(memory_set_json)
    if not isinstance(memory_sets, list):
        raise RuntimeError("memory_set_json must be a list of path lists")
    normalized_memory_sets = []
    for row in memory_sets:
        if not isinstance(row, list):
            raise RuntimeError("memory_set_json entries must be path lists")
        normalized_memory_sets.append([Path(path) for path in row])
    report = run_hypothesis_algorithm_designer_stability_batch(
        memory_sets=normalized_memory_sets,
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        initial_submission_dir=Path(initial_submission_dir)
        if initial_submission_dir is not None
        else Path(""),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        max_sandbox_items_per_set=max_sandbox_items_per_set,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hypothesis-algorithm-designer-stability-batch",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_STATE_HINT,
        artifacts={
            "stability_report_json": report["stability_report_path"],
            "stability_report_path": report["stability_report_path"],
        },
        metrics=report["summary"],
        next_hint=report["next_command"],
    )


def prepare_pde_literature_extraction_skill_command(
    *,
    literature_extraction_context_path: str,
    output_path: str,
    operator_family: str,
    target_problem: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_pde_literature_extraction_skill(
        literature_extraction_context_path=Path(literature_extraction_context_path),
        output_path=Path(output_path),
        operator_family=operator_family,
        target_problem=target_problem,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-pde-literature-extraction-skill",
        state_hint=PDE_LITERATURE_EXTRACTION_SKILL_STATE_HINT,
        artifacts={"pde_literature_extraction_skill_path": output_path},
        metrics={
            "source_id": report["source_id"],
            "operator_family": report["operator_family"],
            "selected_context_unit_count": report["selected_context_unit_count"],
        },
        next_hint=report["next_command"],
    )


def run_hypothesis_learning_constraint_smoke_command(
    *,
    trace_path: str,
    output_dir: str,
    local_strategy_library_path: str,
    checkpoint_paths: list[str],
    candidate_generator: str,
    min_viable_candidate_count: int,
    force: bool,
) -> dict[str, object]:
    report = run_hypothesis_learning_constraint_smoke(
        trace_path=Path(trace_path),
        output_dir=Path(output_dir),
        local_strategy_library_path=Path(local_strategy_library_path),
        checkpoint_paths=[Path(path) for path in checkpoint_paths],
        candidate_generator=candidate_generator,
        min_viable_candidate_count=min_viable_candidate_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hypothesis-learning-constraint-smoke",
        state_hint=LEARNING_CONSTRAINT_SMOKE_STATE_HINT,
        artifacts={
            **_artifact_json_aliases(
                dict(report["artifacts"])
                if isinstance(report.get("artifacts"), dict)
                else {},
                {
                    "learning_constraint_smoke_json": "smoke_report_path",
                    "hypothesis_kpi_report_json": "hypothesis_kpi_report_path",
                    "candidate_plan_json": "candidate_plan_path",
                    "expansion_handoff_report_json": (
                        "expansion_handoff_report_path"
                    ),
                },
            ),
            **(
                report.get("artifacts")
                if isinstance(report.get("artifacts"), dict)
                else {}
            ),
        },
        metrics={
            **report["metrics"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )


def run_hermes_runtime_memory_bridge_smoke_command(
    *,
    trace_path: str,
    output_dir: str,
    initial_submission_dir: str,
    task1_test_path: str,
    task1_target_path: str,
    audit_repair_tasks_path: str,
    candidate_generator: str,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_runtime_memory_bridge_smoke(
        trace_path=Path(trace_path),
        output_dir=Path(output_dir),
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=Path(task1_test_path),
        task1_target_path=Path(task1_target_path),
        audit_repair_tasks_path=Path(audit_repair_tasks_path),
        candidate_generator=candidate_generator,
        force=force,
    )
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    return success_payload(
        run_id="local",
        command="run-hermes-runtime-memory-bridge-smoke",
        state_hint=HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_STATE_HINT,
        artifacts={
            **_artifact_json_aliases(
                dict(artifacts),
                {
                    "runtime_memory_bridge_smoke_json": "smoke_report_path",
                    "runtime_loop_report_json": "runtime_loop_report_path",
                    "runtime_closed_loop_audit_json": (
                        "runtime_closed_loop_audit_path"
                    ),
                    "heartbeat_report_json": "heartbeat_report_path",
                },
            ),
            **artifacts,
            "memory_bridge": report.get("memory_bridge", {}),
        },
        metrics={
            **(
                report.get("metrics")
                if isinstance(report.get("metrics"), dict)
                else {}
            ),
            "decision": report.get("decision"),
        },
        next_hint=str(report.get("next_command") or "use-hermes-runtime-closed-loop"),
    )


def run_hermes_auto_repair_memory_acceptance_smoke_command(
    *,
    output_dir: str,
    trace_path: str,
    initial_submission_dir: str,
    task1_test_path: str,
    task1_target_path: str,
    candidate_generator: str,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_auto_repair_memory_acceptance_smoke(
        output_dir=Path(output_dir),
        trace_path=Path(trace_path),
        initial_submission_dir=Path(initial_submission_dir),
        task1_test_path=Path(task1_test_path),
        task1_target_path=Path(task1_target_path),
        candidate_generator=candidate_generator,
        force=force,
    )
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    return success_payload(
        run_id="local",
        command="run-hermes-auto-repair-memory-acceptance-smoke",
        state_hint=HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_STATE_HINT,
        artifacts={
            **_artifact_json_aliases(
                dict(artifacts),
                {
                    "auto_repair_memory_acceptance_smoke_json": (
                        "smoke_report_path"
                    ),
                    "runtime_audit_chain_report_json": "chain_report_path",
                    "candidate_plan_json": "candidate_plan_path",
                },
            ),
            **artifacts,
            **checks,
        },
        metrics=report.get("metrics", {}),
        next_hint=str(report.get("next_command") or "run-hermes-runtime-audit-chain"),
    )


def audit_hermes_hypothesis_lab_infrastructure_command(
    *,
    acceptance_smoke_path: str,
    preflight_chain_report_path: str,
    runbook_path: str,
    output_path: str,
    min_completed_iterations: int,
    force: bool,
    stability_chain_report_path: str | None = None,
    candidate_pressure_chain_report_path: str | None = None,
) -> dict[str, object]:
    report = audit_hermes_hypothesis_lab_infrastructure(
        acceptance_smoke_path=Path(acceptance_smoke_path),
        preflight_chain_report_path=Path(preflight_chain_report_path),
        runbook_path=Path(runbook_path),
        output_path=Path(output_path),
        min_completed_iterations=min_completed_iterations,
        stability_chain_report_path=Path(stability_chain_report_path)
        if stability_chain_report_path is not None
        else None,
        candidate_pressure_chain_report_path=Path(candidate_pressure_chain_report_path)
        if candidate_pressure_chain_report_path is not None
        else None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="audit-hermes-hypothesis-lab-infrastructure",
        state_hint=HERMES_HYPOTHESIS_LAB_INFRASTRUCTURE_AUDIT_STATE_HINT,
        artifacts={
            "infrastructure_audit_json": report["audit_report_path"],
            "infrastructure_audit_path": report["audit_report_path"],
            **report["artifacts"],
            "checks": report["checks"],
            "errors": report["errors"],
        },
        metrics={
            "decision": report["decision"],
            "check_count": report["summary"]["check_count"],
            "failed_check_count": report["summary"]["failed_check_count"],
        },
        next_hint=report["next_command"],
    )


def resolve_audit_repair_task_command(
    *,
    audit_repair_tasks_path: str,
    output_path: str,
    task_id: str | None,
    provided_inputs: list[str] | None = None,
    force: bool,
) -> dict[str, object]:
    report = resolve_audit_repair_task(
        audit_repair_tasks_path=Path(audit_repair_tasks_path),
        output_path=Path(output_path),
        task_id=task_id,
        provided_inputs=_parse_key_value_inputs(provided_inputs or []),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="resolve-audit-repair-task",
        state_hint=REPAIR_TASK_RESOLUTION_STATE_HINT,
        artifacts={
            "repair_task_resolution_json": report["repair_task_resolution_path"],
            "audit_repair_tasks_path": report["audit_repair_tasks_path"],
            "task_id": report.get("task_id"),
            "audit_error": report.get("audit_error"),
            "next_command": report["next_command"],
            "argv": report["argv"],
            "command_string": report["command_string"],
            "missing_inputs": report["missing_inputs"],
            "provided_inputs": report["provided_inputs"],
        },
        metrics={
            "ready_to_run": report["ready_to_run"],
            "missing_input_count": len(report["missing_inputs"]),
            "argument_count": len(report["argv"]),
        },
        next_hint=report["next_hint"],
    )


def run_resolved_audit_repair_task_command(
    *,
    audit_repair_tasks_path: str,
    output_dir: str,
    task_id: str | None,
    provided_inputs: list[str] | None = None,
    force: bool,
) -> dict[str, object]:
    report = run_resolved_audit_repair_task(
        audit_repair_tasks_path=Path(audit_repair_tasks_path),
        output_dir=Path(output_dir),
        task_id=task_id,
        provided_inputs=_parse_key_value_inputs(provided_inputs or []),
        force=force,
    )
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    return success_payload(
        run_id="local",
        command="run-resolved-audit-repair-task",
        state_hint=REPAIR_TASK_RUN_STATE_HINT,
        artifacts={
            "repair_task_run_report_json": artifacts.get(
                "repair_task_run_report_path",
            ),
            "repair_task_run_report_path": artifacts.get(
                "repair_task_run_report_path",
            ),
            "repair_task_resolution_json": artifacts.get(
                "repair_task_resolution_path",
            ),
            "repair_task_resolution_path": artifacts.get(
                "repair_task_resolution_path",
            ),
            "runtime_loop_report_json": artifacts.get("runtime_loop_report_path"),
            "runtime_loop_report_path": artifacts.get("runtime_loop_report_path"),
            "audit_report_json": artifacts.get("audit_report_path"),
            "audit_report_path": artifacts.get("audit_report_path"),
            "runtime_memory_bridge_smoke_json": artifacts.get("smoke_report_path"),
            "runtime_memory_bridge_smoke_path": artifacts.get("smoke_report_path"),
            "decision": report.get("decision"),
            "repair_task_resolution": report.get("repair_task_resolution", {}),
        },
        metrics=report.get("metrics", {}),
        next_hint=str(report.get("next_command") or "run-hermes-runtime-audit-chain"),
    )


def run_hermes_runtime_closed_loop_command(
    *,
    output_dir: str,
    trace_path: str,
    initial_submission_dir: str,
    candidate_generator: str,
    candidate_generator_fallback: str,
    max_batches: int,
    recent_window: int = 50,
    repeated_family_threshold: int = 3,
    failed_family_threshold: int = 2,
    min_score_delta: float = 0.1,
    max_rounds_per_batch: int = 1,
    batch_deadline_seconds: float = 300.0,
    submit_score_threshold: float = 120.0,
    learning_objective: str = "score",
    task1_test_path: str | None = None,
    task1_target_path: str | None = None,
    task1_reference_path: str | None = None,
    task2_test_path: str | None = None,
    task2_target_path: str | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    evidence_accumulation_plan_path: str | None = None,
    expansion_registry_summary_path: str | None = None,
    expansion_handoff_report_path: str | None = None,
    audit_repair_tasks_path: str | None = None,
    input_research_chronicle_path: str | None = None,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_runtime_closed_loop(
        output_dir=Path(output_dir),
        trace_path=Path(trace_path),
        initial_submission_dir=Path(initial_submission_dir),
        candidate_generator=candidate_generator,
        candidate_generator_fallback=candidate_generator_fallback,
        max_batches=max_batches,
        recent_window=recent_window,
        repeated_family_threshold=repeated_family_threshold,
        failed_family_threshold=failed_family_threshold,
        min_score_delta=min_score_delta,
        max_rounds_per_batch=max_rounds_per_batch,
        batch_deadline_seconds=batch_deadline_seconds,
        submit_score_threshold=submit_score_threshold,
        learning_objective=learning_objective,
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
        evidence_accumulation_plan_path=Path(evidence_accumulation_plan_path)
        if evidence_accumulation_plan_path is not None
        else None,
        expansion_registry_summary_path=Path(expansion_registry_summary_path)
        if expansion_registry_summary_path is not None
        else None,
        expansion_handoff_report_path=Path(expansion_handoff_report_path)
        if expansion_handoff_report_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        input_research_chronicle_path=Path(input_research_chronicle_path)
        if input_research_chronicle_path is not None
        else None,
        force=force,
    )
    candidate_plan = (
        report.get("candidate_plan") if isinstance(report.get("candidate_plan"), dict) else {}
    )
    report_artifacts = (
        dict(report["artifacts"]) if isinstance(report.get("artifacts"), dict) else {}
    )
    return success_payload(
        run_id="local",
        command="run-hermes-runtime-closed-loop",
        state_hint=HERMES_RUNTIME_CLOSED_LOOP_STATE_HINT,
        artifacts={
            **_artifact_json_aliases(
                report_artifacts,
                {
                    "trace_summary_json": "trace_summary_path",
                    "hypothesis_kpi_report_json": "hypothesis_kpi_report_path",
                    "repair_policy_summary_json": "repair_policy_summary_path",
                    "doom_loop_report_json": "doom_loop_report_path",
                    "candidate_plan_json": "candidate_plan_path",
                    "heartbeat_report_json": "heartbeat_report_path",
                },
            ),
            **report_artifacts,
            "audit_command": report.get("audit_command", {}),
            "repair_nonconvergence_errors": candidate_plan.get(
                "repair_nonconvergence_errors",
                [],
            ),
            "repair_convergence_errors": candidate_plan.get(
                "repair_convergence_errors",
                [],
            ),
            "audit_repair_converged_error_counts": (
                candidate_plan.get("audit_repair_converged_error_counts", {})
                or (
                    candidate_plan.get("learning_constraints", {}).get(
                        "audit_repair_converged_error_counts",
                        {},
                    )
                    if isinstance(candidate_plan.get("learning_constraints"), dict)
                    else {}
                )
            ),
            "audit_repair_policies": candidate_plan.get(
                "audit_repair_policies",
                [],
            ),
            "repair_policy_recommended_actions": (
                candidate_plan.get("learning_constraints", {}).get(
                    "repair_policy_recommended_actions",
                    [],
                )
                if isinstance(candidate_plan.get("learning_constraints"), dict)
                else []
            ),
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def audit_hermes_runtime_closed_loop_command(
    *,
    runtime_report_path: str,
    output_path: str,
    require_guard_effect: bool,
    require_learning_constraint_smoke: bool,
    require_runtime_memory_bridge_smoke: bool,
    force: bool,
) -> dict[str, object]:
    report = audit_hermes_runtime_closed_loop(
        runtime_report_path=Path(runtime_report_path),
        output_path=Path(output_path),
        require_guard_effect=require_guard_effect,
        require_learning_constraint_smoke=require_learning_constraint_smoke,
        require_runtime_memory_bridge_smoke=require_runtime_memory_bridge_smoke,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="audit-hermes-runtime-closed-loop",
        state_hint=HERMES_RUNTIME_CLOSED_LOOP_AUDIT_STATE_HINT,
        artifacts={
            "runtime_closed_loop_audit_json": report["audit_report_path"],
            "runtime_closed_loop_audit_path": report["audit_report_path"],
            "audit_errors": report.get("errors", []),
            "errors": report.get("errors", []),
            "failed_checks": [
                check
                for check in report.get("checks", [])
                if isinstance(check, dict) and check.get("passed") is False
            ],
        },
        metrics={
            "decision": report["decision"],
            "check_count": report["summary"]["check_count"],
            "failed_check_count": report["summary"]["failed_check_count"],
        },
        next_hint=report["next_command"],
    )


def run_hermes_runtime_audit_chain_command(
    *,
    output_dir: str,
    trace_path: str,
    initial_submission_dir: str,
    candidate_generator: str,
    candidate_generator_fallback: str,
    iterations: int,
    max_batches: int,
    recent_window: int,
    repeated_family_threshold: int,
    failed_family_threshold: int,
    min_score_delta: float,
    max_rounds_per_batch: int,
    batch_deadline_seconds: float,
    submit_score_threshold: float,
    learning_objective: str,
    task1_test_path: str | None,
    task1_target_path: str | None,
    task1_reference_path: str | None,
    task2_test_path: str | None,
    task2_target_path: str | None,
    timing_profile: str | None,
    timing_factor: float | None,
    audit_repair_tasks_path: str | None,
    hypothesis_candidate_options_json: str | None = None,
    research_directive_path: str | None = None,
    auto_repair: bool = False,
    force: bool = False,
    preflight_acceptance_smoke_path: str | None = None,
    preflight_infrastructure_audit_path: str | None = None,
) -> dict[str, object]:
    hypothesis_candidate_options = (
        _parse_json_object(
            hypothesis_candidate_options_json,
            "hypothesis_candidate_options_json",
        )
        if hypothesis_candidate_options_json is not None
        else None
    )
    report = run_hermes_runtime_audit_chain(
        output_dir=Path(output_dir),
        trace_path=Path(trace_path),
        initial_submission_dir=Path(initial_submission_dir),
        candidate_generator=candidate_generator,
        candidate_generator_fallback=candidate_generator_fallback,
        iterations=iterations,
        max_batches=max_batches,
        recent_window=recent_window,
        repeated_family_threshold=repeated_family_threshold,
        failed_family_threshold=failed_family_threshold,
        min_score_delta=min_score_delta,
        max_rounds_per_batch=max_rounds_per_batch,
        batch_deadline_seconds=batch_deadline_seconds,
        submit_score_threshold=submit_score_threshold,
        learning_objective=learning_objective,
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
        hypothesis_candidate_options=hypothesis_candidate_options,
        research_directive_path=Path(research_directive_path)
        if research_directive_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        preflight_acceptance_smoke_path=Path(preflight_acceptance_smoke_path)
        if preflight_acceptance_smoke_path is not None
        else None,
        preflight_infrastructure_audit_path=Path(preflight_infrastructure_audit_path)
        if preflight_infrastructure_audit_path is not None
        else None,
        auto_repair=auto_repair,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-runtime-audit-chain",
        state_hint=HERMES_RUNTIME_AUDIT_CHAIN_STATE_HINT,
        artifacts={
            "runtime_audit_chain_report_json": report["chain_report_path"],
            "chain_report_path": report["chain_report_path"],
            "hypothesis_lab_trace_event": report.get("trace_event_id"),
            "trace_event_id": report.get("trace_event_id"),
            "repair_convergence": report.get("repair_convergence"),
            "auto_repair": report.get("auto_repair", {}),
            **(
                {
                    "auto_repair_report_path": report["artifacts"][
                        "auto_repair_report_path"
                    ],
                    "auto_repair_report_json": report["artifacts"][
                        "auto_repair_report_path"
                    ],
                }
                if isinstance(report.get("artifacts"), dict)
                and isinstance(report["artifacts"].get("auto_repair_report_path"), str)
                else {}
            ),
            **(
                {
                    "audit_repair_tasks_json": report["artifacts"][
                        "audit_repair_tasks_path"
                    ]
                }
                if isinstance(report.get("artifacts"), dict)
                and isinstance(
                    report["artifacts"].get("audit_repair_tasks_path"),
                    str,
                )
                else {}
            ),
            **(
                report.get("artifacts")
                if isinstance(report.get("artifacts"), dict)
                else {}
            ),
        },
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )


def _parse_json_object(payload_json: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _parse_key_value_inputs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"provided input must use key=value syntax: {value}")
        key, item = value.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError("provided input key must not be empty")
        parsed[key] = item
    return parsed


def _artifact_json_aliases(
    artifacts: dict[str, object],
    aliases: dict[str, str],
) -> dict[str, object]:
    return {
        alias: artifacts[source]
        for alias, source in aliases.items()
        if isinstance(artifacts.get(source), str) and artifacts[source]
    }
