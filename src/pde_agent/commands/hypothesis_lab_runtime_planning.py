"""Command functions for planning domain."""

from __future__ import annotations

import json
from pathlib import Path

from pde_agent.commands.hypothesis_lab_runtime_misc import (
    _dedupe_strings,
    _hypothesis_lab_policy_constraints,
)
from pde_agent.core.hypothesis_candidate_planner import (
    STATE_HINT as CANDIDATE_PLAN_STATE_HINT,
)
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_kpi import STATE_HINT as KPI_STATE_HINT
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
from pde_agent.core.json_output import success_payload

def plan_hypothesis_candidates_command(
    *,
    output_path: str,
    candidate_generator: str,
    hypothesis_lab_next_policy_path: str | None = None,
    doom_loop_report_path: str | None,
    hypothesis_kpi_report_path: str | None,
    repair_policy_summary_path: str | None = None,
    research_chronicle_path: str | None = None,
    expansion_handoff_report_path: str | None = None,
    expansion_report_path: str | None = None,
    audit_repair_tasks_path: str | None = None,
    preferred_error_modes: list[str] | None = None,
    max_schedule_length: int = 5,
    min_viable_candidate_count: int = 1,
    force: bool = False,
) -> dict[str, object]:
    policy_constraints = _hypothesis_lab_policy_constraints(
        hypothesis_lab_next_policy_path
    )
    if policy_constraints.get("candidate_generator"):
        candidate_generator = str(policy_constraints["candidate_generator"])
    merged_preferred_error_modes = _dedupe_strings(
        [
            *(preferred_error_modes or []),
            *policy_constraints.get("preferred_error_modes", []),
        ]
    )
    guard_overrides = []
    if policy_constraints.get("guard_override_candidate_generator"):
        guard_overrides.append(str(policy_constraints["guard_override_candidate_generator"]))
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
        expansion_report_path=Path(expansion_report_path)
        if expansion_report_path is not None
        else None,
        audit_repair_tasks_path=Path(audit_repair_tasks_path)
        if audit_repair_tasks_path is not None
        else None,
        preferred_error_modes=merged_preferred_error_modes,
        guard_override_candidate_generators=guard_overrides,
        max_schedule_length=max_schedule_length,
        min_viable_candidate_count=min_viable_candidate_count,
        force=force,
    )
    if policy_constraints:
        report["hypothesis_lab_policy_constraints"] = policy_constraints
        if Path(output_path).exists():
            Path(output_path).write_text(
                json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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
            "hypothesis_lab_policy_constraints": report.get(
                "hypothesis_lab_policy_constraints",
                {},
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
