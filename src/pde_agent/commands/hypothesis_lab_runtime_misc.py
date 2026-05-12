"""Command functions for misc domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pde_agent.core.hypothesis_doom_loop import (
    STATE_HINT as DOOM_LOOP_STATE_HINT,
)
from pde_agent.core.availability_audit import (
    STATE_HINT as AGENT_RUNTIME_AVAILABILITY_STATE_HINT,
)
from pde_agent.core.availability_audit import audit_agent_runtime_availability
from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
from pde_agent.core.json_output import success_payload
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
    include_compatibility_tools: bool = False,
    force: bool,
) -> dict[str, object]:
    catalog = build_scientific_tool_catalog(
        output_path=Path(output_path),
        include_compatibility_tools=include_compatibility_tools,
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
        state_hint=AGENT_RUNTIME_AVAILABILITY_STATE_HINT,
        artifacts={
            "availability_audit_path": output_path,
            "tool_catalog_path": tool_catalog_path,
        },
        metrics=report["metrics"],
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

def materialize_training_windows_for_hypothesis_sandbox_command(
    *,
    windows_path: str,
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    from pde_agent.core.task1_short_window_training import (
        materialize_training_windows_for_hypothesis_sandbox,
    )

    report = materialize_training_windows_for_hypothesis_sandbox(
        windows_path=Path(windows_path),
        output_dir=Path(output_dir),
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="materialize-training-windows-for-hypothesis-sandbox",
        state_hint="HYPOTHESIS_SANDBOX_WINDOW_BUNDLE_MATERIALIZED",
        artifacts={
            "context_path": report["artifacts"]["context_path"],
            "target_path": report["artifacts"]["target_path"],
            "reference_path": report["artifacts"]["reference_path"],
            "manifest_path": report["artifacts"]["manifest_path"],
        },
        metrics={},
        next_hint=report["next_command"],
    )

def convert_hypothesis_algorithm_evidence_for_candidate_promotion_command(
    *,
    hypothesis_evidence_path: str,
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    from pde_agent.core.task1_short_window_training import (
        convert_hypothesis_algorithm_evidence_for_candidate_promotion,
    )

    report = convert_hypothesis_algorithm_evidence_for_candidate_promotion(
        hypothesis_evidence_path=Path(hypothesis_evidence_path),
        output_dir=Path(output_dir),
        force=force,
    )
    return success_payload(
        run_id=Path(output_dir).name or "local",
        command="convert-hypothesis-algorithm-evidence-for-candidate-promotion",
        state_hint="HYPOTHESIS_EVIDENCE_PROMOTION_BRIDGE_BUILT",
        artifacts={
            "algorithm_candidates_path": report["artifacts"]["algorithm_candidates_path"],
            "sandbox_evidence_path": report["artifacts"]["sandbox_evidence_path"],
            "manifest_path": report["artifacts"]["manifest_path"],
        },
        metrics={},
        next_hint=report["next_command"],
    )

def _parse_json_objects(*, values: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} must be valid JSON object")
        rows.append(parsed)
    return rows

def _parse_json_object(payload_json: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload

def _hypothesis_lab_policy_constraints(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("state_hint") != "HYPOTHESIS_LAB_NEXT_POLICY_BUILT":
        raise ValueError(
            "hypothesis_lab_next_policy_path must point to "
            "HYPOTHESIS_LAB_NEXT_POLICY_BUILT"
        )
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        return {}
    candidate_generator = policy.get("candidate_generator")
    preferred_error_modes = policy.get("preferred_error_modes")
    constraints: dict[str, Any] = {
        "source_policy_path": path,
        "next_action": policy.get("next_action"),
        "learning_objective": policy.get("learning_objective"),
    }
    if isinstance(candidate_generator, str) and candidate_generator:
        constraints["candidate_generator"] = candidate_generator
        if policy.get("next_action") == "retry_with_policy_constraints":
            constraints["guard_override_candidate_generator"] = candidate_generator
    if isinstance(preferred_error_modes, list):
        constraints["preferred_error_modes"] = [
            item for item in preferred_error_modes if isinstance(item, str) and item
        ]
    required = policy.get("required_constraints")
    if isinstance(required, dict):
        constraints["required_constraints"] = required
    options = policy.get("hypothesis_candidate_options")
    if isinstance(options, dict):
        constraints["hypothesis_candidate_options"] = options
    return constraints

def _dedupe_strings(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, str) and value and value not in result:
            result.append(value)
    return result
