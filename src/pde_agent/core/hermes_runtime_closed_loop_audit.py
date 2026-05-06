"""Audit gate for Hermes runtime closed-loop reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_heartbeat import get_candidate_generator_registry


STATE_HINT = "HERMES_RUNTIME_CLOSED_LOOP_AUDITED"
RUNTIME_STATE_HINT = "HERMES_RUNTIME_CLOSED_LOOP_RAN"

REQUIRED_ARTIFACT_STATE_HINTS = {
    "tool_catalog_path": "SCIENTIFIC_TOOL_CATALOG_BUILT",
    "trace_summary_path": "HYPOTHESIS_LAB_TRACE_SUMMARIZED",
    "hypothesis_kpi_report_path": "HYPOTHESIS_KPI_REPORT_BUILT",
    "doom_loop_report_path": "HYPOTHESIS_DOOM_LOOP_CHECKED",
    "candidate_plan_path": "HYPOTHESIS_CANDIDATE_PLAN_BUILT",
    "heartbeat_report_path": "HERMES_HEARTBEAT_RAN",
}
OPTIONAL_ARTIFACT_STATE_HINTS = {
    "repair_policy_summary_path": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
}


class HermesRuntimeClosedLoopAuditError(RuntimeError):
    pass


def audit_hermes_runtime_closed_loop(
    *,
    runtime_report_path: str | Path,
    output_path: str | Path,
    require_guard_effect: bool = False,
    require_learning_constraint_smoke: bool = False,
    require_runtime_memory_bridge_smoke: bool = False,
    force: bool,
) -> dict[str, Any]:
    runtime_file = Path(runtime_report_path)
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise HermesRuntimeClosedLoopAuditError(f"audit report exists: {output_file}")
    runtime = _read_json(runtime_file, "runtime_report")
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _add_check(
        checks,
        errors,
        "runtime_state_hint",
        runtime.get("state_hint") == RUNTIME_STATE_HINT,
        {"actual": runtime.get("state_hint"), "expected": RUNTIME_STATE_HINT},
    )
    artifacts = runtime.get("artifacts") if isinstance(runtime.get("artifacts"), dict) else {}
    _add_check(
        checks,
        errors,
        "required_artifact_keys",
        _required_artifact_keys_present(artifacts),
        {"required_keys": _required_artifact_keys(), "actual_keys": sorted(artifacts)},
    )
    loaded_artifacts = _load_artifacts(artifacts)
    required_loaded_artifacts = {
        key: value
        for key, value in loaded_artifacts.items()
        if key in _required_artifact_keys()
    }
    _add_check(
        checks,
        errors,
        "artifact_files_exist",
        all(item["exists"] for item in required_loaded_artifacts.values()),
        {
            "artifacts": loaded_artifacts,
            "required_artifacts": required_loaded_artifacts,
        },
    )
    _add_check(
        checks,
        errors,
        "artifact_state_hints",
        _artifact_state_hints_ok(loaded_artifacts),
        {
            "expected": {
                **REQUIRED_ARTIFACT_STATE_HINTS,
                **{
                    key: value
                    for key, value in OPTIONAL_ARTIFACT_STATE_HINTS.items()
                    if key in loaded_artifacts
                },
            },
            "actual": {
                key: item.get("state_hint")
                for key, item in loaded_artifacts.items()
                if key in REQUIRED_ARTIFACT_STATE_HINTS
                or key in OPTIONAL_ARTIFACT_STATE_HINTS
            },
        },
    )
    trace_ok, latest_event = _trace_has_runtime_event(artifacts.get("trace_path"))
    _add_check(
        checks,
        errors,
        "trace_runtime_event",
        trace_ok,
        {"latest_runtime_event": latest_event},
    )
    attempt_sync_ok, attempt_sync_details = _candidate_attempt_trace_sync(
        runtime=runtime,
        trace_path=artifacts.get("trace_path"),
    )
    _add_check(
        checks,
        errors,
        "candidate_attempt_trace_sync",
        attempt_sync_ok,
        attempt_sync_details,
    )
    guard_ok = _guard_consistency_ok(runtime, loaded_artifacts)
    _add_check(
        checks,
        errors,
        "guard_consistency",
        guard_ok,
        {
            "runtime_guard": runtime.get("guard"),
            "doom_guard": loaded_artifacts.get("doom_loop_report_path", {}).get(
                "payload", {}
            ),
        },
    )
    if require_guard_effect:
        _add_check(
            checks,
            errors,
            "guard_effect_observed",
            _guard_effect_observed(loaded_artifacts),
            {
                "heartbeat": loaded_artifacts.get("heartbeat_report_path", {}).get(
                    "payload", {}
                )
            },
        )
    if require_learning_constraint_smoke:
        smoke_ok, smoke_details = _learning_constraint_smoke_gate(artifacts)
        _add_check(
            checks,
            errors,
            "learning_constraint_smoke_gate",
            smoke_ok,
            smoke_details,
        )
    if require_runtime_memory_bridge_smoke:
        memory_smoke_ok, memory_smoke_details = _runtime_memory_bridge_smoke_gate(
            artifacts
        )
        _add_check(
            checks,
            errors,
            "runtime_memory_bridge_smoke_gate",
            memory_smoke_ok,
            memory_smoke_details,
        )
    enforcement_ok, enforcement_details = _learning_constraint_enforcement_gate(
        loaded_artifacts
    )
    _add_check(
        checks,
        errors,
        "learning_constraint_enforcement",
        enforcement_ok,
        enforcement_details,
    )
    registry_ok, registry_details = _candidate_generator_registry_policy(
        loaded_artifacts
    )
    _add_check(
        checks,
        errors,
        "candidate_generator_registry_policy",
        registry_ok,
        registry_details,
    )
    heartbeat_saturation_ok, heartbeat_saturation_details = (
        _heartbeat_saturation_policy_gate(loaded_artifacts)
    )
    _add_check(
        checks,
        errors,
        "heartbeat_saturation_policy",
        heartbeat_saturation_ok,
        heartbeat_saturation_details,
    )
    saturation_ok, saturation_details = _cluster_saturation_consistency_gate(
        runtime=runtime,
        loaded_artifacts=loaded_artifacts,
    )
    _add_check(
        checks,
        errors,
        "cluster_saturation_consistency",
        saturation_ok,
        saturation_details,
    )
    memory_bridge_ok, memory_bridge_details = _memory_bridge_inputs_gate(
        runtime=runtime,
        loaded_artifacts=loaded_artifacts,
    )
    _add_check(
        checks,
        errors,
        "memory_bridge_inputs",
        memory_bridge_ok,
        memory_bridge_details,
    )
    search_guidance_ok, search_guidance_details = (
        _persistent_search_guidance_consistency_gate(
            runtime=runtime,
            trace_path=artifacts.get("trace_path"),
            loaded_artifacts=loaded_artifacts,
        )
    )
    _add_check(
        checks,
        errors,
        "persistent_search_guidance_consistency",
        search_guidance_ok,
        search_guidance_details,
    )
    repair_nonconvergence_ok, repair_nonconvergence_details = (
        _repair_nonconvergence_consistency_gate(
            runtime=runtime,
            trace_path=artifacts.get("trace_path"),
            loaded_artifacts=loaded_artifacts,
        )
    )
    _add_check(
        checks,
        errors,
        "repair_nonconvergence_consistency",
        repair_nonconvergence_ok,
        repair_nonconvergence_details,
    )
    repair_convergence_ok, repair_convergence_details = (
        _repair_convergence_consistency_gate(
            runtime=runtime,
            trace_path=artifacts.get("trace_path"),
            loaded_artifacts=loaded_artifacts,
        )
    )
    _add_check(
        checks,
        errors,
        "repair_convergence_consistency",
        repair_convergence_ok,
        repair_convergence_details,
    )
    repair_policy_ok, repair_policy_details = _repair_policy_summary_consistency_gate(
        runtime=runtime,
        trace_path=artifacts.get("trace_path"),
        loaded_artifacts=loaded_artifacts,
    )
    _add_check(
        checks,
        errors,
        "repair_policy_summary_consistency",
        repair_policy_ok,
        repair_policy_details,
    )
    memory_constraints_ok, memory_constraints_details = (
        _repair_policy_memory_constraints_consistency_gate(
            loaded_artifacts=loaded_artifacts,
        )
    )
    _add_check(
        checks,
        errors,
        "repair_policy_memory_constraints_consistency",
        memory_constraints_ok,
        memory_constraints_details,
    )
    chronicle_ok, chronicle_details = _research_chronicle_repair_policy_gate(
        artifacts=artifacts
    )
    _add_check(
        checks,
        errors,
        "research_chronicle_repair_policy_memory",
        chronicle_ok,
        chronicle_details,
    )
    bounded_ok, bounded_details = _bounded_executor_authoring_gate(runtime)
    _add_check(
        checks,
        errors,
        "bounded_executor_authoring_gate",
        bounded_ok,
        bounded_details,
    )

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "audit_report_path": str(output_file),
        "runtime_report_path": str(runtime_file),
        "decision": "passed" if not errors else "failed",
        "summary": {
            "artifact_count": len(artifacts),
            "check_count": len(checks),
            "failed_check_count": len(errors),
        },
        "checks": checks,
        "errors": errors,
        "next_command": "run-hermes-runtime-closed-loop"
        if errors
        else "use-hermes-runtime-closed-loop",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _required_artifact_keys() -> list[str]:
    return sorted(
        {
            "runtime_loop_report_path",
            "trace_path",
            *REQUIRED_ARTIFACT_STATE_HINTS.keys(),
        }
    )


def _required_artifact_keys_present(artifacts: dict[str, Any]) -> bool:
    return all(key in artifacts for key in _required_artifact_keys())


def _load_artifacts(artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for key, value in artifacts.items():
        if not isinstance(value, str):
            loaded[key] = {"path": value, "exists": False, "payload": {}}
            continue
        path = Path(value)
        payload = {}
        if path.exists() and path.is_file() and path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        loaded[key] = {
            "path": str(path),
            "exists": path.exists(),
            "state_hint": payload.get("state_hint") if isinstance(payload, dict) else None,
            "payload": payload if isinstance(payload, dict) else {},
        }
    return loaded


def _artifact_state_hints_ok(loaded_artifacts: dict[str, dict[str, Any]]) -> bool:
    for key, expected in REQUIRED_ARTIFACT_STATE_HINTS.items():
        if loaded_artifacts.get(key, {}).get("state_hint") != expected:
            return False
    for key, expected in OPTIONAL_ARTIFACT_STATE_HINTS.items():
        if key not in loaded_artifacts:
            continue
        if loaded_artifacts.get(key, {}).get("state_hint") != expected:
            return False
    return True


def _trace_has_runtime_event(trace_path: Any) -> tuple[bool, dict[str, Any]]:
    if not isinstance(trace_path, str):
        return False, {}
    path = Path(trace_path)
    if not path.exists():
        return False, {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event_type") == (
            "hermes_runtime_closed_loop"
        ):
            latest = event
    return bool(latest), latest


def _candidate_attempt_trace_sync(
    *,
    runtime: dict[str, Any],
    trace_path: Any,
) -> tuple[bool, dict[str, Any]]:
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    expected = metrics.get("recorded_candidate_attempt_count", 0)
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        expected = 0
    expected_count = int(expected)
    actual_count = _count_candidate_attempt_events(trace_path)
    return actual_count >= expected_count, {
        "expected_min_candidate_attempt_events": expected_count,
        "actual_candidate_attempt_events": actual_count,
    }


def _count_candidate_attempt_events(trace_path: Any) -> int:
    if not isinstance(trace_path, str):
        return 0
    path = Path(trace_path)
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        tags = event.get("tags")
        if (
            event.get("event_type") == "hypothesis_lab_run"
            and isinstance(tags, list)
            and "candidate_attempt" in tags
        ):
            count += 1
    return count


def _guard_consistency_ok(
    runtime: dict[str, Any],
    loaded_artifacts: dict[str, dict[str, Any]],
) -> bool:
    runtime_guard = runtime.get("guard") if isinstance(runtime.get("guard"), dict) else {}
    doom = loaded_artifacts.get("doom_loop_report_path", {}).get("payload", {})
    if not isinstance(doom, dict):
        return False
    return (
        bool(runtime_guard.get("guard_active")) == bool(doom.get("guard_active"))
        and list(runtime_guard.get("blocked_candidate_families") or [])
        == list(doom.get("blocked_candidate_families") or [])
    )


def _guard_effect_observed(loaded_artifacts: dict[str, dict[str, Any]]) -> bool:
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    if not isinstance(heartbeat, dict):
        return False
    guard = heartbeat.get("guard")
    batches = heartbeat.get("batches")
    parameters = heartbeat.get("parameters") if isinstance(heartbeat.get("parameters"), dict) else {}
    blocked = set(guard.get("blocked_candidate_families") or []) if isinstance(guard, dict) else set()
    requested = parameters.get("candidate_generator")
    return (
        isinstance(guard, dict)
        and guard.get("guard_active") is True
        and isinstance(batches, list)
        and any(
            isinstance(batch, dict)
            and (
                batch.get("guarded_candidate_generator") is True
                or (
                    isinstance(requested, str)
                    and requested in blocked
                    and batch.get("candidate_generator") != requested
                    and batch.get("candidate_generator") not in blocked
                )
            )
            for batch in batches
        )
    )


def _learning_constraint_smoke_gate(
    artifacts: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    path_value = artifacts.get("learning_constraint_smoke_report_path")
    runtime_trace_path = artifacts.get("trace_path")
    details: dict[str, Any] = {
        "learning_constraint_smoke_report_path": path_value,
        "runtime_trace_path": runtime_trace_path,
    }
    if not isinstance(path_value, str) or not path_value.strip():
        details["reason"] = "missing_learning_constraint_smoke_report_path"
        return False, details
    path = Path(path_value)
    payload = _read_optional_json(path)
    smoke_trace_path = payload.get("trace_path")
    trace_matches = (
        isinstance(runtime_trace_path, str)
        and isinstance(smoke_trace_path, str)
        and smoke_trace_path == runtime_trace_path
    )
    details.update(
        {
            "exists": path.is_file(),
            "state_hint": payload.get("state_hint"),
            "decision": payload.get("decision"),
            "trace_path": smoke_trace_path,
            "trace_matches": trace_matches,
        }
    )
    smoke_passed = (
        path.is_file()
        and payload.get("state_hint") == "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"
        and payload.get("decision") == "passed"
    )
    ok = smoke_passed and trace_matches
    if not ok:
        details["reason"] = (
            "learning_constraint_smoke_trace_mismatch"
            if smoke_passed and not trace_matches
            else "learning_constraint_smoke_not_passed"
        )
    return ok, details


def _runtime_memory_bridge_smoke_gate(
    artifacts: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    path_value = artifacts.get("runtime_memory_bridge_smoke_report_path")
    runtime_path = artifacts.get("runtime_loop_report_path")
    details: dict[str, Any] = {
        "runtime_memory_bridge_smoke_report_path": path_value,
        "runtime_loop_report_path": runtime_path,
    }
    if not isinstance(path_value, str) or not path_value.strip():
        details["reason"] = "missing_runtime_memory_bridge_smoke_report_path"
        return False, details
    path = Path(path_value)
    payload = _read_optional_json(path)
    payload_artifacts = (
        payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    )
    smoke_runtime_path = payload_artifacts.get("runtime_loop_report_path")
    memory_bridge = (
        payload.get("memory_bridge")
        if isinstance(payload.get("memory_bridge"), dict)
        else {}
    )
    runtime_matches = (
        isinstance(runtime_path, str)
        and isinstance(smoke_runtime_path, str)
        and smoke_runtime_path == runtime_path
    )
    memory_bridge_passed = memory_bridge.get("passed") is True
    details.update(
        {
            "exists": path.is_file(),
            "state_hint": payload.get("state_hint"),
            "decision": payload.get("decision"),
            "smoke_runtime_loop_report_path": smoke_runtime_path,
            "runtime_matches": runtime_matches,
            "memory_bridge_passed": memory_bridge_passed,
        }
    )
    smoke_passed = (
        path.is_file()
        and payload.get("state_hint") == "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN"
        and payload.get("decision") == "passed"
        and memory_bridge_passed
    )
    ok = smoke_passed and runtime_matches
    if not ok:
        details["reason"] = (
            "runtime_memory_bridge_smoke_runtime_mismatch"
            if smoke_passed and not runtime_matches
            else "runtime_memory_bridge_smoke_not_passed"
        )
    return ok, details


def _learning_constraint_enforcement_gate(
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    candidate_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    constraints = (
        candidate_plan.get("learning_constraints")
        if isinstance(candidate_plan, dict)
        else {}
    )
    avoid = {
        item
        for item in constraints.get("avoid_candidate_families", [])
        if isinstance(item, str) and item
    } if isinstance(constraints, dict) else set()
    violations: list[dict[str, Any]] = []
    if isinstance(candidate_plan, dict):
        schedule = candidate_plan.get("candidate_generator_schedule")
        if isinstance(schedule, list):
            violations.extend(
                {
                    "location": "candidate_plan_schedule",
                    "candidate_generator": item,
                }
                for item in schedule
                if isinstance(item, str) and item in avoid
            )
    if isinstance(heartbeat, dict):
        batches = heartbeat.get("batches")
        if isinstance(batches, list):
            for batch in batches:
                if not isinstance(batch, dict):
                    continue
                generator = batch.get("candidate_generator")
                if isinstance(generator, str) and generator in avoid:
                    violations.append(
                        {
                            "location": "heartbeat_batch",
                            "batch_index": batch.get("batch_index"),
                            "candidate_generator": generator,
                        }
                    )
    return not violations, {
        "avoid_candidate_families": sorted(avoid),
        "violations": violations,
    }


def _candidate_generator_registry_policy(
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    if not isinstance(heartbeat, dict):
        return False, {"reason": "missing_heartbeat_payload"}
    batches = heartbeat.get("batches")
    if not isinstance(batches, list):
        return True, {"checked_batch_count": 0, "violations": []}
    registry = get_candidate_generator_registry()
    violations: list[dict[str, Any]] = []
    checked = 0
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict) or batch.get("guarded_candidate_generator") is not True:
            continue
        checked += 1
        generator = batch.get("candidate_generator")
        if generator == "none":
            continue
        expected = registry.get(generator) if isinstance(generator, str) else None
        actual = batch.get("candidate_generator_metadata")
        if expected is None:
            violations.append(
                {
                    "batch_index": index,
                    "candidate_generator": generator,
                    "reason": "candidate_generator_not_in_registry",
                }
            )
            continue
        if expected.get("auto_fallback_allowed") is not True:
            violations.append(
                {
                    "batch_index": index,
                    "candidate_generator": generator,
                    "reason": "auto_fallback_not_allowed",
                    "registry_metadata": expected,
                }
            )
        if actual != expected:
            violations.append(
                {
                    "batch_index": index,
                    "candidate_generator": generator,
                    "reason": "metadata_mismatch",
                    "actual_metadata": actual,
                    "registry_metadata": expected,
                }
            )
    return not violations, {
        "checked_guarded_batch_count": checked,
        "violations": violations,
    }


def _heartbeat_saturation_policy_gate(
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    if not isinstance(heartbeat, dict):
        return False, {"reason": "missing_heartbeat_payload"}
    parameters = (
        heartbeat.get("parameters")
        if isinstance(heartbeat.get("parameters"), dict)
        else {}
    )
    saturated = _string_list(parameters.get("saturated_candidate_families"))
    probe_interval = _int_metric(parameters.get("saturation_probe_interval_batches"))
    details: dict[str, Any] = {
        "saturated_candidate_families": saturated,
        "saturation_probe_interval_batches": probe_interval,
        "violations": [],
    }
    if not saturated:
        return True, details
    batches = heartbeat.get("batches")
    if not isinstance(batches, list):
        return True, details
    registry = get_candidate_generator_registry()
    violations: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            continue
        generator = batch.get("candidate_generator")
        if not isinstance(generator, str) or not generator:
            continue
        if _generator_saturation_blocked(
            generator,
            saturated_candidate_families=saturated,
            registry=registry,
        ):
            if _valid_saturation_probe(
                batch=batch,
                fallback_batch_index=index + 1,
                saturation_probe_interval_batches=probe_interval,
            ):
                continue
            violations.append(
                {
                    "batch_index": index,
                    "candidate_generator": generator,
                    "original_candidate_generator": batch.get(
                        "original_candidate_generator"
                    ),
                    "reason": "candidate_generator_blocked_by_saturation",
                }
            )
    details["violations"] = violations
    return not violations, details


def _valid_saturation_probe(
    *,
    batch: dict[str, Any],
    fallback_batch_index: int,
    saturation_probe_interval_batches: int,
) -> bool:
    if saturation_probe_interval_batches <= 0:
        return False
    if batch.get("saturation_probe") is not True:
        return False
    if batch.get("saturation_probe_reason") != (
        "interval_reprobe_of_saturated_candidate_family"
    ):
        return False
    batch_index = batch.get("batch_index")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        batch_index = fallback_batch_index
    return batch_index % saturation_probe_interval_batches == 0


def _generator_saturation_blocked(
    candidate_generator: str,
    *,
    saturated_candidate_families: list[str],
    registry: dict[str, dict[str, Any]],
) -> bool:
    if candidate_generator in saturated_candidate_families:
        return True
    aliases = registry.get(candidate_generator, {}).get("saturation_blocked_by")
    if not isinstance(aliases, list):
        return False
    saturated = set(saturated_candidate_families)
    return any(isinstance(alias, str) and alias in saturated for alias in aliases)


def _cluster_saturation_consistency_gate(
    *,
    runtime: dict[str, Any],
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    runtime_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    artifact_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    if not isinstance(artifact_plan, dict):
        artifact_plan = {}

    metric_count = _int_metric(metrics.get("cluster_saturation_signal_count"))
    metric_families = _string_list(metrics.get("saturated_candidate_families"))
    runtime_signals = _dict_list(runtime_plan.get("cluster_saturation_signals"))
    artifact_signals = _dict_list(artifact_plan.get("cluster_saturation_signals"))
    runtime_families = _saturated_candidate_families(runtime_plan)
    artifact_families = _saturated_candidate_families(artifact_plan)
    signal_required = bool(
        metric_count
        or metric_families
        or runtime_signals
        or artifact_signals
        or runtime_families
        or artifact_families
    )
    details = {
        "metric_count": metric_count,
        "metric_saturated_candidate_families": metric_families,
        "runtime_signal_count": len(runtime_signals),
        "artifact_signal_count": len(artifact_signals),
        "runtime_saturated_candidate_families": runtime_families,
        "artifact_saturated_candidate_families": artifact_families,
        "violations": [],
    }
    if not signal_required:
        return True, details

    violations: list[dict[str, Any]] = []
    if metric_count != len(runtime_signals) or metric_count != len(artifact_signals):
        violations.append(
            {
                "reason": "cluster_saturation_signal_count_mismatch",
                "metric_count": metric_count,
                "runtime_signal_count": len(runtime_signals),
                "artifact_signal_count": len(artifact_signals),
            }
        )
    if sorted(metric_families) != sorted(runtime_families):
        violations.append(
            {
                "reason": "runtime_saturated_candidate_families_mismatch",
                "metric_families": metric_families,
                "runtime_families": runtime_families,
            }
        )
    if sorted(metric_families) != sorted(artifact_families):
        violations.append(
            {
                "reason": "artifact_saturated_candidate_families_mismatch",
                "metric_families": metric_families,
                "artifact_families": artifact_families,
            }
        )
    if runtime_signals != artifact_signals:
        violations.append(
            {
                "reason": "cluster_saturation_signal_payload_mismatch",
                "runtime_signals": runtime_signals,
                "artifact_signals": artifact_signals,
            }
        )
    details["violations"] = violations
    return not violations, details


def _saturated_candidate_families(plan: dict[str, Any]) -> list[str]:
    constraints = (
        plan.get("learning_constraints")
        if isinstance(plan.get("learning_constraints"), dict)
        else {}
    )
    return _string_list(constraints.get("saturated_candidate_families"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _memory_bridge_inputs_gate(
    *,
    runtime: dict[str, Any],
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    artifacts = runtime.get("artifacts") if isinstance(runtime.get("artifacts"), dict) else {}
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    handoff_path = artifacts.get("input_expansion_handoff_report_path")
    summary_path = artifacts.get("input_expansion_registry_summary_path")
    details: dict[str, Any] = {
        "input_expansion_handoff_report_path": handoff_path,
        "input_expansion_registry_summary_path": summary_path,
        "required": bool(handoff_path or summary_path),
        "violations": [],
    }
    if not handoff_path and not summary_path:
        return True, details
    violations: list[dict[str, Any]] = []
    if handoff_path:
        handoff = loaded_artifacts.get("input_expansion_handoff_report_path", {})
        if (
            handoff.get("exists") is not True
            or handoff.get("state_hint") != "HERMES_CANDIDATE_EXPANSION_HANDOFF_RAN"
        ):
            violations.append(
                {
                    "artifact": "input_expansion_handoff_report_path",
                    "reason": "missing_or_invalid_handoff_report",
                    "state_hint": handoff.get("state_hint"),
                }
            )
        if metrics.get("expansion_handoff_report_loaded") is not True:
            violations.append(
                {
                    "artifact": "input_expansion_handoff_report_path",
                    "reason": "runtime_metric_not_loaded",
                }
            )
        candidate_plan = loaded_artifacts.get("candidate_plan_path", {}).get(
            "payload",
            {},
        )
        prior = (
            candidate_plan.get("prior_expansion_exhaustion")
            if isinstance(candidate_plan, dict)
            else {}
        )
        handoff_payload = handoff.get("payload", {})
        expected = (
            handoff_payload.get("expansion_exhaustion")
            if isinstance(handoff_payload, dict)
            else {}
        )
        if isinstance(expected, dict) and expected.get("exhausted") is True:
            if not isinstance(prior, dict) or prior.get("exhausted") is not True:
                violations.append(
                    {
                        "artifact": "candidate_plan_path",
                        "reason": "prior_expansion_exhaustion_not_carried",
                    }
                )
    if summary_path:
        summary = loaded_artifacts.get("input_expansion_registry_summary_path", {})
        if (
            summary.get("exists") is not True
            or summary.get("state_hint") != "HERMES_LITERATURE_REGISTRY_SUMMARY_BUILT"
        ):
            violations.append(
                {
                    "artifact": "input_expansion_registry_summary_path",
                    "reason": "missing_or_invalid_registry_summary",
                    "state_hint": summary.get("state_hint"),
                }
            )
        if metrics.get("expansion_registry_summary_loaded") is not True:
            violations.append(
                {
                    "artifact": "input_expansion_registry_summary_path",
                    "reason": "runtime_metric_not_loaded",
                }
            )
    details["violations"] = violations
    return not violations, details


def _bounded_executor_authoring_gate(
    runtime: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    artifacts = runtime.get("artifacts") if isinstance(runtime.get("artifacts"), dict) else {}
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    next_command = runtime.get("next_command")
    task_count = _int_metric(metrics.get("blocked_executor_authoring_task_count"))
    if next_command != "agent-author-bounded-executor" and task_count == 0:
        return True, {"required": False}
    tasks_path = artifacts.get("blocked_executor_tasks_path")
    audit_paths = artifacts.get("blocked_executor_authoring_audit_paths")
    completion_paths = artifacts.get("blocked_executor_completion_task_paths")
    details: dict[str, Any] = {
        "required": True,
        "blocked_executor_tasks_path": tasks_path,
        "audit_paths": audit_paths,
        "completion_task_paths": completion_paths,
    }
    if not isinstance(tasks_path, str) or not Path(tasks_path).is_file():
        details["reason"] = "missing_blocked_executor_tasks_path"
        return False, details
    if not isinstance(audit_paths, list) or not audit_paths:
        details["reason"] = "missing_authoring_audit_paths"
        return False, details
    audits = []
    for path_value in audit_paths:
        if not isinstance(path_value, str) or not path_value.strip():
            audits.append({"path": path_value, "exists": False, "decision": None})
            continue
        path = Path(path_value)
        payload = _read_optional_json(path)
        audits.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "state_hint": payload.get("state_hint"),
                "authoring_task_path": payload.get("authoring_task_path"),
                "decision": payload.get("decision"),
                "issue_count": payload.get("issue_count"),
            }
        )
    details["audits"] = audits
    approved = [
        audit
        for audit in audits
        if audit.get("exists") is True
        and audit.get("state_hint") == "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED"
        and audit.get("decision") == "approved"
        and _int_metric(audit.get("issue_count")) == 0
    ]
    expected_approved = _int_metric(
        metrics.get("blocked_executor_authoring_approved_count")
    )
    if expected_approved <= 0 and next_command == "agent-author-bounded-executor":
        details["reason"] = "agent_author_requested_without_approved_count"
        return False, details
    if len(approved) < max(1, expected_approved):
        details["reason"] = "approved_authoring_audits_missing"
        return False, details
    completion_ok, completion_details = _bounded_executor_completion_tasks_ok(
        completion_paths,
        expected_count=max(1, expected_approved),
        expected_authoring_paths={
            str(audit.get("authoring_task_path"))
            for audit in approved
            if isinstance(audit.get("authoring_task_path"), str)
        },
        expected_trace_path=artifacts.get("trace_path"),
    )
    details["completion_tasks"] = completion_details
    if not completion_ok:
        details["reason"] = "completion_task_missing_or_invalid"
        return False, details
    return True, details


def _persistent_search_guidance_consistency_gate(
    *,
    runtime: dict[str, Any],
    trace_path: Any,
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    runtime_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    runtime_constraints = (
        runtime_plan.get("learning_constraints")
        if isinstance(runtime_plan.get("learning_constraints"), dict)
        else {}
    )
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    kpi = loaded_artifacts.get("hypothesis_kpi_report_path", {}).get("payload", {})
    artifact_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    artifact_constraints = (
        artifact_plan.get("learning_constraints")
        if isinstance(artifact_plan, dict)
        and isinstance(artifact_plan.get("learning_constraints"), dict)
        else {}
    )
    trace_payload = _latest_runtime_trace_payload(trace_path)
    output_sources = {
        "heartbeat_report_path": _dict_list(
            heartbeat.get("persistent_search_guidance")
            if isinstance(heartbeat, dict)
            else []
        ),
        "trace_runtime_event": _dict_list(
            trace_payload.get("persistent_search_guidance")
            if isinstance(trace_payload, dict)
            else []
        ),
    }
    input_sources = {
        "candidate_plan_path": _dict_list(
            artifact_constraints.get("persistent_search_guidance")
            if isinstance(artifact_constraints, dict)
            else []
        ),
        "runtime_report": _dict_list(
            runtime_constraints.get("persistent_search_guidance")
            if isinstance(runtime_constraints, dict)
            else []
        ),
        "hypothesis_kpi_report_path": _dict_list(
            kpi.get("persistent_search_guidance") if isinstance(kpi, dict) else []
        ),
    }
    canonical_output_sources = {
        source: _canonical_guidance_list(items)
        for source, items in output_sources.items()
    }
    canonical_input_sources = {
        source: _canonical_guidance_list(items) for source, items in input_sources.items()
    }
    output_expected = _first_non_empty_guidance(canonical_output_sources)
    input_expected = _first_non_empty_guidance(canonical_input_sources)
    violations: list[dict[str, Any]] = []
    if output_expected:
        for source, canonical in canonical_output_sources.items():
            if canonical != output_expected:
                violations.append(
                    {
                        "reason": "persistent_search_guidance_payload_mismatch",
                        "source": source,
                    }
                )
    if input_expected:
        for source, canonical in canonical_input_sources.items():
            if canonical != input_expected:
                violations.append(
                    {
                        "reason": "persistent_search_guidance_payload_mismatch",
                        "source": source,
                    }
                )
        output_set = set(output_expected)
        missing_from_output = sorted(
            item for item in input_expected if item not in output_set
        )
        if missing_from_output:
            violations.append(
                {
                    "reason": "persistent_search_guidance_input_not_carried_to_output",
                    "source": "heartbeat_report_path",
                    "missing_count": len(missing_from_output),
                }
            )
    output_count = len(output_expected)
    input_count = len(input_expected)
    count_sources = {
        "runtime_metrics": _int_metric(
            (
                runtime.get("metrics")
                if isinstance(runtime.get("metrics"), dict)
                else {}
            ).get("persistent_search_guidance_count")
        ),
        "heartbeat_parameters": _int_metric(
            (
                heartbeat.get("parameters")
                if isinstance(heartbeat, dict)
                and isinstance(heartbeat.get("parameters"), dict)
                else {}
            ).get("persistent_search_guidance_count")
        ),
    }
    input_count_sources = {
        "kpi_metrics": _int_metric(
            (
                kpi.get("metrics")
                if isinstance(kpi, dict) and isinstance(kpi.get("metrics"), dict)
                else {}
            ).get("persistent_search_guidance_count")
        ),
    }
    for source, count in count_sources.items():
        if count != output_count:
            violations.append(
                {
                    "reason": "persistent_search_guidance_count_mismatch",
                    "source": source,
                    "actual_count": count,
                    "expected_count": output_count,
                }
            )
    for source, count in input_count_sources.items():
        if count != input_count:
            violations.append(
                {
                    "reason": "persistent_search_guidance_count_mismatch",
                    "source": source,
                    "actual_count": count,
                    "expected_count": input_count,
                }
            )
    return not violations, {
        "output_expected_count": output_count,
        "input_expected_count": input_count,
        "output_source_counts": {
            source: len(items) for source, items in output_sources.items()
        },
        "input_source_counts": {
            source: len(items) for source, items in input_sources.items()
        },
        "output_metric_counts": count_sources,
        "input_metric_counts": input_count_sources,
        "violations": violations,
    }


def _repair_nonconvergence_consistency_gate(
    *,
    runtime: dict[str, Any],
    trace_path: Any,
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    runtime_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    runtime_constraints = (
        runtime_plan.get("learning_constraints")
        if isinstance(runtime_plan.get("learning_constraints"), dict)
        else {}
    )
    artifact_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    if not isinstance(artifact_plan, dict):
        artifact_plan = {}
    artifact_constraints = (
        artifact_plan.get("learning_constraints")
        if isinstance(artifact_plan.get("learning_constraints"), dict)
        else {}
    )
    kpi = loaded_artifacts.get("hypothesis_kpi_report_path", {}).get("payload", {})
    trace_payload = _latest_runtime_trace_payload(trace_path)

    sources = {
        "runtime_metrics": _string_list(metrics.get("repair_nonconvergence_errors")),
        "runtime_report": _string_list(
            runtime_plan.get("repair_nonconvergence_errors")
        )
        or _string_list(runtime_constraints.get("repair_nonconvergence_errors")),
        "candidate_plan_path": _string_list(
            artifact_plan.get("repair_nonconvergence_errors")
        )
        or _string_list(artifact_constraints.get("repair_nonconvergence_errors")),
        "hypothesis_kpi_report_path": _string_list(
            kpi.get("repair_nonconvergence_errors") if isinstance(kpi, dict) else []
        ),
        "trace_runtime_event": _string_list(
            trace_payload.get("repair_nonconvergence_errors")
            if isinstance(trace_payload, dict)
            else []
        ),
    }
    expected = _first_non_empty_string_list(sources)
    count_sources = {
        "runtime_metrics": _int_metric(
            metrics.get("repair_nonconvergence_error_count")
        ),
        "kpi_metrics": _int_metric(
            (
                kpi.get("metrics")
                if isinstance(kpi, dict) and isinstance(kpi.get("metrics"), dict)
                else {}
            ).get("repair_nonconvergence_error_count")
        ),
    }
    violations: list[dict[str, Any]] = []
    if expected:
        for source, values in sources.items():
            if sorted(values) != expected:
                violations.append(
                    {
                        "reason": "repair_nonconvergence_errors_mismatch",
                        "source": source,
                        "actual_errors": sorted(values),
                        "expected_errors": expected,
                    }
                )
    expected_count = len(expected)
    for source, count in count_sources.items():
        if count != expected_count:
            violations.append(
                {
                    "reason": "repair_nonconvergence_error_count_mismatch",
                    "source": source,
                    "actual_count": count,
                    "expected_count": expected_count,
                }
            )
    return not violations, {
        "expected_errors": expected,
        "source_errors": {source: sorted(values) for source, values in sources.items()},
        "count_sources": count_sources,
        "violations": violations,
    }


def _repair_convergence_consistency_gate(
    *,
    runtime: dict[str, Any],
    trace_path: Any,
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    runtime_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    runtime_constraints = (
        runtime_plan.get("learning_constraints")
        if isinstance(runtime_plan.get("learning_constraints"), dict)
        else {}
    )
    artifact_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    if not isinstance(artifact_plan, dict):
        artifact_plan = {}
    artifact_constraints = (
        artifact_plan.get("learning_constraints")
        if isinstance(artifact_plan.get("learning_constraints"), dict)
        else {}
    )
    kpi = loaded_artifacts.get("hypothesis_kpi_report_path", {}).get("payload", {})
    trace_payload = _latest_runtime_trace_payload(trace_path)

    sources = {
        "runtime_metrics": _string_list(metrics.get("repair_convergence_errors")),
        "runtime_report": _string_list(runtime_plan.get("repair_convergence_errors"))
        or _string_list(runtime_constraints.get("repair_convergence_errors")),
        "candidate_plan_path": _string_list(
            artifact_plan.get("repair_convergence_errors")
        )
        or _string_list(artifact_constraints.get("repair_convergence_errors")),
        "hypothesis_kpi_report_path": _string_list(
            kpi.get("repair_convergence_errors") if isinstance(kpi, dict) else []
        ),
        "trace_runtime_event": _string_list(
            trace_payload.get("repair_convergence_errors")
            if isinstance(trace_payload, dict)
            else []
        ),
    }
    expected = _first_non_empty_string_list(sources)
    count_sources = {
        "runtime_metrics": _int_metric(metrics.get("repair_convergence_error_count")),
        "kpi_metrics": _int_metric(
            (
                kpi.get("metrics")
                if isinstance(kpi, dict) and isinstance(kpi.get("metrics"), dict)
                else {}
            ).get("repair_convergence_error_count")
        ),
    }
    violations: list[dict[str, Any]] = []
    if expected:
        for source, values in sources.items():
            if sorted(values) != expected:
                violations.append(
                    {
                        "reason": "repair_convergence_errors_mismatch",
                        "source": source,
                        "actual_errors": sorted(values),
                        "expected_errors": expected,
                    }
                )
    expected_count = len(expected)
    for source, count in count_sources.items():
        if count != expected_count:
            violations.append(
                {
                    "reason": "repair_convergence_error_count_mismatch",
                    "source": source,
                    "actual_count": count,
                    "expected_count": expected_count,
                }
            )
    return not violations, {
        "expected_errors": expected,
        "source_errors": {source: sorted(values) for source, values in sources.items()},
        "count_sources": count_sources,
        "violations": violations,
    }


def _repair_policy_summary_consistency_gate(
    *,
    runtime: dict[str, Any],
    trace_path: Any,
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    repair_policy = loaded_artifacts.get("repair_policy_summary_path", {}).get(
        "payload",
        {},
    )
    if not isinstance(repair_policy, dict) or not repair_policy:
        return True, {
            "policy_present": False,
            "expected_action_count": 0,
            "violations": [],
        }
    metrics = runtime.get("metrics") if isinstance(runtime.get("metrics"), dict) else {}
    runtime_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    runtime_constraints = (
        runtime_plan.get("learning_constraints")
        if isinstance(runtime_plan.get("learning_constraints"), dict)
        else {}
    )
    artifact_plan = loaded_artifacts.get("candidate_plan_path", {}).get("payload", {})
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    artifact_constraints = (
        artifact_plan.get("learning_constraints")
        if isinstance(artifact_plan, dict)
        and isinstance(artifact_plan.get("learning_constraints"), dict)
        else {}
    )
    trace_payload = _latest_runtime_trace_payload(trace_path)
    sources = {
        "repair_policy_summary_path": _dict_list(
            repair_policy.get("recommended_actions")
        ),
        "candidate_plan_path": _dict_list(
            artifact_constraints.get("repair_policy_recommended_actions")
        ),
        "runtime_report": _dict_list(
            runtime_constraints.get("repair_policy_recommended_actions")
        ),
        "heartbeat_report_path": _dict_list(
            heartbeat.get("repair_policy_recommended_actions")
            if isinstance(heartbeat, dict)
            else []
        ),
        "trace_runtime_event": _dict_list(
            trace_payload.get("repair_policy_recommended_actions")
            if isinstance(trace_payload, dict)
            else []
        ),
    }
    canonical_sources = {
        source: _canonical_guidance_list(items) for source, items in sources.items()
    }
    expected = canonical_sources["repair_policy_summary_path"]
    violations: list[dict[str, Any]] = []
    if expected:
        for source, canonical in canonical_sources.items():
            if canonical != expected:
                violations.append(
                    {
                        "reason": "repair_policy_recommended_actions_mismatch",
                        "source": source,
                        "actual_count": len(canonical),
                        "expected_count": len(expected),
                    }
                )
    count_sources = {
        "runtime_metrics": _int_metric(
            metrics.get("repair_policy_recommended_action_count")
        ),
        "heartbeat_parameters": _int_metric(
            (
                heartbeat.get("parameters")
                if isinstance(heartbeat, dict)
                and isinstance(heartbeat.get("parameters"), dict)
                else {}
            ).get("repair_policy_recommended_action_count")
        ),
        "repair_policy_metrics": _int_metric(
            (
                repair_policy.get("metrics")
                if isinstance(repair_policy.get("metrics"), dict)
                else {}
            ).get("recommended_action_count")
        ),
    }
    expected_count = len(expected)
    for source, count in count_sources.items():
        if count != expected_count:
            violations.append(
                {
                    "reason": "repair_policy_recommended_action_count_mismatch",
                    "source": source,
                    "actual_count": count,
                    "expected_count": expected_count,
                }
            )
    if expected and isinstance(heartbeat, dict):
        for index, batch in enumerate(_dict_list(heartbeat.get("batches")), start=1):
            canonical = _canonical_guidance_list(
                _dict_list(batch.get("repair_policy_recommended_actions"))
            )
            if canonical != expected:
                violations.append(
                    {
                        "reason": "repair_policy_batch_actions_mismatch",
                        "source": "heartbeat_report_path",
                        "batch_index": index,
                        "actual_count": len(canonical),
                        "expected_count": expected_count,
                    }
                )
            batch_count = _int_metric(
                batch.get("repair_policy_recommended_action_count")
            )
            if batch_count != expected_count:
                violations.append(
                    {
                        "reason": "repair_policy_batch_action_count_mismatch",
                        "source": "heartbeat_report_path",
                        "batch_index": index,
                        "actual_count": batch_count,
                        "expected_count": expected_count,
                    }
                )
    return not violations, {
        "policy_present": True,
        "expected_action_count": expected_count,
        "source_counts": {source: len(items) for source, items in sources.items()},
        "count_sources": count_sources,
        "violations": violations,
    }


def _repair_policy_memory_constraints_consistency_gate(
    *,
    loaded_artifacts: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    repair_policy = loaded_artifacts.get("repair_policy_summary_path", {}).get(
        "payload",
        {},
    )
    expected = _canonical_guidance_list(
        _dict_list(
            repair_policy.get("recommended_actions")
            if isinstance(repair_policy, dict)
            else []
        )
    )
    expected_count = len(expected)
    heartbeat = loaded_artifacts.get("heartbeat_report_path", {}).get("payload", {})
    batches = (
        _dict_list(heartbeat.get("batches")) if isinstance(heartbeat, dict) else []
    )
    violations: list[dict[str, Any]] = []
    if expected:
        for index, batch in enumerate(batches, start=1):
            diagnostics = (
                batch.get("candidate_diagnostics")
                if isinstance(batch.get("candidate_diagnostics"), dict)
                else {}
            )
            constraints = (
                diagnostics.get("memory_constraints")
                if isinstance(diagnostics.get("memory_constraints"), dict)
                else {}
            )
            if not constraints:
                violations.append(
                    {
                        "reason": "missing_batch_memory_constraints",
                        "source": "heartbeat_report_path",
                        "batch_index": index,
                        "expected_count": expected_count,
                    }
                )
                continue
            actual = _canonical_guidance_list(
                _dict_list(constraints.get("repair_policy_recommended_actions"))
            )
            if actual != expected:
                violations.append(
                    {
                        "reason": "repair_policy_memory_constraints_actions_mismatch",
                        "source": "heartbeat_report_path",
                        "batch_index": index,
                        "actual_count": len(actual),
                        "expected_count": expected_count,
                    }
                )
            actual_count = _int_metric(
                constraints.get("repair_policy_recommended_action_count")
            )
            if actual_count != expected_count:
                violations.append(
                    {
                        "reason": "repair_policy_memory_constraints_count_mismatch",
                        "source": "heartbeat_report_path",
                        "batch_index": index,
                        "actual_count": actual_count,
                        "expected_count": expected_count,
                    }
                )
    return not violations, {
        "policy_present": bool(expected),
        "expected_action_count": expected_count,
        "batch_count": len(batches),
        "violations": violations,
    }


def _research_chronicle_repair_policy_gate(
    *,
    artifacts: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    path_value = artifacts.get("research_chronicle_path")
    if path_value is None:
        return True, {"present": False, "reason": "research_chronicle_not_declared"}
    if not isinstance(path_value, str) or not path_value.strip():
        return False, {"present": True, "reason": "invalid_research_chronicle_path"}
    path = Path(path_value)
    if not path.is_file():
        return False, {
            "present": True,
            "path": str(path),
            "reason": "research_chronicle_missing",
        }
    text = path.read_text(encoding="utf-8")
    required_markers = ["Hermes Research Chronicle", "Repair Policy Memory:"]
    missing_markers = [marker for marker in required_markers if marker not in text]
    return not missing_markers, {
        "present": True,
        "path": str(path),
        "required_markers": required_markers,
        "missing_markers": missing_markers,
    }


def _first_non_empty_string_list(sources: dict[str, list[str]]) -> list[str]:
    for values in sources.values():
        if values:
            return sorted(values)
    return []


def _latest_runtime_trace_payload(trace_path: Any) -> dict[str, Any]:
    ok, event = _trace_has_runtime_event(trace_path)
    if not ok:
        return {}
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _canonical_guidance_list(items: list[dict[str, Any]]) -> list[str]:
    return sorted(
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for item in items
    )


def _first_non_empty_guidance(
    canonical_sources: dict[str, list[str]],
) -> list[str]:
    for items in canonical_sources.values():
        if items:
            return items
    return []


def _bounded_executor_completion_tasks_ok(
    completion_paths: Any,
    *,
    expected_count: int,
    expected_authoring_paths: set[str],
    expected_trace_path: Any,
) -> tuple[bool, list[dict[str, Any]]]:
    if not isinstance(completion_paths, list) or not completion_paths:
        return False, []
    tasks = []
    valid_count = 0
    for path_value in completion_paths:
        if not isinstance(path_value, str) or not path_value.strip():
            tasks.append({"path": path_value, "exists": False, "valid": False})
            continue
        path = Path(path_value)
        payload = _read_optional_json(path)
        command = payload.get("command")
        authoring_task_path = payload.get("authoring_task_path")
        trace_path = payload.get("trace_path")
        authoring_matches = (
            not expected_authoring_paths
            or (
                isinstance(authoring_task_path, str)
                and authoring_task_path in expected_authoring_paths
            )
        )
        trace_matches = (
            isinstance(expected_trace_path, str)
            and isinstance(trace_path, str)
            and trace_path == expected_trace_path
            and isinstance(command, str)
            and f"--trace-path {expected_trace_path}" in command
        )
        valid = (
            path.is_file()
            and payload.get("state_hint") == "BOUNDED_EXECUTOR_COMPLETION_TASK_PREPARED"
            and payload.get("next_command") == "complete-bounded-executor-authoring-task"
            and isinstance(command, str)
            and "complete-bounded-executor-authoring-task" in command
            and authoring_matches
            and trace_matches
        )
        if valid:
            valid_count += 1
        tasks.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "state_hint": payload.get("state_hint"),
                "authoring_task_path": authoring_task_path,
                "authoring_matches": authoring_matches,
                "trace_path": trace_path,
                "trace_matches": trace_matches,
                "next_command": payload.get("next_command"),
                "valid": valid,
            }
        )
    return valid_count >= expected_count, tasks


def _int_metric(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append({"check_id": check_id, "passed": passed, "details": details})
    if not passed:
        errors.append(check_id)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise HermesRuntimeClosedLoopAuditError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeClosedLoopAuditError(f"{label} invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesRuntimeClosedLoopAuditError(f"{label} must be a JSON object")
    return payload
