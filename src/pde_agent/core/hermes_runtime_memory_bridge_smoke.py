"""Runtime acceptance smoke for Hermes learning-memory bridges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_runtime_closed_loop import run_hermes_runtime_closed_loop
from pde_agent.core.hermes_runtime_closed_loop_audit import (
    audit_hermes_runtime_closed_loop,
)


STATE_HINT = "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN"


class HermesRuntimeMemoryBridgeSmokeError(RuntimeError):
    pass


def run_hermes_runtime_memory_bridge_smoke(
    *,
    trace_path: str | Path,
    output_dir: str | Path,
    initial_submission_dir: str | Path,
    task1_test_path: str | Path,
    task1_target_path: str | Path,
    audit_repair_tasks_path: str | Path,
    candidate_generator: str = "postprocess",
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    smoke_path = root / "hermes_runtime_memory_bridge_smoke.json"
    if smoke_path.exists() and not force:
        raise HermesRuntimeMemoryBridgeSmokeError(
            f"runtime memory bridge smoke exists: {smoke_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    runtime = run_hermes_runtime_closed_loop(
        output_dir=root / "runtime_loop",
        trace_path=Path(trace_path),
        initial_submission_dir=Path(initial_submission_dir),
        candidate_generator=candidate_generator,
        candidate_generator_fallback="none",
        max_batches=1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=30.0,
        task1_test_path=Path(task1_test_path),
        task1_target_path=Path(task1_target_path),
        audit_repair_tasks_path=Path(audit_repair_tasks_path),
        force=force,
    )
    audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=Path(runtime["artifacts"]["runtime_loop_report_path"]),
        output_path=root / "runtime_loop" / "hermes_runtime_closed_loop_audit.json",
        require_guard_effect=False,
        force=force,
    )
    heartbeat = _read_json(Path(runtime["artifacts"]["heartbeat_report_path"]))
    expected_actions = _repair_policy_actions(runtime)
    memory_constraints = _first_batch_memory_constraints(heartbeat)
    memory_actions = _dict_list(
        memory_constraints.get("repair_policy_recommended_actions")
    )
    expected = _canonical_items(expected_actions)
    actual = _canonical_items(memory_actions)
    memory_count = _int_metric(
        memory_constraints.get("repair_policy_recommended_action_count")
    )
    bridge_passed = (
        bool(expected)
        and expected == actual
        and memory_count == len(expected_actions)
    )
    passed = audit["decision"] == "passed" and bridge_passed
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "smoke_report_path": str(smoke_path),
        "decision": "passed" if passed else "failed",
        "artifacts": {
            "smoke_report_path": str(smoke_path),
            "runtime_loop_report_path": runtime["artifacts"][
                "runtime_loop_report_path"
            ],
            "runtime_closed_loop_audit_path": audit["audit_report_path"],
            "heartbeat_report_path": runtime["artifacts"]["heartbeat_report_path"],
            "hypothesis_kpi_report_path": runtime["artifacts"][
                "hypothesis_kpi_report_path"
            ],
            "repair_policy_summary_path": runtime["artifacts"][
                "repair_policy_summary_path"
            ],
        },
        "metrics": {
            "runtime_audit_failed_check_count": audit["summary"][
                "failed_check_count"
            ],
            "repair_policy_recommended_action_count": len(expected_actions),
            "batch_memory_constraint_action_count": len(memory_actions),
        },
        "memory_bridge": {
            "passed": bridge_passed,
            "expected_actions": expected_actions,
            "batch_memory_constraints": memory_constraints,
            "canonical_expected_actions": expected,
            "canonical_actual_actions": actual,
        },
        "runtime_audit": {
            "decision": audit["decision"],
            "errors": audit.get("errors", []),
        },
        "next_command": "use-hermes-runtime-closed-loop"
        if passed
        else "inspect-hermes-runtime-memory-bridge-smoke",
    }
    smoke_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _attach_smoke_to_runtime_report(
        runtime_report_path=Path(runtime["artifacts"]["runtime_loop_report_path"]),
        smoke_report_path=smoke_path,
    )
    return report


def _repair_policy_actions(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_plan = (
        runtime.get("candidate_plan")
        if isinstance(runtime.get("candidate_plan"), dict)
        else {}
    )
    constraints = (
        candidate_plan.get("learning_constraints")
        if isinstance(candidate_plan.get("learning_constraints"), dict)
        else {}
    )
    return _dict_list(constraints.get("repair_policy_recommended_actions"))


def _first_batch_memory_constraints(heartbeat: dict[str, Any]) -> dict[str, Any]:
    batches = _dict_list(heartbeat.get("batches"))
    if not batches:
        return {}
    diagnostics = (
        batches[0].get("candidate_diagnostics")
        if isinstance(batches[0].get("candidate_diagnostics"), dict)
        else {}
    )
    constraints = diagnostics.get("memory_constraints")
    return constraints if isinstance(constraints, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _canonical_items(items: list[dict[str, Any]]) -> list[str]:
    return sorted(
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for item in items
    )


def _int_metric(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesRuntimeMemoryBridgeSmokeError(f"artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesRuntimeMemoryBridgeSmokeError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesRuntimeMemoryBridgeSmokeError(f"artifact must be an object: {path}")
    return payload


def _attach_smoke_to_runtime_report(
    *,
    runtime_report_path: Path,
    smoke_report_path: Path,
) -> None:
    runtime = _read_json(runtime_report_path)
    artifacts = runtime.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise HermesRuntimeMemoryBridgeSmokeError(
            "runtime report artifacts must be an object"
        )
    artifacts["runtime_memory_bridge_smoke_report_path"] = str(smoke_report_path)
    audit_command = runtime.setdefault("audit_command", {})
    if not isinstance(audit_command, dict):
        raise HermesRuntimeMemoryBridgeSmokeError(
            "runtime report audit_command must be an object"
        )
    audit_command["require_runtime_memory_bridge_smoke"] = True
    runtime_report_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
