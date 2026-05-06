"""Repair policy summary artifacts for Hermes runtime loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_trace import read_hypothesis_lab_trace


STATE_HINT = "HERMES_REPAIR_POLICY_SUMMARY_BUILT"


class HermesRepairPolicySummaryError(RuntimeError):
    pass


def build_repair_policy_summary(
    *,
    trace_path: str | Path,
    output_path: str | Path,
    recent_window: int = 50,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    events = read_hypothesis_lab_trace(trace_path)
    recent_events = events[-max(1, recent_window) :]
    repair_events = [
        event
        for event in recent_events
        if event.get("event_type") == "hermes_runtime_audit_chain"
        and isinstance(event.get("payload"), dict)
        and isinstance(event["payload"].get("repair_convergence"), dict)
    ]

    stats: dict[str, dict[str, Any]] = {}
    for event in repair_events:
        convergence = event["payload"]["repair_convergence"]
        decision = _text(convergence.get("decision"))
        if decision == "converged":
            errors = _text_list(convergence.get("input_repair_errors"))
        elif decision == "nonconverged":
            errors = _text_list(convergence.get("repeated_audit_errors"))
            if not errors:
                errors = _text_list(convergence.get("input_repair_errors"))
        else:
            errors = (
                _text_list(convergence.get("repeated_audit_errors"))
                + _text_list(convergence.get("input_repair_errors"))
            )
        for error in errors:
            entry = stats.setdefault(
                error,
                {
                    "attempt_count": 0,
                    "converged_count": 0,
                    "nonconverged_count": 0,
                    "last_decision": None,
                },
            )
            entry["attempt_count"] += 1
            if decision == "converged":
                entry["converged_count"] += 1
            elif decision == "nonconverged":
                entry["nonconverged_count"] += 1
            entry["last_decision"] = decision

    converged_errors = sorted(
        error for error, entry in stats.items() if entry["converged_count"] > 0
    )
    nonconverged_errors = sorted(
        error for error, entry in stats.items() if entry["nonconverged_count"] > 0
    )
    recommended_actions = _recommended_actions(
        converged_errors=converged_errors,
        nonconverged_errors=nonconverged_errors,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "trace_path": str(trace_path),
        "repair_policy_summary_path": str(output_file),
        "recent_window": recent_window,
        "event_count": len(repair_events),
        "converged_repair_errors": converged_errors,
        "nonconverged_repair_errors": nonconverged_errors,
        "repair_error_stats": stats,
        "recommended_actions": recommended_actions,
        "metrics": {
            "event_count": len(repair_events),
            "repair_attempt_count": sum(
                entry["attempt_count"] for entry in stats.values()
            ),
            "converged_repair_error_count": len(converged_errors),
            "nonconverged_repair_error_count": len(nonconverged_errors),
            "recommended_action_count": len(recommended_actions),
        },
        "next_command": "plan-hypothesis-candidates",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _recommended_actions(
    *,
    converged_errors: list[str],
    nonconverged_errors: list[str],
) -> list[dict[str, str]]:
    actions = [
        {
            "audit_error": error,
            "action": "switch_repair_strategy_or_reload_missing_evidence",
            "reason": "repair_nonconverged",
        }
        for error in nonconverged_errors
    ]
    actions.extend(
        {
            "audit_error": error,
            "action": "prefer_previously_converged_repair_path",
            "reason": "repair_converged",
        }
        for error in converged_errors
        if error not in set(nonconverged_errors)
    )
    return actions


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise HermesRepairPolicySummaryError(f"output already exists: {path}")


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, str) and item})
