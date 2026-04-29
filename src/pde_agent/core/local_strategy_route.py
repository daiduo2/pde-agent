"""Route Hermes local strategy-loop artifacts to the next harness action."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROUTE_STATE_HINT = "LOCAL_STRATEGY_NEXT_ACTION_ROUTED"


class LocalStrategyRouteError(RuntimeError):
    pass


def route_local_strategy_next_action(
    *,
    artifact_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    artifact_file = Path(artifact_path)
    artifact = _read_json_object(artifact_file)
    route = _route(artifact)
    report = {
        "schema_version": 1,
        "state_hint": ROUTE_STATE_HINT,
        "artifact_path": str(artifact_file),
        "route_report_path": str(output_path),
        "source_state_hint": artifact.get("state_hint"),
        "route": route,
        "next_command": route["command"],
    }
    _write_json(Path(output_path), report)
    return report


def _route(artifact: dict[str, Any]) -> dict[str, Any]:
    state_hint = artifact.get("state_hint")
    if state_hint == "LOCAL_STRATEGY_LEADERBOARD_UPDATED":
        return _route_leaderboard(artifact)
    if state_hint == "LOCAL_STRATEGY_LOOP_AUDITED":
        return _route_audit(artifact)
    if state_hint == "SCHEDULED_EXPERIMENT_CLAIMED":
        return {
            "action": "execute_claimed_experiment",
            "command": str(artifact["next_command"]),
            "arguments": {
                "claim_path": artifact.get("claim_path"),
                "selected_experiment": artifact.get("selected_experiment"),
            },
            "requires_human_review": False,
            "reason_codes": ["scheduled_experiment_claimed"],
        }
    raise LocalStrategyRouteError(f"unsupported artifact state_hint: {state_hint!r}")


def _route_leaderboard(artifact: dict[str, Any]) -> dict[str, Any]:
    decision = artifact.get("decision")
    if not isinstance(decision, dict):
        raise LocalStrategyRouteError("leaderboard artifact must include decision object")
    decision_name = decision.get("decision")
    next_command = str(decision.get("next_command"))
    if decision_name == "accept_best_strategy":
        best = artifact.get("best_result")
        return {
            "action": "record_strategy_proposal",
            "command": next_command,
            "arguments": {"best_result": best},
            "requires_human_review": False,
            "reason_codes": ["accepted_local_strategy"],
        }
    if decision_name == "continue_scheduled_experiments":
        return {
            "action": "claim_next_scheduled_experiment",
            "command": next_command,
            "arguments": {
                "schedule_path": artifact.get("schedule_path"),
                "leaderboard_path": artifact.get("leaderboard_path"),
                "completed_experiment_ids": artifact.get(
                    "completed_experiment_ids",
                    [],
                ),
            },
            "requires_human_review": False,
            "reason_codes": ["pending_local_strategy_experiments"],
        }
    if decision_name == "local_strategy_queue_exhausted":
        return {
            "action": "build_research_chronicle",
            "command": next_command,
            "arguments": {
                "leaderboard_path": artifact.get("leaderboard_path"),
                "result_count": artifact.get("result_count"),
            },
            "requires_human_review": False,
            "reason_codes": ["local_strategy_queue_exhausted"],
        }
    raise LocalStrategyRouteError(f"unsupported leaderboard decision: {decision_name!r}")


def _route_audit(artifact: dict[str, Any]) -> dict[str, Any]:
    decision = artifact.get("decision")
    if decision == "passed":
        return {
            "action": "use_local_strategy_loop",
            "command": str(artifact.get("next_command", "use-local-strategy-loop")),
            "arguments": {
                "audit_report_path": artifact.get("audit_report_path"),
                "scenario_count": artifact.get("scenario_count"),
            },
            "requires_human_review": False,
            "reason_codes": ["local_strategy_loop_audit_passed"],
        }
    return {
        "action": "review_local_strategy_loop_audit",
        "command": "review-local-strategy-loop-audit",
        "arguments": {"audit_report_path": artifact.get("audit_report_path")},
        "requires_human_review": True,
        "reason_codes": ["local_strategy_loop_audit_failed"],
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LocalStrategyRouteError(f"missing artifact path: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalStrategyRouteError(f"invalid artifact JSON: {path}") from exc
    if not isinstance(data, dict):
        raise LocalStrategyRouteError("artifact JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
