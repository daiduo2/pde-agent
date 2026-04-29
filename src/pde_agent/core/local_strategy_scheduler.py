"""Schedule local strategy evaluations from Hermes strategy-library lookups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEDULE_STATE_HINT = "LOCAL_STRATEGY_EVALUATION_SCHEDULED"
CLAIM_STATE_HINT = "SCHEDULED_EXPERIMENT_CLAIMED"


class LocalStrategySchedulerError(RuntimeError):
    pass


def schedule_local_strategy_evaluation(
    *,
    lookup_report_path: str | Path,
    output_path: str | Path,
    workspace_root: str | Path,
    max_experiments: int,
    max_train_seconds: float,
    max_inference_seconds: float,
    force: bool,
) -> dict[str, Any]:
    if max_experiments < 1:
        raise LocalStrategySchedulerError("max_experiments must be positive")
    lookup_file = _require_file(Path(lookup_report_path), "local strategy lookup report")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise LocalStrategySchedulerError(f"schedule already exists: {output_file}")

    lookup = _read_json_object(lookup_file, "local strategy lookup report")
    _validate_lookup(lookup)
    workspace_dir = Path(workspace_root)
    matches = lookup["matches"][:max_experiments]
    experiments = [
        _build_experiment(
            match=match,
            index=index,
            workspace_root=workspace_dir,
            max_train_seconds=max_train_seconds,
            max_inference_seconds=max_inference_seconds,
        )
        for index, match in enumerate(matches, start=1)
    ]
    schedule = {
        "schema_version": 1,
        "state_hint": SCHEDULE_STATE_HINT,
        "lookup_report_path": str(lookup_file),
        "schedule_path": str(output_file),
        "workspace_root": str(workspace_dir),
        "lookup": lookup["lookup"],
        "experiment_count": len(experiments),
        "experiments": experiments,
        "scheduler_policy": {
            "selection_order": [
                "descending selection_score",
                "ascending priority_rank",
            ],
            "stop_conditions": [
                "accepted strategy reaches acceptance criteria",
                "all queued experiments are completed or rejected",
                "execution budget is exhausted",
            ],
            "requires_leaderboard_update": True,
        },
        "next_command": "claim-next-scheduled-experiment",
    }
    _write_json(output_file, schedule)
    return schedule


def claim_next_scheduled_experiment(
    *,
    schedule_path: str | Path,
    completed_experiment_ids: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    schedule_file = _require_file(Path(schedule_path), "local strategy schedule")
    schedule = _read_json_object(schedule_file, "local strategy schedule")
    _validate_schedule(schedule)
    completed = set(completed_experiment_ids)
    selected = None
    for experiment in schedule["experiments"]:
        if experiment.get("experiment_id") in completed:
            continue
        if experiment.get("status") != "queued":
            continue
        selected = dict(experiment)
        break
    if selected is None:
        raise LocalStrategySchedulerError("no queued scheduled experiment is available")

    selected["status"] = "claimed"
    claim = {
        "schema_version": 1,
        "state_hint": CLAIM_STATE_HINT,
        "schedule_path": str(schedule_file),
        "claim_path": str(output_path),
        "selected_experiment": selected,
        "next_command": selected["next_command"],
    }
    _write_json(Path(output_path), claim)
    return claim


def _build_experiment(
    *,
    match: dict[str, Any],
    index: int,
    workspace_root: Path,
    max_train_seconds: float,
    max_inference_seconds: float,
) -> dict[str, Any]:
    strategy_id = _require_string(match, "strategy_id", "strategy match")
    planned_commands = [
        {
            "step": command_index,
            "command": command,
            "status": "planned",
            "requires_runtime_arguments": True,
        }
        for command_index, command in enumerate(
            _require_string_list(match, "implementation_hooks", "strategy match"),
            start=1,
        )
    ]
    if not planned_commands:
        raise LocalStrategySchedulerError("strategy match has no implementation hooks")
    experiment_id = f"{strategy_id}_{index:03d}"
    workspace_dir = workspace_root / experiment_id
    return {
        "experiment_id": experiment_id,
        "strategy_id": strategy_id,
        "title": _require_string(match, "title", "strategy match"),
        "source_type": match.get("source_type"),
        "priority_rank": index,
        "selection_score": match.get("selection_score"),
        "status": "queued",
        "workspace_dir": str(workspace_dir),
        "next_command": planned_commands[0]["command"],
        "execution_plan": {
            "planned_commands": planned_commands,
            "budget": {
                "max_train_seconds": float(max_train_seconds),
                "max_inference_seconds": float(max_inference_seconds),
            },
            "gates": [
                "audit produced artifacts before scoring",
                "preserve first 10 initial-condition steps",
                "record failed experiments in research log",
            ],
        },
        "evaluation_protocol": match.get("evaluation_protocol"),
        "acceptance_criteria": match.get("acceptance_criteria", []),
        "known_risks": match.get("known_risks", []),
    }


def _validate_lookup(lookup: dict[str, Any]) -> None:
    if lookup.get("state_hint") != "LOCAL_STRATEGY_LOOKUP_COMPLETED":
        raise LocalStrategySchedulerError(
            "lookup report must have state_hint LOCAL_STRATEGY_LOOKUP_COMPLETED"
        )
    if not isinstance(lookup.get("lookup"), dict):
        raise LocalStrategySchedulerError("lookup report must include lookup object")
    matches = lookup.get("matches")
    if not isinstance(matches, list) or not matches:
        raise LocalStrategySchedulerError("lookup report must include non-empty matches")


def _validate_schedule(schedule: dict[str, Any]) -> None:
    if schedule.get("state_hint") != SCHEDULE_STATE_HINT:
        raise LocalStrategySchedulerError(
            f"schedule must have state_hint {SCHEDULE_STATE_HINT}"
        )
    experiments = schedule.get("experiments")
    if not isinstance(experiments, list):
        raise LocalStrategySchedulerError("schedule experiments must be a list")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LocalStrategySchedulerError(f"missing {label}: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalStrategySchedulerError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(data, dict):
        raise LocalStrategySchedulerError(f"{label} JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise LocalStrategySchedulerError(f"{label} must include string {field}")
    return value


def _require_string_list(
    data: dict[str, Any],
    field: str,
    label: str,
) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list):
        raise LocalStrategySchedulerError(f"{label} must include list {field}")
    if not all(isinstance(item, str) and item for item in value):
        raise LocalStrategySchedulerError(f"{label} {field} must contain strings")
    return value
