"""Unified local strategy result memory and leaderboard."""


import json
from pathlib import Path
from typing import Any


LEADERBOARD_STATE_HINT = "LOCAL_STRATEGY_LEADERBOARD_UPDATED"
VALID_RESULT_STATUSES = {"completed", "failed", "rejected"}


class LocalStrategyLeaderboardError(RuntimeError):
    pass


def record_local_strategy_result(
    *,
    schedule_path: str | Path,
    experiment_id: str,
    result_summary_path: str | Path,
    leaderboard_path: str | Path,
    result_status: str,
    min_score_delta: float,
    max_inference_seconds: float,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    if result_status not in VALID_RESULT_STATUSES:
        raise LocalStrategyLeaderboardError(
            f"result_status must be one of {sorted(VALID_RESULT_STATUSES)}"
        )
    schedule_file = _require_file(Path(schedule_path), "local strategy schedule")
    result_file = _require_file(Path(result_summary_path), "experiment result summary")
    leaderboard_file = Path(leaderboard_path)
    schedule = _read_json_object(schedule_file, "local strategy schedule")
    result = _read_json_object(result_file, "experiment result summary")
    experiment = _find_experiment(schedule=schedule, experiment_id=experiment_id)
    previous_rows = _read_existing_rows(leaderboard_file)
    row = _build_row(
        schedule_file=schedule_file,
        result_file=result_file,
        experiment=experiment,
        result=result,
        result_status=result_status,
        failure_reason=failure_reason,
        min_score_delta=min_score_delta,
        max_inference_seconds=max_inference_seconds,
    )
    rows = [
        existing
        for existing in previous_rows
        if existing.get("experiment_id") != experiment_id
    ]
    rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item.get("decision_score", -1_000_000.0)),
            int(item.get("priority_rank", 999)),
            str(item.get("experiment_id", "")),
        )
    )
    completed_ids = sorted(
        str(item["experiment_id"])
        for item in rows
        if item.get("status") in VALID_RESULT_STATUSES
    )
    scheduled_ids = [
        str(item["experiment_id"])
        for item in schedule.get("experiments", [])
        if isinstance(item, dict) and isinstance(item.get("experiment_id"), str)
    ]
    pending_ids = [
        scheduled_id
        for scheduled_id in scheduled_ids
        if scheduled_id not in completed_ids
    ]
    best = rows[0] if rows else None
    decision = _decision(
        best=best,
        pending_experiment_ids=pending_ids,
        min_score_delta=min_score_delta,
        max_inference_seconds=max_inference_seconds,
    )
    leaderboard = {
        "schema_version": 1,
        "state_hint": LEADERBOARD_STATE_HINT,
        "leaderboard_path": str(leaderboard_file),
        "schedule_path": str(schedule_file),
        "result_count": len(rows),
        "results": rows,
        "best_result": best,
        "completed_experiment_ids": completed_ids,
        "pending_experiment_ids": pending_ids,
        "decision": decision,
        "next_command": decision["next_command"],
    }
    _write_json(leaderboard_file, leaderboard)
    return leaderboard


def _build_row(
    *,
    schedule_file: Path,
    result_file: Path,
    experiment: dict[str, Any],
    result: dict[str, Any],
    result_status: str,
    failure_reason: str | None,
    min_score_delta: float,
    max_inference_seconds: float,
) -> dict[str, Any]:
    score = _number_or_none(result.get("score"))
    if score is None:
        score = _number_or_none(result.get("segment_score"))
    baseline_score = _number_or_none(result.get("baseline_score"))
    score_delta = _number_or_none(result.get("score_delta"))
    if score_delta is None and score is not None and baseline_score is not None:
        score_delta = float(score - baseline_score)
    inference_seconds = _number_or_none(result.get("inference_time_seconds"))
    if inference_seconds is None:
        inference_seconds = _number_or_none(result.get("inference_time"))
    accepted = (
        result_status == "completed"
        and score_delta is not None
        and score_delta >= min_score_delta
        and (
            inference_seconds is None
            or inference_seconds <= max_inference_seconds
        )
    )
    if accepted:
        outcome = "accepted"
    elif result_status == "completed":
        outcome = "measured_rejected"
    else:
        outcome = result_status
    return {
        "schema_version": 1,
        "state_hint": "LOCAL_STRATEGY_RESULT_RECORDED",
        "schedule_path": str(schedule_file),
        "result_summary_path": str(result_file),
        "experiment_id": experiment["experiment_id"],
        "strategy_id": experiment["strategy_id"],
        "title": experiment["title"],
        "priority_rank": experiment["priority_rank"],
        "selection_score": experiment.get("selection_score"),
        "status": result_status,
        "outcome": outcome,
        "score": score,
        "baseline_score": baseline_score,
        "score_delta": score_delta,
        "inference_time_seconds": inference_seconds,
        "decision_score": _decision_score(
            score_delta=score_delta,
            result_status=result_status,
            inference_seconds=inference_seconds,
            max_inference_seconds=max_inference_seconds,
        ),
        "accepted": accepted,
        "failure_reason": failure_reason or result.get("error"),
        "acceptance_criteria": experiment.get("acceptance_criteria", []),
        "evaluation_protocol": experiment.get("evaluation_protocol"),
    }


def _decision(
    *,
    best: dict[str, Any] | None,
    pending_experiment_ids: list[str],
    min_score_delta: float,
    max_inference_seconds: float,
) -> dict[str, Any]:
    if best is not None and bool(best.get("accepted")):
        return {
            "decision": "accept_best_strategy",
            "reason_codes": [
                "score_delta_meets_threshold",
                "inference_time_within_budget",
            ],
            "thresholds": {
                "min_score_delta": float(min_score_delta),
                "max_inference_seconds": float(max_inference_seconds),
            },
            "next_command": "record-strategy-proposal",
        }
    if pending_experiment_ids:
        return {
            "decision": "continue_scheduled_experiments",
            "reason_codes": ["no_accepted_result_yet", "pending_experiments_remain"],
            "thresholds": {
                "min_score_delta": float(min_score_delta),
                "max_inference_seconds": float(max_inference_seconds),
            },
            "next_command": "claim-next-scheduled-experiment",
        }
    return {
        "decision": "local_strategy_queue_exhausted",
        "reason_codes": ["no_accepted_result", "no_pending_experiments"],
        "thresholds": {
            "min_score_delta": float(min_score_delta),
            "max_inference_seconds": float(max_inference_seconds),
        },
        "next_command": "build-research-chronicle",
    }


def _decision_score(
    *,
    score_delta: float | None,
    result_status: str,
    inference_seconds: float | None,
    max_inference_seconds: float,
) -> float:
    if result_status != "completed":
        return -1_000_000.0
    value = float(score_delta) if score_delta is not None else -1_000.0
    if inference_seconds is not None and inference_seconds > max_inference_seconds:
        value -= 1_000.0
    return value


def _find_experiment(
    *,
    schedule: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    if schedule.get("state_hint") != "LOCAL_STRATEGY_EVALUATION_SCHEDULED":
        raise LocalStrategyLeaderboardError("schedule has invalid state_hint")
    experiments = schedule.get("experiments")
    if not isinstance(experiments, list):
        raise LocalStrategyLeaderboardError("schedule experiments must be a list")
    for experiment in experiments:
        if (
            isinstance(experiment, dict)
            and experiment.get("experiment_id") == experiment_id
        ):
            return experiment
    raise LocalStrategyLeaderboardError(
        f"experiment_id {experiment_id!r} is not present in schedule"
    )


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    leaderboard = _read_json_object(path, "local strategy leaderboard")
    if leaderboard.get("state_hint") != LEADERBOARD_STATE_HINT:
        raise LocalStrategyLeaderboardError("leaderboard has invalid state_hint")
    rows = leaderboard.get("results")
    if not isinstance(rows, list):
        raise LocalStrategyLeaderboardError("leaderboard results must be a list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LocalStrategyLeaderboardError(f"missing {label}: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalStrategyLeaderboardError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(data, dict):
        raise LocalStrategyLeaderboardError(f"{label} JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
