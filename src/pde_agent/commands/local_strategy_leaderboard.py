from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_leaderboard import (
    LEADERBOARD_STATE_HINT,
    record_local_strategy_result,
)


def record_local_strategy_result_command(
    *,
    schedule_path: str,
    experiment_id: str,
    result_summary_path: str,
    leaderboard_path: str,
    result_status: str,
    min_score_delta: float,
    max_inference_seconds: float,
    failure_reason: str | None,
) -> dict[str, object]:
    leaderboard = record_local_strategy_result(
        schedule_path=Path(schedule_path),
        experiment_id=experiment_id,
        result_summary_path=Path(result_summary_path),
        leaderboard_path=Path(leaderboard_path),
        result_status=result_status,
        min_score_delta=min_score_delta,
        max_inference_seconds=max_inference_seconds,
        failure_reason=failure_reason,
    )
    best = leaderboard["best_result"] or {}
    decision = leaderboard["decision"]
    return success_payload(
        run_id="local",
        command="record-local-strategy-result",
        state_hint=LEADERBOARD_STATE_HINT,
        artifacts={"local_strategy_leaderboard_path": leaderboard["leaderboard_path"]},
        metrics={
            "result_count": leaderboard["result_count"],
            "decision": decision["decision"],
            "best_experiment_id": best.get("experiment_id"),
            "best_strategy_id": best.get("strategy_id"),
            "best_score_delta": best.get("score_delta"),
        },
        next_hint=str(decision["next_command"]),
    )
