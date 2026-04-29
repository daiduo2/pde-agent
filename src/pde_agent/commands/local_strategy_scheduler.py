from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_scheduler import (
    CLAIM_STATE_HINT,
    SCHEDULE_STATE_HINT,
    claim_next_scheduled_experiment,
    schedule_local_strategy_evaluation,
)


def schedule_local_strategy_evaluation_command(
    *,
    lookup_report_path: str,
    output_path: str,
    workspace_root: str,
    max_experiments: int,
    max_train_seconds: float,
    max_inference_seconds: float,
    force: bool,
) -> dict[str, object]:
    schedule = schedule_local_strategy_evaluation(
        lookup_report_path=Path(lookup_report_path),
        output_path=Path(output_path),
        workspace_root=Path(workspace_root),
        max_experiments=max_experiments,
        max_train_seconds=max_train_seconds,
        max_inference_seconds=max_inference_seconds,
        force=force,
    )
    first = schedule["experiments"][0] if schedule["experiments"] else {}
    return success_payload(
        run_id="local",
        command="schedule-local-strategy-evaluation",
        state_hint=SCHEDULE_STATE_HINT,
        artifacts={"local_strategy_schedule_path": schedule["schedule_path"]},
        metrics={
            "experiment_count": schedule["experiment_count"],
            "top_strategy_id": first.get("strategy_id"),
            "top_next_command": first.get("next_command"),
        },
        next_hint=str(schedule["next_command"]),
    )


def claim_next_scheduled_experiment_command(
    *,
    schedule_path: str,
    completed_experiment_ids: list[str],
    output_path: str,
) -> dict[str, object]:
    claim = claim_next_scheduled_experiment(
        schedule_path=Path(schedule_path),
        completed_experiment_ids=completed_experiment_ids,
        output_path=Path(output_path),
    )
    selected = claim["selected_experiment"]
    return success_payload(
        run_id="local",
        command="claim-next-scheduled-experiment",
        state_hint=CLAIM_STATE_HINT,
        artifacts={"scheduled_experiment_claim_path": claim["claim_path"]},
        metrics={
            "experiment_id": selected["experiment_id"],
            "strategy_id": selected["strategy_id"],
            "next_command": claim["next_command"],
        },
        next_hint=str(claim["next_command"]),
    )
