
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_route import (
    ROUTE_STATE_HINT,
    route_local_strategy_next_action,
)


def route_local_strategy_next_action_command(
    *,
    artifact_path: str,
    output_path: str,
) -> dict[str, object]:
    report = route_local_strategy_next_action(
        artifact_path=Path(artifact_path),
        output_path=Path(output_path),
    )
    route = report["route"]
    return success_payload(
        run_id="local",
        command="route-local-strategy-next-action",
        state_hint=ROUTE_STATE_HINT,
        artifacts={"local_strategy_route_path": report["route_report_path"]},
        metrics={
            "source_state_hint": report["source_state_hint"],
            "action": route["action"],
            "command": route["command"],
            "requires_human_review": route["requires_human_review"],
        },
        next_hint=str(route["command"]),
    )
