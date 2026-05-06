from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_exploration_scheduler import (
    STATE_HINT,
    build_hermes_exploration_schedule,
)
from pde_agent.core.json_output import success_payload


def build_hermes_exploration_schedule_command(
    *,
    memory_json_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_hermes_exploration_schedule(
        memory_json_path=Path(memory_json_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hermes-exploration-schedule",
        state_hint=STATE_HINT,
        artifacts={"hermes_exploration_schedule_json": output_path},
        metrics={
            "decision": report["decision"],
            "candidate_generator": report["schedule"]["candidate_generator"],
            "blocked_candidate_family_count": len(
                report["schedule"].get("blocked_candidate_families", [])
            ),
        },
        next_hint=report["next_command"],
    )
