from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_next_action_policy import STATE_HINT, plan_hermes_next_action
from pde_agent.core.json_output import success_payload


def plan_hermes_next_action_command(
    *,
    diagnosis_path: str,
    output_path: str,
) -> dict[str, object]:
    plan = plan_hermes_next_action(
        diagnosis_path=Path(diagnosis_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="plan-hermes-next-action",
        state_hint=STATE_HINT,
        artifacts={"next_action_plan_path": plan["plan_path"]},
        metrics={
            "action": plan["action"],
            "command": plan["command"],
            "requires_human_review": plan["requires_human_review"],
            "terminal": plan["terminal"],
        },
        next_hint=plan["command"],
    )
