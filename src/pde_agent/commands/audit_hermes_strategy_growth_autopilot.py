from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_strategy_growth_autopilot_audit import (
    STATE_HINT,
    audit_hermes_strategy_growth_autopilot_report,
)
from pde_agent.core.json_output import success_payload


def audit_hermes_strategy_growth_autopilot_command(
    *,
    autopilot_report_path: str,
    output_path: str,
) -> dict[str, object]:
    audit = audit_hermes_strategy_growth_autopilot_report(
        autopilot_report_path=Path(autopilot_report_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="audit-hermes-strategy-growth-autopilot",
        state_hint=STATE_HINT,
        artifacts={"audit_path": audit["audit_path"]},
        metrics={
            "decision": audit["decision"],
            "issue_count": audit["issue_count"],
            "step_count": audit["summary"]["step_count"],
        },
        next_hint=str(audit["next_command"]),
    )
