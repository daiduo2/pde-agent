from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_loop_audit import (
    AUDIT_STATE_HINT,
    audit_local_strategy_loop,
)


def audit_local_strategy_loop_command(
    *,
    output_dir: str,
    report_path: str,
    force: bool,
) -> dict[str, object]:
    report = audit_local_strategy_loop(
        output_dir=Path(output_dir),
        report_path=Path(report_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="audit-local-strategy-loop",
        state_hint=AUDIT_STATE_HINT,
        artifacts={"local_strategy_loop_audit_path": report["audit_report_path"]},
        metrics={
            "decision": report["decision"],
            "scenario_count": report["scenario_count"],
            "accepted_count": report["summary"]["accepted_count"],
            "continued_count": report["summary"]["continued_count"],
            "exhausted_count": report["summary"]["exhausted_count"],
        },
        next_hint=str(report["next_command"]),
    )
