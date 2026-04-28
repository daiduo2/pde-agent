from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_strategy_growth_autopilot import (
    STATE_HINT,
    run_hermes_strategy_growth_autopilot,
)
from pde_agent.core.json_output import success_payload


def hermes_strategy_growth_autopilot_command(
    *,
    seed_artifact_path: str,
    workspace_dir: str,
    proposal_path: str,
    rubric_path: str,
    skill_name: str,
    install_root: str,
    registry_path: str,
    reviewer_id: str,
    max_steps: int,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_strategy_growth_autopilot(
        seed_artifact_path=Path(seed_artifact_path),
        workspace_dir=Path(workspace_dir),
        proposal_path=Path(proposal_path),
        rubric_path=Path(rubric_path),
        skill_name=skill_name,
        install_root=Path(install_root),
        registry_path=Path(registry_path),
        reviewer_id=reviewer_id,
        max_steps=max_steps,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="hermes-strategy-growth-autopilot",
        state_hint=STATE_HINT,
        artifacts={
            "autopilot_report_path": report["autopilot_report_path"],
            "registry_path": report["registry_path"],
            "installed_skill_manifest_path": report["installed_skill_manifest_path"],
        },
        metrics={
            "final_state_hint": report["final_state_hint"],
            "step_count": report["step_count"],
            "human_review_required": report["human_review_required"],
        },
        next_hint=str(report.get("next_command", "strategy-growth-complete")),
    )
