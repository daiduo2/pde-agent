from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_literature_expansion import (
    STATE_HINT,
    run_hermes_literature_strategy_expansion,
)
from pde_agent.core.json_output import success_payload


def run_hermes_literature_strategy_expansion_command(
    *,
    algorithm_candidates_path: str,
    workspace_dir: str,
    checkpoint_paths: list[str] | None,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=Path(algorithm_candidates_path),
        workspace_dir=Path(workspace_dir),
        checkpoint_paths=[Path(path) for path in (checkpoint_paths or [])],
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hermes-literature-strategy-expansion",
        state_hint=STATE_HINT,
        artifacts={"expansion_report_path": report["expansion_report_path"]},
        metrics={
            "action_count": report["action_count"],
            "executable_count": report["executable_count"],
            "proposal_count": report["proposal_count"],
        },
        next_hint=report["next_command"],
    )
