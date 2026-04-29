from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.research_chronicle import (
    STATE_HINT,
    build_research_chronicle,
)


def build_research_chronicle_command(
    *,
    artifact_paths: list[str],
    output_path: str,
    submission_dir: str | None,
    task_ids: list[int] | None,
    force: bool,
) -> dict[str, object]:
    report = build_research_chronicle(
        artifact_paths=[Path(path) for path in artifact_paths],
        output_path=Path(output_path),
        submission_dir=Path(submission_dir) if submission_dir is not None else None,
        task_ids=task_ids,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-research-chronicle",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics=report["metrics"],
        next_hint=str(report["next_command"]),
    )
