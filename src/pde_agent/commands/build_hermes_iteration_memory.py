from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_iteration_memory import (
    STATE_HINT,
    build_hermes_iteration_memory,
)
from pde_agent.core.json_output import success_payload


def build_hermes_iteration_memory_command(
    *,
    batch_loop_report_path: str,
    output_path: str,
    memory_json_path: str,
    submission_dir: str | None,
    task_ids: list[int],
    force: bool,
) -> dict[str, object]:
    report = build_hermes_iteration_memory(
        batch_loop_report_path=Path(batch_loop_report_path),
        output_path=Path(output_path),
        memory_json_path=Path(memory_json_path),
        submission_dir=Path(submission_dir) if submission_dir is not None else None,
        task_ids=task_ids or None,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hermes-iteration-memory",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics=report["metrics"],
        next_hint=report["next_command"],
    )
