from __future__ import annotations

from pathlib import Path

from pde_agent.core.competition_windows import (
    STATE_HINT,
    prepare_competition_windows,
)
from pde_agent.core.json_output import success_payload


def prepare_competition_windows_command(
    *,
    task_id: int,
    test_path: str,
    output_path: str,
    dataset_key: str,
    force: bool,
) -> dict[str, object]:
    report = prepare_competition_windows(
        task_id=task_id,
        test_path=Path(test_path),
        output_path=Path(output_path),
        dataset_key=dataset_key,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="prepare-competition-windows",
        state_hint=STATE_HINT,
        artifacts={"windows_path": report["windows_path"]},
        metrics={
            "task_id": report["task_id"],
            "sample_count": report["sample_count"],
            "context_steps": report["context_steps"],
            "forecast_steps": report["forecast_steps"],
            "space_points": report["space_points"],
        },
        next_hint="predict-unet-local-or-predict-fno-local",
    )
