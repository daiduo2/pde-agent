
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_windows import build_local_windows as build_windows


def build_local_windows(
    *,
    source_path: str,
    output_path: str,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    force: bool,
) -> dict[str, object]:
    summary = build_windows(
        source_path=Path(source_path),
        output_path=Path(output_path),
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    metadata = summary["metadata"]
    return success_payload(
        run_id="local",
        command="build-local-windows",
        state_hint="LOCAL_WINDOWS_BUILT",
        artifacts={"windows_npz_path": summary["output_path"]},
        metrics={
            "sample_count": metadata["actual_sample_count"],
            "context_steps": metadata["context_steps"],
            "forecast_steps": metadata["forecast_steps"],
            "space_points": summary["target_shape"][2],
        },
        next_hint="score-local-rollout once a prediction NPZ is available",
    )
