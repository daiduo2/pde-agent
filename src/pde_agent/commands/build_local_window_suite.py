
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_window_suite import build_local_window_suite as build_suite


def build_local_window_suite(
    *,
    source_path: str,
    output_dir: str,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: list[int],
    time_reduction: int,
    space_reduction: int,
    force: bool,
) -> dict[str, object]:
    manifest = build_suite(
        source_path=Path(source_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-local-window-suite",
        state_hint="LOCAL_WINDOW_SUITE_BUILT",
        artifacts={
            "manifest_path": manifest["manifest_path"],
            "window_paths": [
                profile["windows_npz_path"] for profile in manifest["profiles"]
            ],
        },
        metrics={
            "profile_count": manifest["profile_count"],
            "requested_forecast_steps": manifest["requested_forecast_steps"],
            "max_forecast_steps": manifest["source"]["max_forecast_steps"],
            "sample_count": manifest["source"]["actual_sample_count"],
            "space_points": manifest["source"]["reduced_shape"][2],
        },
        next_hint="run-fno-iteration-dryrun",
    )
