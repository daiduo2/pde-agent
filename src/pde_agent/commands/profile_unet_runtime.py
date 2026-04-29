from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_runtime_profile import STATE_HINT, profile_unet_runtime


def profile_unet_runtime_command(
    *,
    task_id: int,
    test_path: str,
    checkpoint_path: str,
    output_dir: str,
    shard_sizes: list[int],
    dataset_key: str,
    timeout_seconds: float,
    full_sample_count: int,
    force: bool,
) -> dict[str, object]:
    report = profile_unet_runtime(
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=Path(output_dir),
        shard_sizes=shard_sizes,
        dataset_key=dataset_key,
        timeout_seconds=timeout_seconds,
        full_sample_count=full_sample_count,
        force=force,
    )
    completed = [
        record for record in report["shards"] if record["status"] == "completed"
    ]
    max_shard_sample_count = max(
        (int(record["sample_count"]) for record in completed),
        default=0,
    )
    best_projected = min(
        (
            float(record["projected_full_inference_seconds"])
            for record in completed
        ),
        default=None,
    )
    return success_payload(
        run_id="local",
        command="profile-unet-runtime",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            "task_id": report["task_id"],
            "max_shard_sample_count": max_shard_sample_count,
            "best_projected_full_inference_seconds": best_projected,
            "timeout_seconds": report["timeout_seconds"],
            "recommended_strategy_id": report["recommendation"]["strategy_id"],
        },
        next_hint=str(report["recommendation"]["next_command"]),
    )
