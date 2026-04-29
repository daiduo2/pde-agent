from __future__ import annotations

from pathlib import Path

from pde_agent.core.hermes_operator_tuning_loop import (
    STATE_HINT,
    run_hermes_operator_tuning_loop,
)
from pde_agent.core.json_output import success_payload


def run_hermes_operator_tuning_loop_command(
    *,
    output_dir: str,
    operator_family: str,
    task_id: int,
    test_path: str,
    checkpoint_path: str,
    shard_sizes: list[int],
    inference_batch_sizes: list[int],
    torch_num_threads_values: list[int],
    dataset_key: str,
    timeout_seconds: float,
    full_sample_count: int,
    force: bool,
) -> dict[str, object]:
    report = run_hermes_operator_tuning_loop(
        output_dir=Path(output_dir),
        operator_family=operator_family,
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        shard_sizes=shard_sizes,
        inference_batch_sizes=inference_batch_sizes,
        torch_num_threads_values=torch_num_threads_values,
        dataset_key=dataset_key,
        timeout_seconds=timeout_seconds,
        full_sample_count=full_sample_count,
        force=force,
    )
    selected = report["selected_config"] or {}
    return success_payload(
        run_id="local",
        command="run-hermes-operator-tuning-loop",
        state_hint=STATE_HINT,
        artifacts={
            "operator_tuning_report_path": report["loop_report_path"],
            "local_strategy_result_path": report["artifacts"][
                "local_strategy_result_path"
            ],
        },
        metrics={
            "operator_family": report["operator_family"],
            "evaluated_config_count": report["evaluated_config_count"],
            "selected_inference_batch_size": selected.get("inference_batch_size"),
            "selected_torch_num_threads": selected.get("torch_num_threads"),
            "best_projected_full_inference_seconds": report[
                "best_projected_full_inference_seconds"
            ],
            "decision": report["decision"]["action"],
        },
        next_hint=str(report["next_command"]),
    )
