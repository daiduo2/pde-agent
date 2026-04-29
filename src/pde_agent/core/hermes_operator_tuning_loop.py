"""Hermes-owned operator execution tuning loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pde_agent.core.unet_runtime_profile import profile_unet_runtime


STATE_HINT = "HERMES_OPERATOR_TUNING_LOOP_RAN"
Profiler = Callable[..., dict[str, Any]]


class HermesOperatorTuningLoopError(RuntimeError):
    pass


def run_hermes_operator_tuning_loop(
    *,
    output_dir: str | Path,
    operator_family: str,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    shard_sizes: Sequence[int],
    inference_batch_sizes: Sequence[int],
    torch_num_threads_values: Sequence[int],
    dataset_key: str = "tensor",
    timeout_seconds: float = 120.0,
    full_sample_count: int = 1000,
    profiler: Profiler | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if operator_family != "UNet":
        raise HermesOperatorTuningLoopError(
            "operator tuning loop currently supports UNet only"
        )
    batches = _positive_unique(inference_batch_sizes, "inference_batch_sizes")
    thread_values = _positive_unique(torch_num_threads_values, "torch_num_threads_values")
    if not shard_sizes:
        raise HermesOperatorTuningLoopError("at least one shard size is required")

    root = Path(output_dir)
    report_path = root / "hermes_operator_tuning_loop_report.json"
    result_path = root / "local_strategy_result.json"
    if report_path.exists() and not force:
        raise HermesOperatorTuningLoopError(
            f"operator tuning report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    run_profiler = profiler or profile_unet_runtime
    evaluations: list[dict[str, Any]] = []
    for batch in batches:
        for threads in thread_values:
            profile_dir = root / "profiles" / f"unet_b{batch}_t{threads}"
            profile = run_profiler(
                task_id=task_id,
                test_path=Path(test_path),
                checkpoint_path=Path(checkpoint_path),
                output_dir=profile_dir,
                shard_sizes=shard_sizes,
                dataset_key=dataset_key,
                timeout_seconds=timeout_seconds,
                full_sample_count=full_sample_count,
                inference_batch_size=batch,
                torch_num_threads=threads,
                force=True,
            )
            evaluations.append(
                _summarize_profile(
                    profile=profile,
                    inference_batch_size=batch,
                    torch_num_threads=threads,
                )
            )

    best = _select_best(evaluations)
    decision = _decision(best=best, timeout_seconds=timeout_seconds)
    local_result = _build_local_result(
        result_path=result_path,
        operator_family=operator_family,
        best=best,
        decision=decision,
        timeout_seconds=timeout_seconds,
    )
    _write_json(result_path, local_result)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "loop_report_path": str(report_path),
        "operator_family": operator_family,
        "task_id": task_id,
        "test_path": str(test_path),
        "checkpoint_path": str(checkpoint_path),
        "permissions": {
            "diagnostic_profile": True,
            "execution_path_tuning": True,
            "model_logic_tuning": False,
        },
        "evaluated_config_count": len(evaluations),
        "evaluations": evaluations,
        "selected_config": best["config"] if best is not None else None,
        "best_projected_full_inference_seconds": (
            best["projected_full_inference_seconds"] if best is not None else None
        ),
        "decision": decision,
        "artifacts": {
            "operator_tuning_report_path": str(report_path),
            "local_strategy_result_path": str(result_path),
        },
        "next_command": decision["next_command"],
    }
    _write_json(report_path, report)
    return report


def _positive_unique(values: Sequence[int], label: str) -> list[int]:
    normalized: list[int] = []
    for value in values:
        if value <= 0:
            raise HermesOperatorTuningLoopError(f"{label} must contain positive values")
        if int(value) not in normalized:
            normalized.append(int(value))
    if not normalized:
        raise HermesOperatorTuningLoopError(f"{label} must not be empty")
    return normalized


def _summarize_profile(
    *,
    profile: dict[str, Any],
    inference_batch_size: int,
    torch_num_threads: int,
) -> dict[str, Any]:
    completed = [
        shard
        for shard in profile.get("shards", [])
        if isinstance(shard, dict) and shard.get("status") == "completed"
    ]
    best_shard = min(
        completed,
        key=lambda shard: float(shard["projected_full_inference_seconds"]),
        default=None,
    )
    return {
        "config": {
            "inference_batch_size": inference_batch_size,
            "torch_num_threads": torch_num_threads,
        },
        "profile_path": profile.get("artifacts", {}).get("runtime_profile_path"),
        "completed_shard_count": len(completed),
        "best_shard": best_shard,
        "projected_full_inference_seconds": (
            float(best_shard["projected_full_inference_seconds"])
            if best_shard is not None
            else None
        ),
        "profile_recommendation": profile.get("recommendation", {}),
    }


def _select_best(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [
        item
        for item in evaluations
        if item.get("projected_full_inference_seconds") is not None
    ]
    if not completed:
        return None
    return min(
        completed,
        key=lambda item: (
            float(item["projected_full_inference_seconds"]),
            int(item["config"]["inference_batch_size"]),
            int(item["config"]["torch_num_threads"]),
        ),
    )


def _decision(
    *,
    best: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if best is None:
        return {
            "action": "record_timeout_evidence",
            "reason_codes": ["no_completed_runtime_profile"],
            "next_command": "record-unet-timeout-evidence",
        }
    projected = float(best["projected_full_inference_seconds"])
    if projected <= timeout_seconds:
        return {
            "action": "accept_execution_config",
            "reason_codes": ["projected_inference_under_timeout"],
            "next_command": "predict-unet-competition",
        }
    return {
        "action": "record_timeout_evidence",
        "reason_codes": ["best_projected_inference_over_timeout"],
        "next_command": "record-unet-timeout-evidence",
    }


def _build_local_result(
    *,
    result_path: Path,
    operator_family: str,
    best: dict[str, Any] | None,
    decision: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    selected = best["config"] if best is not None else None
    projected = (
        float(best["projected_full_inference_seconds"]) if best is not None else None
    )
    completed = decision["action"] == "accept_execution_config"
    return {
        "schema_version": 1,
        "state_hint": "LOCAL_EXPERIMENT_RESULT",
        "result_path": str(result_path),
        "result_status": "completed" if completed else "failed",
        "score": None,
        "baseline_score": None,
        "score_delta": 0.0 if completed else None,
        "inference_time_seconds": projected,
        "failure_reason": None if completed else "operator_runtime_timeout",
        "result_summary": (
            f"Hermes selected {operator_family} execution config {selected} "
            f"with projected inference {projected} seconds."
        ),
        "source": {
            "state_hint": STATE_HINT,
            "operator_family": operator_family,
            "selected_config": selected,
            "projected_full_inference_seconds": projected,
            "timeout_seconds": float(timeout_seconds),
            "decision": decision,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
