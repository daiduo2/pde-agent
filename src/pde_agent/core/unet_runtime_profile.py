"""Profile official UNet inference runtime on increasing competition shards."""


import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "UNET_RUNTIME_PROFILED"


class UNetRuntimeProfileError(RuntimeError):
    pass


Runner = Callable[..., dict[str, Any]]


def profile_unet_runtime(
    *,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    shard_sizes: Sequence[int],
    dataset_key: str = "tensor",
    timeout_seconds: float = 120.0,
    full_sample_count: int = 1000,
    inference_batch_size: int | None = None,
    torch_num_threads: int | None = None,
    force: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise UNetRuntimeProfileError("task_id must be 1 or 2")
    if timeout_seconds <= 0.0 or full_sample_count <= 0:
        raise UNetRuntimeProfileError(
            "timeout_seconds and full_sample_count must be positive"
        )
    shard_counts = _normalize_shard_sizes(shard_sizes)
    test_file = Path(test_path)
    if not test_file.exists():
        raise UNetRuntimeProfileError(f"test HDF5 not found: {test_file}")
    output_root = Path(output_dir)
    report_path = output_root / "unet_runtime_profile.json"
    if report_path.exists() and not force:
        raise UNetRuntimeProfileError(f"UNet runtime profile already exists: {report_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_sample_count = _read_sample_count(
        test_path=test_file,
        dataset_key=dataset_key,
    )
    execution_runner = runner or predict_unet_competition
    shards: list[dict[str, Any]] = []
    for requested_count in shard_counts:
        sample_count = min(requested_count, source_sample_count)
        if sample_count < 1:
            raise UNetRuntimeProfileError("source test HDF5 contains no samples")
        shard_path = output_root / "shards" / f"task{task_id}_n{sample_count:04d}.hdf5"
        _write_hdf5_shard(
            source_path=test_file,
            output_path=shard_path,
            dataset_key=dataset_key,
            sample_count=sample_count,
            force=force,
        )
        shard_output_dir = output_root / "runs" / f"task{task_id}_n{sample_count:04d}"
        start = time.perf_counter()
        try:
            run_report = execution_runner(
                task_id=task_id,
                test_path=shard_path,
                checkpoint_path=Path(checkpoint_path),
                output_dir=shard_output_dir,
                dataset_key=dataset_key,
                inference_batch_size=inference_batch_size,
                torch_num_threads=torch_num_threads,
                force=True,
            )
            wall_elapsed = time.perf_counter() - start
            inference_elapsed = float(
                run_report.get("inference", {}).get("elapsed_seconds", wall_elapsed)
            )
            shards.append(
                _build_success_record(
                    sample_count=sample_count,
                    requested_sample_count=requested_count,
                    shard_path=shard_path,
                    shard_output_dir=shard_output_dir,
                    run_report=run_report,
                    inference_elapsed=inference_elapsed,
                    wall_elapsed=wall_elapsed,
                    full_sample_count=full_sample_count,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI workflows.
            wall_elapsed = time.perf_counter() - start
            shards.append(
                {
                    "status": "failed",
                    "sample_count": sample_count,
                    "requested_sample_count": requested_count,
                    "shard_test_path": str(shard_path),
                    "shard_output_dir": str(shard_output_dir),
                    "wall_elapsed_seconds": wall_elapsed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    recommendation = _build_recommendation(
        shards=shards,
        timeout_seconds=timeout_seconds,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "test_path": str(test_file),
        "checkpoint_path": str(checkpoint_path),
        "dataset_key": dataset_key,
        "timeout_seconds": float(timeout_seconds),
        "full_sample_count": int(full_sample_count),
        "inference_batch_size": inference_batch_size,
        "torch_num_threads": torch_num_threads,
        "source_sample_count": int(source_sample_count),
        "shards": shards,
        "recommendation": recommendation,
        "artifacts": {"runtime_profile_path": str(report_path)},
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _normalize_shard_sizes(shard_sizes: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for size in shard_sizes:
        if size <= 0:
            raise UNetRuntimeProfileError("shard sizes must be positive")
        if int(size) not in normalized:
            normalized.append(int(size))
    if not normalized:
        raise UNetRuntimeProfileError("at least one shard size is required")
    return normalized


def _read_sample_count(*, test_path: Path, dataset_key: str) -> int:
    h5py = __import__("h5py")
    with h5py.File(test_path, "r") as handle:
        if dataset_key not in handle:
            raise UNetRuntimeProfileError(f"dataset {dataset_key!r} not found")
        dataset = handle[dataset_key]
        if len(dataset.shape) != 3:
            raise UNetRuntimeProfileError(
                f"expected rank-3 tensor, got shape {dataset.shape}"
            )
        return int(dataset.shape[0])


def _write_hdf5_shard(
    *,
    source_path: Path,
    output_path: Path,
    dataset_key: str,
    sample_count: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise UNetRuntimeProfileError(f"UNet runtime shard already exists: {output_path}")
    h5py = __import__("h5py")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as target:
        source_sample_count = int(source[dataset_key].shape[0])
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for name, dataset in source.items():
            if not hasattr(dataset, "shape"):
                continue
            if name == dataset_key:
                data = dataset[:sample_count]
            elif dataset.shape and int(dataset.shape[0]) == source_sample_count:
                data = dataset[:sample_count]
            else:
                data = dataset[:]
            target.create_dataset(name, data=data)


def _build_success_record(
    *,
    sample_count: int,
    requested_sample_count: int,
    shard_path: Path,
    shard_output_dir: Path,
    run_report: dict[str, Any],
    inference_elapsed: float,
    wall_elapsed: float,
    full_sample_count: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    seconds_per_sample = inference_elapsed / float(sample_count)
    projected = seconds_per_sample * float(full_sample_count)
    return {
        "status": "completed",
        "sample_count": int(sample_count),
        "requested_sample_count": int(requested_sample_count),
        "shard_test_path": str(shard_path),
        "shard_output_dir": str(shard_output_dir),
        "candidate_execution_report_path": run_report.get("artifacts", {}).get(
            "candidate_execution_report_path"
        ),
        "inference_elapsed_seconds": inference_elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "seconds_per_sample": seconds_per_sample,
        "projected_full_inference_seconds": projected,
        "timeout_risk": projected > timeout_seconds,
        "phase_timings": run_report.get("inference", {}).get("phase_timings", {}),
    }


def _build_recommendation(
    *,
    shards: Sequence[dict[str, Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    completed = [record for record in shards if record["status"] == "completed"]
    if not completed:
        return {
            "strategy_id": "unet_runtime_profile_repair_v0",
            "next_command": "record-unet-timeout-evidence",
            "rationale": "No UNet shard completed; record negative evidence and repair runtime path.",
        }
    best = min(
        completed,
        key=lambda record: float(record["projected_full_inference_seconds"]),
    )
    if float(best["projected_full_inference_seconds"]) <= timeout_seconds:
        return {
            "strategy_id": "unet_full_submission_runtime_v0",
            "next_command": "predict-unet-competition",
            "rationale": (
                "Best observed UNet shard projects under the competition inference budget."
            ),
            "best_projected_full_inference_seconds": best[
                "projected_full_inference_seconds"
            ],
        }
    return {
        "strategy_id": "unet_timeout_optimization_v0",
        "next_command": "record-unet-timeout-evidence",
        "rationale": (
            "Completed UNet shards project over the competition inference budget; "
            "optimize rollout throughput before full submission."
        ),
        "best_projected_full_inference_seconds": best[
            "projected_full_inference_seconds"
        ],
    }
