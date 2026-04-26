"""Build suites of local validation windows from one PDEBench HDF5 source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.local_windows import build_local_windows


class LocalWindowSuiteError(RuntimeError):
    pass


def build_local_window_suite(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: list[int],
    time_reduction: int,
    space_reduction: int,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    output_root = Path(output_dir)
    horizons = _validate_forecast_steps(forecast_steps)
    source_info = _inspect_source(
        source_path=source,
        dataset_key=dataset_key,
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
    )
    _validate_horizon_capacity(horizons=horizons, source_info=source_info)

    manifest_path = output_root / "local_window_suite_manifest.json"
    output_paths = [_window_path(output_root, horizon) for horizon in horizons]
    _validate_overwrite(
        manifest_path=manifest_path,
        output_paths=output_paths,
        force=force,
    )

    profiles = []
    for horizon, output_path in zip(horizons, output_paths, strict=True):
        summary = build_local_windows(
            source_path=source,
            output_path=output_path,
            dataset_key=dataset_key,
            start_index=start_index,
            sample_limit=sample_limit,
            context_steps=context_steps,
            forecast_steps=horizon,
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            force=force,
        )
        profiles.append(
            {
                "profile_id": _profile_id(horizon),
                "forecast_steps": horizon,
                "windows_npz_path": summary["output_path"],
                "context_shape": summary["context_shape"],
                "target_shape": summary["target_shape"],
                "full_reference_shape": summary["full_reference_shape"],
            }
        )

    manifest = {
        "schema_version": 1,
        "state_hint": "LOCAL_WINDOW_SUITE_BUILT",
        "manifest_path": str(manifest_path),
        "source": source_info,
        "requested_forecast_steps": horizons,
        "context_steps": context_steps,
        "profile_count": len(profiles),
        "profiles": profiles,
        "next_command": "run-fno-iteration-dryrun",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _inspect_source(
    *,
    source_path: Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    time_reduction: int,
    space_reduction: int,
) -> dict[str, Any]:
    if not source_path.exists():
        raise LocalWindowSuiteError(f"source path does not exist: {source_path}")
    h5py = _import_h5py()
    with h5py.File(source_path, "r") as handle:
        if dataset_key not in handle:
            raise LocalWindowSuiteError(
                f"missing dataset key at HDF5 root: {dataset_key}"
            )
        dataset = handle[dataset_key]
        if not isinstance(dataset, h5py.Dataset):
            raise LocalWindowSuiteError(f"HDF5 key is not a dataset: {dataset_key}")
        raw_shape = [int(axis) for axis in dataset.shape]
        if len(raw_shape) != 3:
            raise LocalWindowSuiteError(f"HDF5 dataset {dataset_key!r} must be rank 3")
        sample_count = _actual_sample_count(
            start_index=start_index,
            sample_limit=sample_limit,
            raw_sample_count=raw_shape[0],
        )
        reduced_time = _reduced_length(raw_shape[1], time_reduction)
        reduced_space = _reduced_length(raw_shape[2], space_reduction)
    max_forecast_steps = reduced_time - context_steps
    return {
        "source_path": str(source_path),
        "dataset_key": dataset_key,
        "raw_shape": raw_shape,
        "reduced_shape": [sample_count, reduced_time, reduced_space],
        "start_index": start_index,
        "sample_limit": sample_limit,
        "actual_sample_count": sample_count,
        "time_reduction": time_reduction,
        "space_reduction": space_reduction,
        "max_forecast_steps": max_forecast_steps,
    }


def _validate_forecast_steps(forecast_steps: list[int]) -> list[int]:
    if not forecast_steps:
        raise LocalWindowSuiteError("forecast_steps must be non-empty")
    seen: set[int] = set()
    horizons = []
    for horizon in forecast_steps:
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise LocalWindowSuiteError("forecast_steps must be integers")
        if horizon <= 0:
            raise LocalWindowSuiteError("forecast_steps must be > 0")
        if horizon in seen:
            raise LocalWindowSuiteError(
                f"duplicate forecast_steps value requested: {horizon}"
            )
        seen.add(horizon)
        horizons.append(horizon)
    return horizons


def _validate_horizon_capacity(
    *,
    horizons: list[int],
    source_info: dict[str, Any],
) -> None:
    max_forecast_steps = source_info["max_forecast_steps"]
    if max_forecast_steps <= 0:
        raise LocalWindowSuiteError(
            "reduced time dimension leaves no forecast steps after context"
        )
    requested_max = max(horizons)
    if requested_max > max_forecast_steps:
        raise LocalWindowSuiteError(
            f"requested forecast_steps {requested_max} exceeds "
            f"max_forecast_steps {max_forecast_steps}"
        )


def _validate_overwrite(
    *,
    manifest_path: Path,
    output_paths: list[Path],
    force: bool,
) -> None:
    existing = [path for path in [manifest_path, *output_paths] if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise LocalWindowSuiteError(f"window suite artifact already exists: {names}")


def _actual_sample_count(
    *,
    start_index: int,
    sample_limit: int | None,
    raw_sample_count: int,
) -> int:
    if not isinstance(start_index, int) or isinstance(start_index, bool):
        raise LocalWindowSuiteError("start_index must be an integer")
    if start_index < 0:
        raise LocalWindowSuiteError("start_index must be >= 0")
    if start_index >= raw_sample_count:
        raise LocalWindowSuiteError("start_index is outside sample count")
    if sample_limit is not None:
        if not isinstance(sample_limit, int) or isinstance(sample_limit, bool):
            raise LocalWindowSuiteError("sample_limit must be an integer")
        if sample_limit <= 0:
            raise LocalWindowSuiteError("sample_limit must be > 0")
        stop_index = start_index + sample_limit
    else:
        stop_index = raw_sample_count
    if stop_index > raw_sample_count:
        raise LocalWindowSuiteError(
            f"requested sample range [{start_index}, {stop_index}) "
            f"exceeds sample count {raw_sample_count}"
        )
    return stop_index - start_index


def _profile_id(forecast_steps: int) -> str:
    return f"f{forecast_steps:03d}"


def _window_path(output_dir: Path, forecast_steps: int) -> Path:
    return output_dir / f"windows_{_profile_id(forecast_steps)}.npz"


def _reduced_length(length: int, stride: int) -> int:
    if not isinstance(stride, int) or isinstance(stride, bool):
        raise LocalWindowSuiteError("reduction strides must be integers")
    if stride <= 0:
        raise LocalWindowSuiteError("reduction strides must be > 0")
    if length == 0:
        return 0
    return ((length - 1) // stride) + 1


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise LocalWindowSuiteError(
            "h5py is required for HDF5 window suite building; "
            "run `uv sync --extra data`"
        ) from exc
    return h5py
