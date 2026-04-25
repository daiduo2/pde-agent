from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LocalWindowError(RuntimeError):
    pass


def build_local_windows(
    *,
    source_path: Path | str,
    output_path: Path | str,
    dataset_key: str,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    dataset_key = _validate_dataset_key(dataset_key)
    _validate_window_args(
        start_index=start_index,
        sample_limit=sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
    )
    if not source.exists():
        raise LocalWindowError(f"source path does not exist: {source}")
    if output.exists() and not force:
        raise LocalWindowError(f"output path already exists: {output}")

    h5py = _import_h5py()
    with h5py.File(source, "r") as handle:
        if dataset_key not in handle:
            raise LocalWindowError(f"missing dataset key at HDF5 root: {dataset_key}")
        dataset = handle[dataset_key]
        if not isinstance(dataset, h5py.Dataset):
            raise LocalWindowError(f"HDF5 key is not a dataset: {dataset_key}")
        raw_shape = [int(axis) for axis in dataset.shape]
        raw_dtype = str(dataset.dtype)
        _validate_dataset(dataset_key, dataset.dtype, raw_shape)

        stop_index = _sample_stop_index(
            start_index=start_index,
            sample_limit=sample_limit,
            sample_count=raw_shape[0],
        )
        actual_sample_count = stop_index - start_index
        reduced_time = _reduced_length(raw_shape[1], time_reduction)
        reduced_space = _reduced_length(raw_shape[2], space_reduction)
        required_steps = context_steps + forecast_steps
        if reduced_time < required_steps:
            raise LocalWindowError(
                "reduced time dimension is shorter than "
                "context_steps + forecast_steps"
            )
        reduced = np.asarray(
            dataset[
                start_index:stop_index,
                ::time_reduction,
                ::space_reduction,
            ],
            dtype=np.float32,
        )

    context = reduced[:, :context_steps, :]
    target = reduced[:, context_steps:required_steps, :]
    full_reference = reduced[:, :required_steps, :]
    metadata = {
        "source_path": str(source),
        "dataset_key": dataset_key,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
        "reduced_shape": [actual_sample_count, reduced_time, reduced_space],
        "start_index": start_index,
        "sample_limit": sample_limit,
        "actual_sample_count": actual_sample_count,
        "context_steps": context_steps,
        "forecast_steps": forecast_steps,
        "time_reduction": time_reduction,
        "space_reduction": space_reduction,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        context=context,
        target=target,
        full_reference=full_reference,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )

    return {
        "output_path": str(output),
        "context_shape": [int(axis) for axis in context.shape],
        "target_shape": [int(axis) for axis in target.shape],
        "full_reference_shape": [int(axis) for axis in full_reference.shape],
        "metadata": metadata,
    }


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise LocalWindowError(
            "h5py is required for HDF5 window building; run `uv sync --extra data`"
        ) from exc
    return h5py


def _validate_dataset_key(dataset_key: str) -> str:
    if not dataset_key:
        raise LocalWindowError("dataset_key must be non-empty")
    return dataset_key


def _validate_window_args(
    *,
    start_index: int,
    sample_limit: int | None,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
) -> None:
    if start_index < 0:
        raise LocalWindowError("start_index must be >= 0")
    if sample_limit is not None and sample_limit <= 0:
        raise LocalWindowError("sample_limit must be > 0")
    for name, value in (
        ("context_steps", context_steps),
        ("forecast_steps", forecast_steps),
        ("time_reduction", time_reduction),
        ("space_reduction", space_reduction),
    ):
        if value <= 0:
            raise LocalWindowError(f"{name} must be > 0")


def _validate_dataset(dataset_key: str, dtype: np.dtype[Any], shape: list[int]) -> None:
    if len(shape) != 3:
        raise LocalWindowError(f"HDF5 dataset {dataset_key!r} must be rank 3")
    if not np.issubdtype(dtype, np.number) or np.issubdtype(
        dtype, np.complexfloating
    ):
        raise LocalWindowError(f"HDF5 dataset {dataset_key!r} must be real numeric")


def _sample_stop_index(
    *,
    start_index: int,
    sample_limit: int | None,
    sample_count: int,
) -> int:
    if start_index >= sample_count:
        raise LocalWindowError("start_index is outside sample count")
    stop_index = sample_count if sample_limit is None else start_index + sample_limit
    stop_index = min(stop_index, sample_count)
    if stop_index <= start_index:
        raise LocalWindowError("sample slice is empty")
    return stop_index


def _reduced_length(length: int, stride: int) -> int:
    if length == 0:
        return 0
    return ((length - 1) // stride) + 1
