
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pde_agent.core.array_contract import (
    ArrayContractError,
    BurgersArrayBundle,
    validate_burgers_bundle,
)
from pde_agent.core.data_config import DataConfig


class DataInspectionError(RuntimeError):
    pass


def inspect_data_source(config: DataConfig) -> dict[str, object]:
    _validate_positive_config(config)
    if config.mode == "synthetic":
        return _inspection_result(
            source_mode="synthetic",
            source_path=None,
            dataset_key=None,
            available_keys=[],
            raw_shape=None,
            raw_dtype=None,
            reduced_shape=None,
            contract_shape={
                "context": (None, config.context_steps, None),
                "prediction": (None, config.forecast_steps, None),
                "target": (None, config.forecast_steps, None),
            },
            sample_count=None,
            context_time=config.context_steps,
            forecast_time=config.forecast_steps,
            space_points=None,
            warnings=["synthetic mode does not inspect external array data"],
        )
    if config.mode == "npz_fixture":
        source_path = _require_source_path(config)
        _require_existing_path(source_path)
        with np.load(source_path, allow_pickle=False) as archive:
            missing = _missing_keys(archive.files, ["context", "prediction", "target"])
            if missing:
                raise DataInspectionError(
                    f"NPZ fixture missing keys: {', '.join(missing)}"
                )
            available_keys = tuple(archive.files)
        bundle = load_array_bundle(config)
        return _inspect_validated_bundle(
            bundle,
            source_mode="npz_fixture",
            source_path=str(source_path),
            dataset_key=None,
            available_keys=available_keys,
        )
    if config.mode == "pdebench_hdf5":
        return _inspect_hdf5(config)
    raise DataInspectionError(f"unsupported data mode: {config.mode}")


def load_array_bundle(config: DataConfig) -> BurgersArrayBundle:
    _validate_positive_config(config)
    if config.mode == "synthetic":
        raise DataInspectionError("synthetic mode does not load array bundles")
    if config.mode == "npz_fixture":
        return _load_npz_bundle(config)
    if config.mode == "pdebench_hdf5":
        return _load_hdf5_bundle(config)
    raise DataInspectionError(f"unsupported data mode: {config.mode}")


def _load_npz_bundle(config: DataConfig) -> BurgersArrayBundle:
    source_path = _require_source_path(config)
    _require_existing_path(source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        missing = _missing_keys(archive.files, ["context", "prediction", "target"])
        if missing:
            raise DataInspectionError(f"NPZ fixture missing keys: {', '.join(missing)}")
        metadata = _base_metadata(config)
        if "metadata_json" in archive.files:
            metadata.update(_load_metadata_json(archive["metadata_json"]))
        bundle = BurgersArrayBundle(
            context=archive["context"],
            prediction=archive["prediction"],
            target=archive["target"],
            full_reference=archive["full_reference"]
            if "full_reference" in archive.files
            else None,
            metadata=metadata,
        )
    return _validate_adapter_bundle(bundle)


def _load_hdf5_bundle(config: DataConfig) -> BurgersArrayBundle:
    info = _inspect_hdf5(config)
    h5py = _import_h5py()
    source_path = _require_source_path(config)
    dataset_key = str(info["dataset_key"])
    with h5py.File(source_path, "r") as handle:
        dataset = handle[dataset_key]
        sample_slice = (
            slice(0, config.sample_limit) if config.sample_limit else slice(None)
        )
        reduced = np.asarray(
            dataset[
                sample_slice,
                :: config.time_reduction,
                :: config.space_reduction,
            ]
        )
        hdf5_metadata = _hdf5_metadata(handle.attrs, dataset.attrs)
    required_time = config.context_steps + config.forecast_steps
    if reduced.shape[1] < required_time:
        raise DataInspectionError(
            "reduced HDF5 time dimension is shorter than context_steps + forecast_steps"
        )
    context = reduced[:, : config.context_steps, :]
    target = reduced[
        :,
        config.context_steps : config.context_steps + config.forecast_steps,
        :,
    ]
    metadata = _base_metadata(config)
    metadata.update(
        {
            "dataset_key": dataset_key,
            **hdf5_metadata,
            "time_reduction": config.time_reduction,
            "space_reduction": config.space_reduction,
            "sample_count": target.shape[0],
            "context_time": context.shape[1],
            "forecast_time": target.shape[1],
            "space_points": target.shape[2],
        }
    )
    return _validate_adapter_bundle(
        BurgersArrayBundle(
            context=context,
            prediction=target.copy(),
            target=target,
            full_reference=reduced,
            metadata=metadata,
        )
    )


def _validate_adapter_bundle(bundle: BurgersArrayBundle) -> BurgersArrayBundle:
    try:
        return validate_burgers_bundle(bundle)
    except ArrayContractError as exc:
        raise DataInspectionError(str(exc)) from exc


def _inspect_hdf5(config: DataConfig) -> dict[str, object]:
    h5py = _import_h5py()
    source_path = _require_source_path(config)
    _require_existing_path(source_path)
    with h5py.File(source_path, "r") as handle:
        dataset_keys = _list_hdf5_dataset_keys(handle)
        dataset_key = _select_dataset_key(dataset_keys, config.dataset_key)
        dataset = handle[dataset_key]
        shape = tuple(int(axis) for axis in dataset.shape)
        dtype = str(dataset.dtype)
    if len(shape) != 3:
        raise DataInspectionError(f"HDF5 dataset {dataset_key!r} must be rank 3")
    if not np.issubdtype(np.dtype(dtype), np.number) or np.issubdtype(
        np.dtype(dtype), np.complexfloating
    ):
        raise DataInspectionError(f"HDF5 dataset {dataset_key!r} must be real-valued")
    sample_count = shape[0]
    if config.sample_limit is not None:
        sample_count = min(sample_count, config.sample_limit)
    reduced_shape = (
        sample_count,
        _reduced_length(shape[1], config.time_reduction),
        _reduced_length(shape[2], config.space_reduction),
    )
    required_time = config.context_steps + config.forecast_steps
    if reduced_shape[1] < required_time:
        raise DataInspectionError(
            "reduced HDF5 time dimension is shorter than context_steps + forecast_steps"
        )
    return _inspection_result(
        source_mode="pdebench_hdf5",
        source_path=str(source_path),
        dataset_key=dataset_key,
        available_keys=tuple(dataset_keys),
        raw_shape=shape,
        raw_dtype=dtype,
        reduced_shape=reduced_shape,
        contract_shape={
            "context": (
                reduced_shape[0],
                config.context_steps,
                reduced_shape[2],
            ),
            "prediction": (
                reduced_shape[0],
                config.forecast_steps,
                reduced_shape[2],
            ),
            "target": (
                reduced_shape[0],
                config.forecast_steps,
                reduced_shape[2],
            ),
        },
        sample_count=reduced_shape[0],
        context_time=config.context_steps,
        forecast_time=config.forecast_steps,
        space_points=reduced_shape[2],
        warnings=[],
    )


def _inspection_result(
    *,
    source_mode: str,
    source_path: str | None,
    dataset_key: str | None,
    available_keys: list[str] | tuple[str, ...],
    raw_shape: tuple[int, ...] | None,
    raw_dtype: str | None,
    reduced_shape: tuple[int, ...] | None,
    contract_shape: dict[str, tuple[int | None, ...]],
    sample_count: int | None,
    context_time: int,
    forecast_time: int,
    space_points: int | None,
    warnings: list[str],
) -> dict[str, object]:
    return {
        "source_mode": source_mode,
        "source_path": source_path,
        "dataset_key": dataset_key,
        "available_keys": available_keys,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
        "reduced_shape": reduced_shape,
        "contract_shape": contract_shape,
        "sample_count": sample_count,
        "context_time": context_time,
        "forecast_time": forecast_time,
        "space_points": space_points,
        "is_compatible": True,
        "warnings": warnings,
    }


def _inspect_validated_bundle(
    bundle: BurgersArrayBundle,
    *,
    source_mode: str,
    source_path: str | None,
    dataset_key: str | None,
    available_keys: list[str] | tuple[str, ...],
) -> dict[str, object]:
    return _inspection_result(
        source_mode=source_mode,
        source_path=source_path,
        dataset_key=dataset_key,
        available_keys=available_keys,
        raw_shape=None,
        raw_dtype=str(bundle.target.dtype),
        reduced_shape=None,
        contract_shape={
            "context": tuple(bundle.context.shape),
            "prediction": tuple(bundle.prediction.shape),
            "target": tuple(bundle.target.shape),
        },
        sample_count=bundle.metadata["sample_count"],
        context_time=bundle.metadata["context_time"],
        forecast_time=bundle.metadata["forecast_time"],
        space_points=bundle.metadata["space_points"],
        warnings=[],
    )


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise DataInspectionError("h5py is required for pdebench_hdf5 mode") from exc
    return h5py


def _list_hdf5_dataset_keys(handle: Any) -> list[str]:
    h5py = _import_h5py()
    dataset_keys: list[str] = []

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            dataset_keys.append(name)

    handle.visititems(collect)
    return sorted(dataset_keys)


def _select_dataset_key(dataset_keys: list[str], configured_key: str | None) -> str:
    if configured_key is not None:
        if configured_key not in dataset_keys:
            raise DataInspectionError(f"HDF5 dataset key not found: {configured_key}")
        return configured_key
    if not dataset_keys:
        raise DataInspectionError("HDF5 source contains no datasets")
    if len(dataset_keys) > 1:
        raise DataInspectionError(
            "ambiguous HDF5 dataset keys; configure dataset_key explicitly"
        )
    return dataset_keys[0]


def _base_metadata(config: DataConfig) -> dict[str, object]:
    return {
        "problem_id": config.problem_id,
        "source_mode": config.mode,
        "nu": None,
        "dt": None,
        "dx": None,
        "context_time": config.context_steps,
        "forecast_time": config.forecast_steps,
        "space_points": None,
        "time_reduction": config.time_reduction,
        "space_reduction": config.space_reduction,
        "sample_count": config.sample_limit,
    }


def _load_metadata_json(value: NDArray[Any]) -> dict[str, object]:
    raw = value.item() if value.shape == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise DataInspectionError("metadata_json must contain a JSON string")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise DataInspectionError("metadata_json must decode to an object")
    return dict(parsed)


def _hdf5_metadata(*attrs_sources: Any) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for attrs in attrs_sources:
        for key in ("nu", "dt", "dx"):
            if key in attrs:
                metadata[key] = _to_plain_metadata_value(attrs[key])
    return metadata


def _to_plain_metadata_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _validate_positive_config(config: DataConfig) -> None:
    if config.context_steps <= 0:
        raise DataInspectionError("context_steps must be positive")
    if config.forecast_steps <= 0:
        raise DataInspectionError("forecast_steps must be positive")
    if config.time_reduction <= 0:
        raise DataInspectionError("time_reduction must be positive")
    if config.space_reduction <= 0:
        raise DataInspectionError("space_reduction must be positive")
    if config.sample_limit is not None and config.sample_limit <= 0:
        raise DataInspectionError("sample_limit must be positive when configured")


def _require_source_path(config: DataConfig) -> Path:
    if config.source_path is None:
        raise DataInspectionError("source_path is required")
    return config.source_path


def _require_existing_path(source_path: Path) -> None:
    if not source_path.exists():
        raise DataInspectionError(f"source_path does not exist: {source_path}")
    if not source_path.is_file():
        raise DataInspectionError(f"source_path is not a file: {source_path}")


def _missing_keys(keys: list[str], required: list[str]) -> list[str]:
    available = set(keys)
    return [key for key in required if key not in available]


def _reduced_length(length: int, stride: int) -> int:
    return (length + stride - 1) // stride
