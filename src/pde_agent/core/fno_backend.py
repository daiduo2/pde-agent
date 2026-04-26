"""Optional FNO backend helpers for local PDE agent loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class FNOBackendError(RuntimeError):
    """Raised when optional FNO backend operations cannot proceed."""


def inspect_fno_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    torch = _require_torch()
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise FNOBackendError(
            f"failed to load FNO checkpoint {checkpoint_file}: {exc}"
        ) from exc

    metadata = _inspect_loaded_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_file,
        torch=torch,
    )
    if output_path is not None:
        output_file = Path(output_path)
        metadata["metadata_path"] = str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return metadata


def predict_fno_local(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise FNOBackendError(f"prediction output already exists: {output_file}")

    context, target = _load_windows(windows_file)
    checkpoint = _load_checkpoint(checkpoint_file, torch=torch)
    if metadata_path is not None:
        _require_file(Path(metadata_path), "metadata")

    prediction_rule = _fixture_prediction_rule(checkpoint)
    prediction = _predict_fixture_rule(
        context=context,
        forecast_steps=target.shape[1],
        prediction_rule=prediction_rule,
    )
    metadata = {
        "model_type": "fno",
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_format": "torch_serialized",
        "prediction_rule": prediction_rule,
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    return {
        "prediction_path": str(output_file),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **metadata,
    }


def _inspect_loaded_checkpoint(
    *,
    checkpoint: Any,
    checkpoint_path: Path,
    torch: Any,
) -> dict[str, Any]:
    top_level_keys: list[str] = []
    has_model_state = False
    has_config = False
    tensor_key_count = 0
    parameter_count = 0
    load_warnings: list[str] = []

    if isinstance(checkpoint, dict):
        top_level_keys = sorted(str(key) for key in checkpoint.keys())
        has_model_state = any(
            key in checkpoint for key in ("model_state_dict", "state_dict", "model")
        )
        has_config = any(key in checkpoint for key in ("config", "args", "model_config"))
        tensors = _collect_tensors(checkpoint, torch=torch)
        tensor_key_count = len(tensors)
        parameter_count = int(sum(tensor.numel() for tensor in tensors))
    else:
        load_warnings.append(
            f"unsupported top-level checkpoint object: {type(checkpoint).__name__}"
        )

    return {
        "schema_version": 1,
        "state_hint": "FNO_CHECKPOINT_INSPECTED",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": "torch_serialized",
        "top_level_keys": top_level_keys,
        "has_model_state": has_model_state,
        "has_config": has_config,
        "tensor_summary": {
            "parameter_count": parameter_count,
            "tensor_key_count": tensor_key_count,
        },
        "load_warnings": load_warnings,
    }


def _load_checkpoint(path: Path, *, torch: Any) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FNOBackendError(f"failed to load FNO checkpoint {path}: {exc}") from exc


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "context" not in archive or "target" not in archive:
                raise FNOBackendError("windows NPZ must include context and target")
            context = np.asarray(archive["context"], dtype=np.float32)
            target = np.asarray(archive["target"], dtype=np.float32)
    except FNOBackendError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise FNOBackendError(f"failed to load windows NPZ {path}: {exc}") from exc
    _validate_windows(context=context, target=target)
    return context, target


def _validate_windows(*, context: np.ndarray, target: np.ndarray) -> None:
    if context.ndim != 3 or target.ndim != 3:
        raise FNOBackendError("context and target must be rank-3 arrays [N, T, X]")
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise FNOBackendError("context and target must share sample and space axes")
    if context.shape[1] < 2:
        raise FNOBackendError("context must include at least two time steps")
    if target.shape[1] < 1:
        raise FNOBackendError("target must include at least one forecast step")
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(target)):
        raise FNOBackendError("context and target must contain finite values")


def _fixture_prediction_rule(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        raise FNOBackendError("unsupported FNO checkpoint schema")
    if checkpoint.get("model_type") != "fno_v0_fixture":
        raise FNOBackendError("unsupported FNO checkpoint schema")
    prediction_rule = checkpoint.get("prediction_rule")
    if prediction_rule != "delta_extrapolation":
        raise FNOBackendError(f"unsupported FNO prediction rule: {prediction_rule}")
    return prediction_rule


def _predict_fixture_rule(
    *,
    context: np.ndarray,
    forecast_steps: int,
    prediction_rule: str,
) -> np.ndarray:
    if prediction_rule != "delta_extrapolation":
        raise FNOBackendError(f"unsupported FNO prediction rule: {prediction_rule}")
    last = context[:, -1:, :]
    delta = context[:, -1:, :] - context[:, -2:-1, :]
    steps = np.arange(1, forecast_steps + 1, dtype=np.float32).reshape(1, -1, 1)
    prediction = last + steps * delta
    if not np.all(np.isfinite(prediction)):
        raise FNOBackendError("prediction contains non-finite values")
    return prediction.astype(np.float32)


def _collect_tensors(
    value: Any,
    *,
    torch: Any,
    _visited: set[int] | None = None,
) -> list[Any]:
    if _visited is None:
        _visited = set()

    tensors: list[Any] = []
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _visited:
            return tensors
        _visited.add(value_id)
        for child in value.values():
            tensors.extend(_collect_tensors(child, torch=torch, _visited=_visited))
    elif isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in _visited:
            return tensors
        _visited.add(value_id)
        for child in value:
            tensors.extend(_collect_tensors(child, torch=torch, _visited=_visited))
    return tensors


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise FNOBackendError(
            "FNO backend requires the optional torch extra; run with --extra torch"
        ) from exc
    return torch


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FNOBackendError(f"missing {label} path: {path}")
    if not path.is_file():
        raise FNOBackendError(f"{label} path is not a file: {path}")
    return path
