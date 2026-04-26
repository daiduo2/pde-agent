"""Optional FNO backend helpers for local PDE agent loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def _collect_tensors(value: Any, *, torch: Any) -> list[Any]:
    tensors: list[Any] = []
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        for child in value.values():
            tensors.extend(_collect_tensors(child, torch=torch))
    elif isinstance(value, (list, tuple)):
        for child in value:
            tensors.extend(_collect_tensors(child, torch=torch))
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
