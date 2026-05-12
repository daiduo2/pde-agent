"""Optional FNO backend helpers for local PDE agent loops."""


import json
import tarfile
from dataclasses import dataclass
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

    loaded = _load_checkpoint_with_metadata(checkpoint_file, torch=torch)

    metadata = _inspect_loaded_checkpoint(
        checkpoint=loaded["checkpoint"],
        checkpoint_path=checkpoint_file,
        torch=torch,
    )
    if loaded["archive_member"] is not None:
        metadata["archive_member"] = loaded["archive_member"]
    metadata["load_warnings"].extend(loaded["load_warnings"])
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
    loaded = _load_checkpoint_with_metadata(checkpoint_file, torch=torch)
    checkpoint = loaded["checkpoint"]
    if metadata_path is not None:
        _require_file(Path(metadata_path), "metadata")

    prediction, metadata = _predict_checkpoint(
        checkpoint=checkpoint,
        context=context,
        forecast_steps=target.shape[1],
        checkpoint_file=checkpoint_file,
        archive_member=loaded["archive_member"],
        torch=torch,
    )
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


@dataclass(frozen=True)
class PDEBenchFNO1dConfig:
    history_steps: int
    width: int
    modes: int
    fc_hidden: int
    layers: int = 4


def _extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _looks_like_pdebench_fno1d(checkpoint: Any, *, torch: Any) -> bool:
    state = _extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        return False
    if not all(
        key in state and torch.is_tensor(state[key])
        for key in _pdebench_fno1d_required_keys()
    ):
        return False
    try:
        _infer_pdebench_fno1d_config(state, torch=torch)
    except FNOBackendError:
        return False
    return True


def _pdebench_fno1d_required_keys() -> tuple[str, ...]:
    keys: list[str] = [
        "fc0.weight",
        "fc0.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    ]
    for index in range(4):
        keys.extend(
            [
                f"conv{index}.weights1",
                f"w{index}.weight",
                f"w{index}.bias",
            ]
        )
    return tuple(keys)


def _infer_pdebench_fno1d_config(
    state: Any,
    *,
    torch: Any | None = None,
) -> PDEBenchFNO1dConfig:
    if not isinstance(state, dict):
        raise FNOBackendError("unsupported PDEBench FNO1d state dict: not a dict")
    missing = [key for key in _pdebench_fno1d_required_keys() if key not in state]
    if missing:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: missing keys "
            + ", ".join(missing)
        )
    if torch is not None:
        non_tensors = [
            key
            for key in _pdebench_fno1d_required_keys()
            if not torch.is_tensor(state[key])
        ]
    else:
        non_tensors = [
            key
            for key in _pdebench_fno1d_required_keys()
            if not hasattr(state[key], "shape") or not hasattr(state[key], "ndim")
        ]
    if non_tensors:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: non-tensor values "
            + ", ".join(non_tensors)
        )

    fc0_weight = state["fc0.weight"]
    fc1_weight = state["fc1.weight"]
    fc2_weight = state["fc2.weight"]
    conv0_weight = state["conv0.weights1"]
    if fc0_weight.ndim != 2:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: fc0.weight must be rank 2"
        )
    if conv0_weight.ndim != 3:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: conv0.weights1 must be rank 3"
        )
    width = int(fc0_weight.shape[0])
    history_steps = int(fc0_weight.shape[1]) - 1
    modes = int(conv0_weight.shape[-1])
    if fc1_weight.ndim != 2 or fc2_weight.ndim != 2:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: fc weights must be rank 2"
        )
    fc_hidden = int(fc1_weight.shape[0])
    config = PDEBenchFNO1dConfig(
        history_steps=history_steps,
        width=width,
        modes=modes,
        fc_hidden=fc_hidden,
    )
    _validate_pdebench_fno1d_shapes(state=state, config=config)
    return config


def _validate_pdebench_fno1d_shapes(
    *,
    state: dict[str, Any],
    config: PDEBenchFNO1dConfig,
) -> None:
    if config.history_steps < 1:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: history_steps must be >= 1"
        )
    if config.width < 1:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: width must be >= 1"
        )
    if config.modes < 1:
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: modes must be >= 1"
        )
    if tuple(state["fc0.weight"].shape) != (config.width, config.history_steps + 1):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc0.weight shape"
        )
    if tuple(state["fc0.bias"].shape) != (config.width,):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc0.bias shape"
        )
    for index in range(config.layers):
        if tuple(state[f"conv{index}.weights1"].shape) != (
            config.width,
            config.width,
            config.modes,
        ):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid conv{index}.weights1 shape"
            )
        if tuple(state[f"w{index}.weight"].shape) != (config.width, config.width, 1):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid w{index}.weight shape"
            )
        if tuple(state[f"w{index}.bias"].shape) != (config.width,):
            raise FNOBackendError(
                f"unsupported PDEBench FNO1d state dict: invalid w{index}.bias shape"
            )
    if tuple(state["fc1.weight"].shape) != (config.fc_hidden, config.width):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc1.weight shape"
        )
    if tuple(state["fc1.bias"].shape) != (config.fc_hidden,):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc1.bias shape"
        )
    if tuple(state["fc2.weight"].shape) != (1, config.fc_hidden):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc2.weight shape"
        )
    if tuple(state["fc2.bias"].shape) != (1,):
        raise FNOBackendError(
            "unsupported PDEBench FNO1d state dict: invalid fc2.bias shape"
        )


def _make_spectral_conv1d(torch: Any) -> type:
    class SpectralConv1d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            self.weights1 = torch.nn.Parameter(
                scale
                * torch.randn(
                    in_channels,
                    out_channels,
                    modes,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x: Any) -> Any:
            batch_size = x.shape[0]
            space_points = x.shape[-1]
            x_ft = torch.fft.rfft(x)
            if self.modes > x_ft.shape[-1]:
                raise FNOBackendError(
                    "PDEBench FNO1d modes exceed available Fourier modes for input grid"
                )
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                x_ft.shape[-1],
                device=x.device,
                dtype=torch.cfloat,
            )
            out_ft[:, :, : self.modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, : self.modes],
                self.weights1,
            )
            return torch.fft.irfft(out_ft, n=space_points)

    return SpectralConv1d


def _make_pdebench_fno1d_class(torch: Any) -> type:
    SpectralConv1d = _make_spectral_conv1d(torch)

    class PDEBenchFNO1d(torch.nn.Module):
        def __init__(self, config: PDEBenchFNO1dConfig) -> None:
            super().__init__()
            self.config = config
            self.fc0 = torch.nn.Linear(config.history_steps + 1, config.width)
            for index in range(config.layers):
                setattr(
                    self,
                    f"conv{index}",
                    SpectralConv1d(config.width, config.width, config.modes),
                )
                setattr(
                    self,
                    f"w{index}",
                    torch.nn.Conv1d(config.width, config.width, 1),
                )
            self.fc1 = torch.nn.Linear(config.width, config.fc_hidden)
            self.fc2 = torch.nn.Linear(config.fc_hidden, 1)

        def forward(self, x: Any) -> Any:
            x = self.fc0(x)
            x = x.permute(0, 2, 1)
            for index in range(self.config.layers):
                x1 = getattr(self, f"conv{index}")(x)
                x2 = getattr(self, f"w{index}")(x)
                x = x1 + x2
                if index != self.config.layers - 1:
                    x = torch.nn.functional.gelu(x)
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.gelu(self.fc1(x))
            x = self.fc2(x)
            return x.squeeze(-1)

    return PDEBenchFNO1d


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
    return _load_checkpoint_with_metadata(path, torch=torch)["checkpoint"]


def _load_checkpoint_with_metadata(path: Path, *, torch: Any) -> dict[str, Any]:
    if tarfile.is_tarfile(path):
        return _load_checkpoint_from_tar(path, torch=torch)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FNOBackendError(f"failed to load FNO checkpoint {path}: {exc}") from exc
    return {
        "checkpoint": checkpoint,
        "archive_member": None,
        "load_warnings": [],
    }


def _load_checkpoint_from_tar(path: Path, *, torch: Any) -> dict[str, Any]:
    selected_member_name = "<none>"
    try:
        with tarfile.open(path) as archive:
            checkpoint_members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(".pt")
                ),
                key=lambda member: member.name,
            )
            if not checkpoint_members:
                raise FNOBackendError(
                    f"unsupported FNO checkpoint archive {path}: "
                    "no torch checkpoint member"
                )
            selected_member = checkpoint_members[0]
            selected_member_name = selected_member.name
            extracted = archive.extractfile(selected_member)
            if extracted is None:
                raise FNOBackendError(
                    f"unsupported FNO checkpoint archive {path}: "
                    f"cannot read member {selected_member.name}"
                )
            checkpoint = torch.load(
                extracted,
                map_location="cpu",
                weights_only=False,
            )
    except FNOBackendError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise FNOBackendError(f"failed to load FNO checkpoint {path}: {exc}") from exc
    except Exception as exc:
        raise FNOBackendError(
            f"failed to load FNO checkpoint {path} member {selected_member_name}: {exc}"
        ) from exc

    load_warnings: list[str] = []
    if len(checkpoint_members) > 1:
        load_warnings.append(
            "multiple torch checkpoint members found in archive; "
            f"selected {selected_member.name}"
        )
    return {
        "checkpoint": checkpoint,
        "archive_member": selected_member.name,
        "load_warnings": load_warnings,
    }


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


def _predict_checkpoint(
    *,
    checkpoint: Any,
    context: np.ndarray,
    forecast_steps: int,
    checkpoint_file: Path,
    archive_member: str | None,
    torch: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(checkpoint, dict) and checkpoint.get("model_type") == "fno_v0_fixture":
        prediction_rule = _fixture_prediction_rule(checkpoint)
        prediction = _predict_fixture_rule(
            context=context,
            forecast_steps=forecast_steps,
            prediction_rule=prediction_rule,
        )
        metadata = {
            "model_type": "fno",
            "adapter_type": "fixture_delta_extrapolation",
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_format": "torch_serialized",
            "prediction_rule": prediction_rule,
            "sample_count": int(prediction.shape[0]),
            "forecast_steps": int(prediction.shape[1]),
            "space_points": int(prediction.shape[2]),
        }
        if archive_member is not None:
            metadata["archive_member"] = archive_member
        return prediction, metadata

    if _looks_like_pdebench_fno1d(checkpoint, torch=torch):
        return _predict_pdebench_fno1d_checkpoint(
            checkpoint=checkpoint,
            context=context,
            forecast_steps=forecast_steps,
            checkpoint_file=checkpoint_file,
            archive_member=archive_member,
            torch=torch,
        )

    raise FNOBackendError("unsupported FNO checkpoint schema")


def _predict_pdebench_fno1d_checkpoint(
    *,
    checkpoint: Any,
    context: np.ndarray,
    forecast_steps: int,
    checkpoint_file: Path,
    archive_member: str | None,
    torch: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    state = _extract_state_dict(checkpoint)
    config = _infer_pdebench_fno1d_config(state, torch=torch)
    prediction = _rollout_pdebench_fno1d(
        state=state,
        config=config,
        context=context,
        forecast_steps=forecast_steps,
        torch=torch,
        device="cpu",
    )
    metadata: dict[str, Any] = {
        "model_type": "fno",
        "adapter_type": "pdebench_fno1d",
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_format": "torch_serialized",
        "history_steps": config.history_steps,
        "modes": config.modes,
        "width": config.width,
        "fc_hidden": config.fc_hidden,
        "rollout_steps": int(forecast_steps),
        "device": "cpu",
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
    }
    if archive_member is not None:
        metadata["archive_member"] = archive_member
    return prediction, metadata


def _rollout_pdebench_fno1d(
    *,
    state: dict[str, Any],
    config: PDEBenchFNO1dConfig,
    context: np.ndarray,
    forecast_steps: int,
    torch: Any,
    device: str,
) -> np.ndarray:
    if context.shape[1] < config.history_steps:
        raise FNOBackendError(
            "context_steps must be >= official FNO history steps "
            f"({context.shape[1]} < {config.history_steps})"
        )
    available_modes = (context.shape[2] // 2) + 1
    if config.modes > available_modes:
        raise FNOBackendError(
            "PDEBench FNO1d modes exceed available Fourier modes for input grid "
            f"({config.modes} > {available_modes})"
        )

    PDEBenchFNO1d = _make_pdebench_fno1d_class(torch)
    model = PDEBenchFNO1d(config).to(device)
    model.load_state_dict(state)
    model.eval()
    history = torch.as_tensor(
        context[:, -config.history_steps :, :],
        dtype=torch.float32,
        device=device,
    )
    grid = torch.linspace(
        0.0,
        1.0,
        context.shape[2],
        dtype=torch.float32,
        device=device,
    ).view(1, context.shape[2], 1)
    grid = grid.repeat(context.shape[0], 1, 1)

    predictions: list[Any] = []
    with torch.no_grad():
        for _ in range(forecast_steps):
            model_input = torch.cat([history.permute(0, 2, 1), grid], dim=-1)
            next_step = model(model_input)
            if not torch.isfinite(next_step).all():
                raise FNOBackendError("prediction contains non-finite values")
            predictions.append(next_step)
            history = torch.cat([history[:, 1:, :], next_step.unsqueeze(1)], dim=1)
    prediction_tensor = torch.stack(predictions, dim=1)
    prediction = prediction_tensor.detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise FNOBackendError("prediction contains non-finite values")
    return prediction


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
