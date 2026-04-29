"""Optional official UNet backend helpers for local PDE agent loops."""

from __future__ import annotations

import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class UNetBackendError(RuntimeError):
    """Raised when optional UNet backend operations cannot proceed."""


def inspect_unet_checkpoint(
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


def predict_unet_local(
    *,
    windows_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    force: bool = False,
    inference_batch_size: int | None = None,
    torch_num_threads: int | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    _validate_inference_options(
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
    )
    windows_file = _require_file(Path(windows_path), "windows")
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise UNetBackendError(f"prediction output already exists: {output_file}")
    if metadata_path is not None:
        _require_file(Path(metadata_path), "metadata")

    phase_start = time.perf_counter()
    context, target = _load_windows(windows_file)
    load_windows_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    loaded = _load_checkpoint_with_metadata(checkpoint_file, torch=torch)
    load_checkpoint_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    prediction, metadata = _predict_checkpoint(
        checkpoint=loaded["checkpoint"],
        context=context,
        forecast_steps=target.shape[1],
        checkpoint_file=checkpoint_file,
        archive_member=loaded["archive_member"],
        torch=torch,
        inference_batch_size=inference_batch_size,
        torch_num_threads=torch_num_threads,
    )
    predict_seconds = time.perf_counter() - phase_start
    phase_timings = dict(metadata.get("phase_timings", {}))
    phase_timings.update(
        {
            "load_windows_seconds": load_windows_seconds,
            "load_checkpoint_seconds": load_checkpoint_seconds,
            "predict_checkpoint_seconds": predict_seconds,
            "write_output_seconds": 0.0,
        }
    )
    metadata["phase_timings"] = phase_timings

    phase_start = time.perf_counter()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    write_output_seconds = time.perf_counter() - phase_start
    metadata["phase_timings"]["write_output_seconds"] = write_output_seconds
    return {
        "prediction_path": str(output_file),
        "prediction_shape": [int(axis) for axis in prediction.shape],
        **metadata,
    }


def _validate_inference_options(
    *,
    inference_batch_size: int | None,
    torch_num_threads: int | None,
) -> None:
    if inference_batch_size is not None and inference_batch_size <= 0:
        raise UNetBackendError("inference_batch_size must be positive")
    if torch_num_threads is not None and torch_num_threads <= 0:
        raise UNetBackendError("torch_num_threads must be positive")


@dataclass(frozen=True)
class PDEBenchUNet1dConfig:
    history_steps: int
    base_channels: int
    depth: int = 4


def _extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _looks_like_pdebench_unet1d(checkpoint: Any, *, torch: Any) -> bool:
    state = _extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        return False
    if not all(
        key in state and torch.is_tensor(state[key])
        for key in _pdebench_unet1d_required_keys()
    ):
        return False
    try:
        _infer_pdebench_unet1d_config(state, torch=torch)
    except UNetBackendError:
        return False
    return True


def _pdebench_unet1d_required_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for module, prefix in (
        ("encoder1", "enc1"),
        ("encoder2", "enc2"),
        ("encoder3", "enc3"),
        ("encoder4", "enc4"),
        ("bottleneck", "bottleneck"),
        ("decoder4", "dec4"),
        ("decoder3", "dec3"),
        ("decoder2", "dec2"),
        ("decoder1", "dec1"),
    ):
        keys.extend(_conv_block_keys(module=module, prefix=prefix))
    for index in range(1, 5):
        keys.extend([f"upconv{index}.weight", f"upconv{index}.bias"])
    keys.extend(["conv.weight", "conv.bias"])
    return tuple(keys)


def _conv_block_keys(*, module: str, prefix: str) -> list[str]:
    keys: list[str] = []
    for conv_index in (1, 2):
        keys.extend(
            [
                f"{module}.{prefix}conv{conv_index}.weight",
                f"{module}.{prefix}norm{conv_index}.weight",
                f"{module}.{prefix}norm{conv_index}.bias",
                f"{module}.{prefix}norm{conv_index}.running_mean",
                f"{module}.{prefix}norm{conv_index}.running_var",
                f"{module}.{prefix}norm{conv_index}.num_batches_tracked",
            ]
        )
    return keys


def _infer_pdebench_unet1d_config(
    state: Any,
    *,
    torch: Any | None = None,
) -> PDEBenchUNet1dConfig:
    if not isinstance(state, dict):
        raise UNetBackendError("unsupported PDEBench UNet1d state dict: not a dict")
    required = _pdebench_unet1d_required_keys()
    missing = [key for key in required if key not in state]
    if missing:
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: missing keys "
            + ", ".join(missing)
        )
    if torch is not None:
        non_tensors = [key for key in required if not torch.is_tensor(state[key])]
    else:
        non_tensors = [
            key
            for key in required
            if not hasattr(state[key], "shape") or not hasattr(state[key], "ndim")
        ]
    if non_tensors:
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: non-tensor values "
            + ", ".join(non_tensors)
        )

    first_weight = state["encoder1.enc1conv1.weight"]
    if first_weight.ndim != 3:
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: encoder1.enc1conv1.weight must be rank 3"
        )
    config = PDEBenchUNet1dConfig(
        history_steps=int(first_weight.shape[1]),
        base_channels=int(first_weight.shape[0]),
    )
    _validate_pdebench_unet1d_shapes(state=state, config=config)
    return config


def _validate_pdebench_unet1d_shapes(
    *,
    state: dict[str, Any],
    config: PDEBenchUNet1dConfig,
) -> None:
    if config.history_steps < 1:
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: history_steps must be >= 1"
        )
    if config.base_channels < 1:
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: base_channels must be >= 1"
        )
    channels = [
        config.base_channels,
        config.base_channels * 2,
        config.base_channels * 4,
        config.base_channels * 8,
        config.base_channels * 16,
    ]
    _validate_conv_block_shapes(
        state=state,
        module="encoder1",
        prefix="enc1",
        in_channels=config.history_steps,
        out_channels=channels[0],
    )
    _validate_conv_block_shapes(
        state=state,
        module="encoder2",
        prefix="enc2",
        in_channels=channels[0],
        out_channels=channels[1],
    )
    _validate_conv_block_shapes(
        state=state,
        module="encoder3",
        prefix="enc3",
        in_channels=channels[1],
        out_channels=channels[2],
    )
    _validate_conv_block_shapes(
        state=state,
        module="encoder4",
        prefix="enc4",
        in_channels=channels[2],
        out_channels=channels[3],
    )
    _validate_conv_block_shapes(
        state=state,
        module="bottleneck",
        prefix="bottleneck",
        in_channels=channels[3],
        out_channels=channels[4],
    )
    for index, in_channels, out_channels in (
        (4, channels[4], channels[3]),
        (3, channels[3], channels[2]),
        (2, channels[2], channels[1]),
        (1, channels[1], channels[0]),
    ):
        if tuple(state[f"upconv{index}.weight"].shape) != (
            in_channels,
            out_channels,
            2,
        ):
            raise UNetBackendError(
                f"unsupported PDEBench UNet1d state dict: invalid upconv{index}.weight shape"
            )
        if tuple(state[f"upconv{index}.bias"].shape) != (out_channels,):
            raise UNetBackendError(
                f"unsupported PDEBench UNet1d state dict: invalid upconv{index}.bias shape"
            )
    _validate_conv_block_shapes(
        state=state,
        module="decoder4",
        prefix="dec4",
        in_channels=channels[4],
        out_channels=channels[3],
    )
    _validate_conv_block_shapes(
        state=state,
        module="decoder3",
        prefix="dec3",
        in_channels=channels[3],
        out_channels=channels[2],
    )
    _validate_conv_block_shapes(
        state=state,
        module="decoder2",
        prefix="dec2",
        in_channels=channels[2],
        out_channels=channels[1],
    )
    _validate_conv_block_shapes(
        state=state,
        module="decoder1",
        prefix="dec1",
        in_channels=channels[1],
        out_channels=channels[0],
    )
    if tuple(state["conv.weight"].shape) != (1, channels[0], 1):
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: invalid conv.weight shape"
        )
    if tuple(state["conv.bias"].shape) != (1,):
        raise UNetBackendError(
            "unsupported PDEBench UNet1d state dict: invalid conv.bias shape"
        )


def _validate_conv_block_shapes(
    *,
    state: dict[str, Any],
    module: str,
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> None:
    expected = {
        f"{module}.{prefix}conv1.weight": (out_channels, in_channels, 3),
        f"{module}.{prefix}norm1.weight": (out_channels,),
        f"{module}.{prefix}norm1.bias": (out_channels,),
        f"{module}.{prefix}norm1.running_mean": (out_channels,),
        f"{module}.{prefix}norm1.running_var": (out_channels,),
        f"{module}.{prefix}norm2.weight": (out_channels,),
        f"{module}.{prefix}norm2.bias": (out_channels,),
        f"{module}.{prefix}norm2.running_mean": (out_channels,),
        f"{module}.{prefix}norm2.running_var": (out_channels,),
        f"{module}.{prefix}conv2.weight": (out_channels, out_channels, 3),
    }
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise UNetBackendError(
                f"unsupported PDEBench UNet1d state dict: invalid {key} shape"
            )
    for key in (
        f"{module}.{prefix}norm1.num_batches_tracked",
        f"{module}.{prefix}norm2.num_batches_tracked",
    ):
        if tuple(state[key].shape) != ():
            raise UNetBackendError(
                f"unsupported PDEBench UNet1d state dict: invalid {key} shape"
            )


def _make_pdebench_unet1d_class(torch: Any) -> type:
    class ConvBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, prefix: str) -> None:
            super().__init__()
            setattr(
                self,
                f"{prefix}conv1",
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(self, f"{prefix}norm1", torch.nn.BatchNorm1d(out_channels))
            setattr(
                self,
                f"{prefix}conv2",
                torch.nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(self, f"{prefix}norm2", torch.nn.BatchNorm1d(out_channels))
            self.prefix = prefix

        def forward(self, x: Any) -> Any:
            conv1 = getattr(self, f"{self.prefix}conv1")
            norm1 = getattr(self, f"{self.prefix}norm1")
            conv2 = getattr(self, f"{self.prefix}conv2")
            norm2 = getattr(self, f"{self.prefix}norm2")
            x = torch.nn.functional.relu(norm1(conv1(x)))
            x = torch.nn.functional.relu(norm2(conv2(x)))
            return x

    class PDEBenchUNet1d(torch.nn.Module):
        def __init__(self, config: PDEBenchUNet1dConfig) -> None:
            super().__init__()
            self.config = config
            base = config.base_channels
            self.encoder1 = ConvBlock(config.history_steps, base, "enc1")
            self.encoder2 = ConvBlock(base, base * 2, "enc2")
            self.encoder3 = ConvBlock(base * 2, base * 4, "enc3")
            self.encoder4 = ConvBlock(base * 4, base * 8, "enc4")
            self.bottleneck = ConvBlock(base * 8, base * 16, "bottleneck")
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
            self.upconv4 = torch.nn.ConvTranspose1d(base * 16, base * 8, 2, 2)
            self.decoder4 = ConvBlock(base * 16, base * 8, "dec4")
            self.upconv3 = torch.nn.ConvTranspose1d(base * 8, base * 4, 2, 2)
            self.decoder3 = ConvBlock(base * 8, base * 4, "dec3")
            self.upconv2 = torch.nn.ConvTranspose1d(base * 4, base * 2, 2, 2)
            self.decoder2 = ConvBlock(base * 4, base * 2, "dec2")
            self.upconv1 = torch.nn.ConvTranspose1d(base * 2, base, 2, 2)
            self.decoder1 = ConvBlock(base * 2, base, "dec1")
            self.conv = torch.nn.Conv1d(base, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool(enc1))
            enc3 = self.encoder3(self.pool(enc2))
            enc4 = self.encoder4(self.pool(enc3))
            bottleneck = self.bottleneck(self.pool(enc4))
            dec4 = self._concat_skip(self.upconv4(bottleneck), enc4)
            dec4 = self.decoder4(dec4)
            dec3 = self._concat_skip(self.upconv3(dec4), enc3)
            dec3 = self.decoder3(dec3)
            dec2 = self._concat_skip(self.upconv2(dec3), enc2)
            dec2 = self.decoder2(dec2)
            dec1 = self._concat_skip(self.upconv1(dec2), enc1)
            dec1 = self.decoder1(dec1)
            return self.conv(dec1).squeeze(1)

        def _concat_skip(self, decoded: Any, skip: Any) -> Any:
            if decoded.shape[-1] != skip.shape[-1]:
                decoded = torch.nn.functional.interpolate(
                    decoded,
                    size=skip.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            return torch.cat((decoded, skip), dim=1)

    return PDEBenchUNet1d


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
    official_unet1d_detected = _looks_like_pdebench_unet1d(checkpoint, torch=torch)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": "UNET_CHECKPOINT_INSPECTED",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": "torch_serialized",
        "top_level_keys": top_level_keys,
        "has_model_state": has_model_state,
        "has_config": has_config,
        "tensor_summary": {
            "parameter_count": parameter_count,
            "tensor_key_count": tensor_key_count,
        },
        "official_unet1d_detected": official_unet1d_detected,
        "load_warnings": load_warnings,
    }
    if isinstance(checkpoint, dict):
        metadata["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())
        metadata["has_model_state"] = any(
            key in checkpoint for key in ("model_state_dict", "state_dict", "model")
        )
        metadata["has_config"] = any(
            key in checkpoint for key in ("config", "args", "model_config")
        )
        tensors = _collect_tensors(checkpoint, torch=torch)
        metadata["tensor_summary"] = {
            "parameter_count": int(sum(tensor.numel() for tensor in tensors)),
            "tensor_key_count": len(tensors),
        }
    else:
        load_warnings.append(
            f"unsupported top-level checkpoint object: {type(checkpoint).__name__}"
        )
    if official_unet1d_detected:
        config = _infer_pdebench_unet1d_config(
            _extract_state_dict(checkpoint),
            torch=torch,
        )
        metadata["history_steps"] = config.history_steps
        metadata["base_channels"] = config.base_channels
        metadata["depth"] = config.depth
    return metadata


def _load_checkpoint(path: Path, *, torch: Any) -> Any:
    return _load_checkpoint_with_metadata(path, torch=torch)["checkpoint"]


def _load_checkpoint_with_metadata(path: Path, *, torch: Any) -> dict[str, Any]:
    if tarfile.is_tarfile(path):
        return _load_checkpoint_from_tar(path, torch=torch)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise UNetBackendError(f"failed to load UNet checkpoint {path}: {exc}") from exc
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
                raise UNetBackendError(
                    f"unsupported UNet checkpoint archive {path}: "
                    "no torch checkpoint member"
                )
            selected_member = checkpoint_members[0]
            selected_member_name = selected_member.name
            extracted = archive.extractfile(selected_member)
            if extracted is None:
                raise UNetBackendError(
                    f"unsupported UNet checkpoint archive {path}: "
                    f"cannot read member {selected_member.name}"
                )
            checkpoint = torch.load(
                extracted,
                map_location="cpu",
                weights_only=False,
            )
    except UNetBackendError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise UNetBackendError(f"failed to load UNet checkpoint {path}: {exc}") from exc
    except Exception as exc:
        raise UNetBackendError(
            f"failed to load UNet checkpoint {path} member {selected_member_name}: {exc}"
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
                raise UNetBackendError("windows NPZ must include context and target")
            context = np.asarray(archive["context"], dtype=np.float32)
            target = np.asarray(archive["target"], dtype=np.float32)
    except UNetBackendError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise UNetBackendError(f"failed to load windows NPZ {path}: {exc}") from exc
    _validate_windows(context=context, target=target)
    return context, target


def _validate_windows(*, context: np.ndarray, target: np.ndarray) -> None:
    if context.ndim != 3 or target.ndim != 3:
        raise UNetBackendError("context and target must be rank-3 arrays [N, T, X]")
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise UNetBackendError("context and target must share sample and space axes")
    if context.shape[1] < 2:
        raise UNetBackendError("context must include at least two time steps")
    if target.shape[1] < 1:
        raise UNetBackendError("target must include at least one forecast step")
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(target)):
        raise UNetBackendError("context and target must contain finite values")


def _predict_checkpoint(
    *,
    checkpoint: Any,
    context: np.ndarray,
    forecast_steps: int,
    checkpoint_file: Path,
    archive_member: str | None,
    torch: Any,
    inference_batch_size: int | None,
    torch_num_threads: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not _looks_like_pdebench_unet1d(checkpoint, torch=torch):
        raise UNetBackendError("unsupported UNet checkpoint schema")
    state = _extract_state_dict(checkpoint)
    config = _infer_pdebench_unet1d_config(state, torch=torch)
    prediction, rollout_metadata = _with_torch_threads(
        torch=torch,
        torch_num_threads=torch_num_threads,
        operation=lambda: _rollout_pdebench_unet1d(
            state=state,
            config=config,
            context=context,
            forecast_steps=forecast_steps,
            torch=torch,
            device="cpu",
            inference_batch_size=inference_batch_size,
        ),
    )
    metadata: dict[str, Any] = {
        "model_type": "unet",
        "adapter_type": "pdebench_unet1d",
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_format": "torch_serialized",
        "history_steps": config.history_steps,
        "base_channels": config.base_channels,
        "depth": config.depth,
        "rollout_steps": int(forecast_steps),
        "device": "cpu",
        "inference_batch_size": int(inference_batch_size or context.shape[0]),
        "torch_num_threads": int(torch_num_threads or torch.get_num_threads()),
        "sample_count": int(prediction.shape[0]),
        "forecast_steps": int(prediction.shape[1]),
        "space_points": int(prediction.shape[2]),
        **rollout_metadata,
    }
    if archive_member is not None:
        metadata["archive_member"] = archive_member
    return prediction, metadata


def _rollout_pdebench_unet1d(
    *,
    state: dict[str, Any],
    config: PDEBenchUNet1dConfig,
    context: np.ndarray,
    forecast_steps: int,
    torch: Any,
    device: str,
    inference_batch_size: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if context.shape[1] < config.history_steps:
        raise UNetBackendError(
            "context_steps must be >= official UNet history steps "
            f"({context.shape[1]} < {config.history_steps})"
        )
    if context.shape[2] < 16:
        raise UNetBackendError(
            "space_points must be >= 16 for the official 4-level UNet adapter"
        )

    PDEBenchUNet1d = _make_pdebench_unet1d_class(torch)
    model = PDEBenchUNet1d(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    batch_size = int(inference_batch_size or context.shape[0])
    prediction = np.empty(
        (context.shape[0], forecast_steps, context.shape[2]),
        dtype=np.float32,
    )
    rollout_start = time.perf_counter()
    chunk_count = 0
    context_manager = getattr(torch, "inference_mode", torch.no_grad)
    with context_manager():
        for start in range(0, context.shape[0], batch_size):
            stop = min(start + batch_size, context.shape[0])
            chunk_prediction = _rollout_pdebench_unet1d_chunk(
                model=model,
                context=context[start:stop],
                history_steps=config.history_steps,
                forecast_steps=forecast_steps,
                torch=torch,
                device=device,
            )
            prediction[start:stop] = chunk_prediction
            chunk_count += 1
    rollout_seconds = time.perf_counter() - rollout_start
    if not np.all(np.isfinite(prediction)):
        raise UNetBackendError("prediction contains non-finite values")
    return prediction, {
        "rollout_chunk_count": chunk_count,
        "phase_timings": {
            "rollout_seconds": rollout_seconds,
        },
    }


def _rollout_pdebench_unet1d_chunk(
    *,
    model: Any,
    context: np.ndarray,
    history_steps: int,
    forecast_steps: int,
    torch: Any,
    device: str,
) -> np.ndarray:
    history = torch.as_tensor(
        context[:, -history_steps:, :],
        dtype=torch.float32,
        device=device,
    ).clone()
    prediction_tensor = torch.empty(
        (context.shape[0], forecast_steps, context.shape[2]),
        dtype=torch.float32,
        device=device,
    )
    for step in range(forecast_steps):
        next_step = model(history)
        if next_step.shape[-1] != context.shape[2]:
            next_step = torch.nn.functional.interpolate(
                next_step.unsqueeze(1),
                size=context.shape[2],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        if not torch.isfinite(next_step).all():
            raise UNetBackendError("prediction contains non-finite values")
        prediction_tensor[:, step, :] = next_step
        if history_steps > 1:
            history[:, :-1, :] = history[:, 1:, :].clone()
        history[:, -1, :] = next_step
    return prediction_tensor.detach().cpu().numpy().astype(np.float32)


def _with_torch_threads(
    *,
    torch: Any,
    torch_num_threads: int | None,
    operation: Any,
) -> Any:
    if torch_num_threads is None:
        return operation()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    try:
        return operation()
    finally:
        torch.set_num_threads(previous_threads)


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
        raise UNetBackendError(
            "UNet backend requires the optional torch extra; run with --extra torch"
        ) from exc
    return torch


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise UNetBackendError(f"missing {label} path: {path}")
    if not path.is_file():
        raise UNetBackendError(f"{label} path is not a file: {path}")
    return path
