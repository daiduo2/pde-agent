"""Execute a minimal trainable 1D UNet candidate on local window data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


class UNetCandidateExecutionError(RuntimeError):
    pass


def execute_unet_candidate(
    *,
    windows_path: str | Path,
    output_dir: str | Path,
    start_index: int,
    train_steps: int,
    training_strategy: str,
    base_channels: int = 16,
    depth: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    strategy_label: str | None = None,
    nu: float = 0.001,
    dt: float = 1.0,
    dx: float = 1.0,
    spectral_weight: float = 0.1,
    physics_weight: float = 0.01,
    force: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    windows_file = _require_file(Path(windows_path), "windows")
    context, target = _load_windows(windows_file)
    start = _non_negative_int(start_index, "start_index")
    steps = _positive_int(train_steps, "train_steps")
    strategy = _training_strategy(training_strategy)
    channels = _positive_int(base_channels, "base_channels")
    model_depth = _positive_int(depth, "depth")
    lr = _positive_float(learning_rate, "learning_rate")
    label = _strategy_label(strategy_label)
    pde_nu = _positive_float(nu, "nu")
    pde_dt = _positive_float(dt, "dt")
    pde_dx = _positive_float(dx, "dx")
    spectral_w = _positive_float(spectral_weight, "spectral_weight")
    physics_w = _positive_float(physics_weight, "physics_weight")

    output_root = Path(output_dir)
    paths = {
        "candidate_prediction_path": output_root / "candidate_prediction.npz",
        "candidate_checkpoint_path": output_root / "candidate_unet_checkpoint.pt",
        "candidate_execution_report_path": output_root
        / "candidate_execution_report.json",
    }
    _prepare_output_paths(paths=paths, force=force)

    torch.manual_seed(int(seed))
    model = _MinimalUNet1D(
        torch=torch,
        context_steps=context.shape[1],
        forecast_steps=target.shape[1],
        base_channels=channels,
        depth=model_depth,
        output_space_points=target.shape[2],
    )
    loss_trace = _train_model(
        torch=torch,
        model=model,
        context=context,
        target=target,
        train_steps=steps,
        learning_rate=lr,
        training_strategy=strategy,
        nu=pde_nu,
        dt=pde_dt,
        dx=pde_dx,
        spectral_weight=spectral_w,
        physics_weight=physics_w,
    )
    prediction = _predict(torch=torch, model=model, context=context)

    label_part = f"_{label}" if label is not None else ""
    experiment_id = f"unet_{strategy}{label_part}_s{start:04d}_steps{steps:04d}"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "model_type": "unet",
                "adapter_type": "torch_unet1d_minimal",
                "experiment_id": experiment_id,
                "training_strategy": strategy,
                "strategy_label": label,
                "source_windows_path": str(windows_file),
                "context_shape": list(context.shape),
                "target_shape": list(target.shape),
                "base_channels": channels,
                "depth": model_depth,
                "learning_rate": lr,
                "seed": int(seed),
                "nu": pde_nu,
                "dt": pde_dt,
                "dx": pde_dx,
                "spectral_weight": spectral_w,
                "physics_weight": physics_w,
            },
        },
        paths["candidate_checkpoint_path"],
    )
    with paths["candidate_prediction_path"].open("wb") as handle:
        np.savez(
            handle,
            prediction=prediction.astype(np.float32),
            metadata_json=json.dumps(
                {
                    "model_type": "unet",
                    "adapter_type": "torch_unet1d_minimal",
                    "experiment_id": experiment_id,
                    "training_strategy": strategy,
                    "strategy_label": label,
                    "source_windows_path": str(windows_file),
                    "checkpoint_path": str(paths["candidate_checkpoint_path"]),
                },
                allow_nan=False,
                sort_keys=True,
            ),
        )

    report = {
        "schema_version": 1,
        "state_hint": "UNET_CANDIDATE_EXECUTED",
        "windows_path": str(windows_file),
        "inputs": {
            "context_steps": int(context.shape[1]),
            "forecast_steps": int(target.shape[1]),
            "space_points": int(target.shape[2]),
        },
        "candidate": {
            "experiment_id": experiment_id,
            "start_index": start,
            "train_steps": steps,
            "training_strategy": strategy,
            "strategy_label": label,
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "train": {
            "mode": "torch_unet1d",
            "loss_trace": loss_trace,
            "initial_loss": loss_trace[0],
            "final_loss": loss_trace[-1],
            "loss_terms": _loss_terms(strategy),
            "base_channels": channels,
            "depth": model_depth,
            "learning_rate": lr,
            "pde": {"nu": pde_nu, "dt": pde_dt, "dx": pde_dx},
            "loss_weights": {
                "spectral": spectral_w,
                "burgers_residual": physics_w,
            },
        },
        "next_command": "evaluate-unet-candidate",
    }
    paths["candidate_execution_report_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


class _MinimalUNet1D:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        base_channels: int,
        depth: int,
        output_space_points: int,
    ) -> None:
        self._torch = torch
        self._output_space_points = output_space_points
        self.input_block = _conv_block(
            torch=torch,
            in_channels=context_steps,
            out_channels=base_channels,
        )
        self.encoders = torch.nn.ModuleList()
        skip_channels: list[int] = []
        current_channels = base_channels
        for level in range(depth):
            out_channels = base_channels * (2**level)
            self.encoders.append(
                _conv_block(
                    torch=torch,
                    in_channels=current_channels,
                    out_channels=out_channels,
                )
            )
            skip_channels.append(out_channels)
            current_channels = out_channels
        self.bottleneck = _conv_block(
            torch=torch,
            in_channels=current_channels,
            out_channels=current_channels * 2,
        )
        current_channels *= 2
        self.decoders = torch.nn.ModuleList()
        for skip_channels_value in reversed(skip_channels):
            self.decoders.append(
                _conv_block(
                    torch=torch,
                    in_channels=current_channels + skip_channels_value,
                    out_channels=skip_channels_value,
                )
            )
            current_channels = skip_channels_value
        self.output_projection = torch.nn.Conv1d(
            current_channels,
            forecast_steps,
            kernel_size=1,
        )

    def parameters(self) -> Any:
        modules = [
            self.input_block,
            self.encoders,
            self.bottleneck,
            self.decoders,
            self.output_projection,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_block": self.input_block.state_dict(),
            "encoders": self.encoders.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "decoders": self.decoders.state_dict(),
            "output_projection": self.output_projection.state_dict(),
        }

    def train(self) -> None:
        for module in (
            self.input_block,
            self.encoders,
            self.bottleneck,
            self.decoders,
            self.output_projection,
        ):
            module.train()

    def eval(self) -> None:
        for module in (
            self.input_block,
            self.encoders,
            self.bottleneck,
            self.decoders,
            self.output_projection,
        ):
            module.eval()

    def __call__(self, context_tensor: Any) -> Any:
        torch = self._torch
        x = self.input_block(context_tensor)
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            if x.shape[-1] > 1:
                x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[-1],
                mode="linear",
                align_corners=False,
            )
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        x = self.output_projection(x)
        if x.shape[-1] != self._output_space_points:
            x = torch.nn.functional.interpolate(
                x,
                size=self._output_space_points,
                mode="linear",
                align_corners=False,
            )
        return x


def _conv_block(*, torch: Any, in_channels: int, out_channels: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    )


def _train_model(
    *,
    torch: Any,
    model: _MinimalUNet1D,
    context: np.ndarray,
    target: np.ndarray,
    train_steps: int,
    learning_rate: float,
    training_strategy: str,
    nu: float,
    dt: float,
    dx: float,
    spectral_weight: float,
    physics_weight: float,
) -> list[float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    context_tensor = torch.as_tensor(context, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(train_steps):
        optimizer.zero_grad()
        prediction = model(context_tensor)
        loss = _training_loss(
            torch=torch,
            prediction=prediction,
            target=target_tensor,
            training_strategy=training_strategy,
            nu=nu,
            dt=dt,
            dx=dx,
            spectral_weight=spectral_weight,
            physics_weight=physics_weight,
        )
        if not torch.isfinite(loss):
            raise UNetCandidateExecutionError("training loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def _training_loss(
    *,
    torch: Any,
    prediction: Any,
    target: Any,
    training_strategy: str,
    nu: float,
    dt: float,
    dx: float,
    spectral_weight: float,
    physics_weight: float,
) -> Any:
    mse = torch.nn.functional.mse_loss(prediction, target)
    if training_strategy == "unet_supervised":
        return mse
    if training_strategy == "unet_spectral":
        pred_spec = torch.abs(torch.fft.rfft(prediction, dim=-1))
        target_spec = torch.abs(torch.fft.rfft(target, dim=-1))
        spectral = torch.nn.functional.mse_loss(pred_spec, target_spec)
        return mse + spectral_weight * spectral
    if training_strategy == "unet_physics_residual":
        residual = _burgers_residual(
            torch=torch,
            field=prediction,
            nu=nu,
            dt=dt,
            dx=dx,
        )
        return mse + physics_weight * torch.mean(residual * residual)
    raise UNetCandidateExecutionError(f"unsupported training strategy: {training_strategy}")


def _burgers_residual(
    *,
    torch: Any,
    field: Any,
    nu: float,
    dt: float,
    dx: float,
) -> Any:
    if field.shape[1] < 3 or field.shape[2] < 3:
        return torch.zeros_like(field[:, :1, :1])
    u_mid = field[:, 1:-1, :]
    u_t = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dt)
    u_x = (
        torch.roll(u_mid, shifts=-1, dims=-1)
        - torch.roll(u_mid, shifts=1, dims=-1)
    ) / (2.0 * dx)
    u_xx = (
        torch.roll(u_mid, shifts=-1, dims=-1)
        - 2.0 * u_mid
        + torch.roll(u_mid, shifts=1, dims=-1)
    ) / (dx * dx)
    return u_t + u_mid * u_x - nu * u_xx


def _predict(*, torch: Any, model: _MinimalUNet1D, context: np.ndarray) -> np.ndarray:
    model.eval()
    context_tensor = torch.as_tensor(context, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(context_tensor)
    if not torch.isfinite(prediction).all():
        raise UNetCandidateExecutionError("prediction contains non-finite values")
    return prediction.detach().cpu().numpy()


def _load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.load(path)
        context = data["context"].astype(np.float32)
        target = data["target"].astype(np.float32)
    except KeyError as exc:
        raise UNetCandidateExecutionError(
            "windows npz must include context and target arrays"
        ) from exc
    if context.ndim != 3 or target.ndim != 3:
        raise UNetCandidateExecutionError(
            "context and target must have shape (sample, time, space)"
        )
    if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
        raise UNetCandidateExecutionError(
            "context and target must share sample and space dimensions"
        )
    if context.shape[1] < 1 or target.shape[1] < 1:
        raise UNetCandidateExecutionError("context and target must have time steps")
    return context, target


def _training_strategy(value: str) -> str:
    allowed = {"unet_supervised", "unet_spectral", "unet_physics_residual"}
    if value not in allowed:
        raise UNetCandidateExecutionError(
            "training_strategy must be one of: " + ", ".join(sorted(allowed))
        )
    return value


def _loss_terms(training_strategy: str) -> list[str]:
    if training_strategy == "unet_supervised":
        return ["mse"]
    if training_strategy == "unet_spectral":
        return ["mse", "spectral"]
    if training_strategy == "unet_physics_residual":
        return ["mse", "burgers_residual"]
    raise UNetCandidateExecutionError(f"unsupported training strategy: {training_strategy}")


def _strategy_label(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise UNetCandidateExecutionError(
            "strategy_label must be a non-empty string when provided"
        )
    if re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
        raise UNetCandidateExecutionError(
            "strategy_label may contain only letters, digits, and underscores"
        )
    return value


def _prepare_output_paths(*, paths: dict[str, Path], force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise UNetCandidateExecutionError(f"UNet candidate artifact already exists: {names}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _non_negative_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UNetCandidateExecutionError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UNetCandidateExecutionError(f"{label} must be a positive integer")
    return value


def _positive_float(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        raise UNetCandidateExecutionError(f"{label} must be a positive number")
    return float(value)


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise UNetCandidateExecutionError(f"missing {label} path: {path}")
    if not path.is_file():
        raise UNetCandidateExecutionError(f"{label} path is not a file: {path}")
    return path


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise UNetCandidateExecutionError(
            "torch is required for execute-unet-candidate"
        ) from exc
    return torch
