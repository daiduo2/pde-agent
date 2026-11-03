from __future__ import annotations

from typing import Any

import numpy as np

from pde_agent.core.task1_short_window_common import (
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)


class Conv1DResidualRollout:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        hidden_channels: int,
    ) -> None:
        self._torch = torch
        self.operator_type = "conv1d_residual"
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(context_steps, hidden_channels, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_channels, forecast_steps, kernel_size=1),
        )

    def parameters(self) -> Any:
        return self.network.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict)

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def __call__(self, context: Any) -> Any:
        raw = self.network(context)
        anchor = context[:, -1:, :]
        return anchor + raw


class FNOLiteRollout:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        space_points: int,
        hidden_channels: int,
        modes: int = 8,
        depth: int = 1,
    ) -> None:
        self._torch = torch
        self.operator_type = "spectral_conv1d"
        mode_count = max(1, min(int(modes), (int(space_points) // 2) + 1))
        self.lift = torch.nn.Conv1d(context_steps, hidden_channels, kernel_size=1)
        self.spectral_layers = [
            SpectralConv1D(torch=torch, channels=hidden_channels, modes=mode_count)
            for _ in range(depth)
        ]
        self.local_layers = [
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(depth)
        ]
        self.project = torch.nn.Conv1d(hidden_channels, forecast_steps, kernel_size=1)
        self.activation = torch.nn.GELU()

    def parameters(self) -> Any:
        return (
            list(self.lift.parameters())
            + list(self.project.parameters())
            + [
                parameter
                for layer in self.local_layers
                for parameter in layer.parameters()
            ]
            + [layer.weight for layer in self.spectral_layers]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "lift": self.lift.state_dict(),
            "spectral_layers": [
                layer.state_dict() for layer in self.spectral_layers
            ],
            "local_layers": [layer.state_dict() for layer in self.local_layers],
            "project": self.project.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lift.load_state_dict(state_dict["lift"])
        for layer, layer_state in zip(self.spectral_layers, state_dict["spectral_layers"]):
            layer.load_state_dict(layer_state)
        for layer, layer_state in zip(self.local_layers, state_dict["local_layers"]):
            layer.load_state_dict(layer_state)
        self.project.load_state_dict(state_dict["project"])

    def train(self) -> None:
        self.lift.train()
        for layer in self.local_layers:
            layer.train()
        self.project.train()

    def eval(self) -> None:
        self.lift.eval()
        for layer in self.local_layers:
            layer.eval()
        self.project.eval()

    def __call__(self, context: Any) -> Any:
        hidden = self.activation(self.lift(context))
        for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
            hidden = self.activation(spectral_layer(hidden) + local_layer(hidden))
        raw = self.project(hidden)
        anchor = context[:, -1:, :]
        return anchor + raw


class FNOAutoregressiveOneStep:
    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        space_points: int,
        hidden_channels: int,
        modes: int = 8,
        depth: int = 1,
    ) -> None:
        self._torch = torch
        self.operator_type = "spectral_conv1d_autoregressive"
        self.context_steps = int(context_steps)
        self.forecast_steps = int(forecast_steps)
        mode_count = max(1, min(int(modes), (int(space_points) // 2) + 1))
        self.lift = torch.nn.Conv1d(context_steps, hidden_channels, kernel_size=1)
        self.spectral_layers = [
            SpectralConv1D(torch=torch, channels=hidden_channels, modes=mode_count)
            for _ in range(depth)
        ]
        self.local_layers = [
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(depth)
        ]
        self.project = torch.nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.activation = torch.nn.GELU()

    def parameters(self) -> Any:
        return (
            list(self.lift.parameters())
            + list(self.project.parameters())
            + [
                parameter
                for layer in self.local_layers
                for parameter in layer.parameters()
            ]
            + [layer.weight for layer in self.spectral_layers]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "lift": self.lift.state_dict(),
            "spectral_layers": [
                layer.state_dict() for layer in self.spectral_layers
            ],
            "local_layers": [layer.state_dict() for layer in self.local_layers],
            "project": self.project.state_dict(),
        }

    def train(self) -> None:
        self.lift.train()
        for layer in self.local_layers:
            layer.train()
        self.project.train()

    def eval(self) -> None:
        self.lift.eval()
        for layer in self.local_layers:
            layer.eval()
        self.project.eval()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lift.load_state_dict(state_dict["lift"])
        for layer, sd in zip(self.spectral_layers, state_dict["spectral_layers"]):
            layer.load_state_dict(sd)
        for layer, sd in zip(self.local_layers, state_dict["local_layers"]):
            layer.load_state_dict(sd)
        self.project.load_state_dict(state_dict["project"])

    def __call__(self, context: Any) -> Any:
        window = context
        predictions = []
        for _ in range(self.forecast_steps):
            hidden = self.activation(self.lift(window))
            for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
                hidden = self.activation(spectral_layer(hidden) + local_layer(hidden))
            residual = self.project(hidden)
            next_step = window[:, -1:, :] + residual
            predictions.append(next_step)
            window = self._torch.cat([window[:, 1:, :], next_step], dim=1)
        return self._torch.cat(predictions, dim=1)


class SpectralConv1D:
    def __init__(self, *, torch: Any, channels: int, modes: int) -> None:
        self._torch = torch
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, channels * channels)
        self.weight = torch.nn.Parameter(
            scale
            * torch.randn(
                channels,
                channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def parameters(self) -> Any:
        return [self.weight]

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.weight.data.copy_(state_dict["weight"].data)

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def __call__(self, value: Any) -> Any:
        torch = self._torch
        spectrum = torch.fft.rfft(value, dim=-1)
        output_spectrum = torch.zeros_like(spectrum)
        modes = min(self.modes, spectrum.shape[-1])
        output_spectrum[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            spectrum[:, :, :modes],
            self.weight[:, :, :modes],
        )
        return torch.fft.irfft(output_spectrum, n=value.shape[-1], dim=-1)


def build_model(
    *,
    torch: Any,
    model_family: str,
    context_steps: int,
    forecast_steps: int,
    space_points: int,
    hidden_channels: int,
    modes: int,
    depth: int,
) -> Any:
    if model_family == "conv1d_residual":
        return Conv1DResidualRollout(
            torch=torch,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            hidden_channels=hidden_channels,
        )
    if model_family == "fno_lite":
        return FNOLiteRollout(
            torch=torch,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            space_points=space_points,
            hidden_channels=hidden_channels,
            modes=modes,
            depth=depth,
        )
    if model_family == "fno_ar1":
        return FNOAutoregressiveOneStep(
            torch=torch,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            space_points=space_points,
            hidden_channels=hidden_channels,
            modes=modes,
            depth=depth,
        )
    raise Task1ShortWindowTrainingError(f"unsupported model_family: {model_family}")


def training_loss(*, torch: Any, prediction: Any, target: Any, loss_type: str) -> Any:
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    if loss_type == "early_weighted_mse":
        weights = torch.linspace(
            1.5,
            0.75,
            steps=prediction.shape[1],
            dtype=prediction.dtype,
            device=prediction.device,
        ).view(1, -1, 1)
        return torch.mean(weights * (prediction - target) ** 2)
    if loss_type == "horizon_weighted_mse":
        first = max(1, prediction.shape[1] // 4)
        second = max(first + 1, prediction.shape[1] // 2)
        weights = torch.full(
            (prediction.shape[1],),
            0.75,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        weights[:first] = 1.5
        weights[first:second] = 1.0
        return torch.mean(weights.view(1, -1, 1) * (prediction - target) ** 2)
    if loss_type == "spectral_mse":
        base = torch.nn.functional.mse_loss(prediction, target)
        pred_spec = torch.fft.rfft(prediction, dim=-1)
        target_spec = torch.fft.rfft(target, dim=-1)
        spectral = torch.mean(torch.abs(pred_spec - target_spec) ** 2)
        return base + 0.01 * spectral
    raise Task1ShortWindowTrainingError(f"unsupported loss_type: {loss_type}")


def normalize_windows(
    *,
    context: np.ndarray,
    target: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if normalization == "none":
        ones = np.ones((context.shape[0], 1, 1), dtype=np.float32)
        zeros = np.zeros((context.shape[0], 1, 1), dtype=np.float32)
        return context, target, {"mean": zeros, "std": ones}
    if normalization == "sample_standard":
        mean = np.mean(context, axis=(1, 2), keepdims=True).astype(np.float32)
        std = np.std(context, axis=(1, 2), keepdims=True).astype(np.float32)
        std = np.maximum(std, np.float32(1e-6))
        return (
            ((context - mean) / std).astype(np.float32),
            ((target - mean) / std).astype(np.float32),
            {"mean": mean, "std": std},
        )
    raise Task1ShortWindowTrainingError(
        "normalization must be one of: " + ", ".join(sorted(SUPPORTED_NORMALIZATION_TYPES))
    )


def denormalize_prediction(
    *,
    prediction: np.ndarray,
    stats: dict[str, np.ndarray],
    normalization: str,
) -> np.ndarray:
    if normalization == "none":
        return prediction.astype(np.float32)
    if normalization == "sample_standard":
        return (prediction * stats["std"] + stats["mean"]).astype(np.float32)
    raise Task1ShortWindowTrainingError(
        "normalization must be one of: " + ", ".join(sorted(SUPPORTED_NORMALIZATION_TYPES))
    )


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


# ---------------------------------------------------------------------------
# Registry registration — enables agent discovery and dynamic extension
# ---------------------------------------------------------------------------


def _register_builtin_models() -> None:
    """Register existing models to the neural_operator component registry."""
    try:
        from pde_agent.core.neural_operator.registry import register_model
        from pde_agent.core.neural_operator.config import ModelConfig
    except Exception:
        return  # registry not available, skip

    @register_model("conv1d_residual")
    def _build_conv1d_residual(cfg: ModelConfig) -> Any:
        import torch
        return Conv1DResidualRollout(
            torch=torch,
            context_steps=cfg.t_in,
            forecast_steps=cfg.t_out,
            hidden_channels=cfg.width,
        )

    @register_model("fno_lite")
    def _build_fno_lite(cfg: ModelConfig) -> Any:
        import torch
        return FNOLiteRollout(
            torch=torch,
            context_steps=cfg.t_in,
            forecast_steps=cfg.t_out,
            space_points=256,
            hidden_channels=cfg.width,
            modes=cfg.modes,
            depth=cfg.depth,
        )

    @register_model("fno_ar1")
    def _build_fno_ar1(cfg: ModelConfig) -> Any:
        import torch
        return FNOAutoregressiveOneStep(
            torch=torch,
            context_steps=cfg.t_in,
            forecast_steps=cfg.t_out,
            space_points=256,
            hidden_channels=cfg.width,
            modes=cfg.modes,
            depth=cfg.depth,
        )


def _register_builtin_losses() -> None:
    """Register existing losses to the neural_operator component registry."""
    try:
        from pde_agent.core.neural_operator.registry import register_loss
        from pde_agent.core.neural_operator.config import LossConfig
    except Exception:
        return  # registry not available, skip

    @register_loss("mse")
    def _build_mse_loss(_cfg: LossConfig) -> Any:
        import torch
        return lambda pred, target: torch.nn.functional.mse_loss(pred, target)

    @register_loss("early_weighted_mse")
    def _build_early_weighted_mse(_cfg: LossConfig) -> Any:
        import torch
        def loss_fn(pred: Any, target: Any) -> Any:
            weights = torch.linspace(
                1.5, 0.75, steps=pred.shape[1],
                dtype=pred.dtype, device=pred.device,
            ).view(1, -1, 1)
            return torch.mean(weights * (pred - target) ** 2)
        return loss_fn

    @register_loss("horizon_weighted_mse")
    def _build_horizon_weighted_mse(_cfg: LossConfig) -> Any:
        import torch
        def loss_fn(pred: Any, target: Any) -> Any:
            first = max(1, pred.shape[1] // 4)
            second = max(first + 1, pred.shape[1] // 2)
            weights = torch.full((pred.shape[1],), 0.75, dtype=pred.dtype, device=pred.device)
            weights[:first] = 1.5
            weights[first:second] = 1.0
            return torch.mean(weights.view(1, -1, 1) * (pred - target) ** 2)
        return loss_fn

    @register_loss("spectral_mse")
    def _build_spectral_mse(_cfg: LossConfig) -> Any:
        import torch
        def loss_fn(pred: Any, target: Any) -> Any:
            base = torch.nn.functional.mse_loss(pred, target)
            pred_spec = torch.fft.rfft(pred, dim=-1)
            target_spec = torch.fft.rfft(target, dim=-1)
            spectral = torch.mean(torch.abs(pred_spec - target_spec) ** 2)
            return base + 0.01 * spectral
        return loss_fn


# Trigger registrations on module import
_register_builtin_models()
_register_builtin_losses()
