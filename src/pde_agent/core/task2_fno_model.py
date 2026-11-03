"""Task 2 FNO model definitions.

Lightweight spectral-convolution operator for multi-Nu Burgers equation.
"""
from __future__ import annotations

from typing import Any


class Task2FnoModelError(RuntimeError):
    pass


class SpectralConv1D:
    """1D spectral convolution using torch.fft."""

    def __init__(
        self,
        *,
        torch: Any,
        channels: int,
        modes: int,
        device: Any = None,
    ) -> None:
        self._torch = torch
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, channels * channels)
        self.weight = torch.nn.Parameter(
            scale
            * torch.randn(channels, channels, self.modes, dtype=torch.cfloat, device=device)
        )

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


class FNOLiteRollout:
    """Direct context-to-forecast FNO for Task 2.

    Maps context [B, T_context, X] -> forecast [B, T_forecast, X].
    """

    def __init__(
        self,
        *,
        torch: Any,
        context_steps: int,
        forecast_steps: int,
        space_points: int,
        hidden_channels: int,
        modes: int = 16,
        depth: int = 4,
        device: Any = None,
    ) -> None:
        self._torch = torch
        self.forecast_steps = forecast_steps
        mode_count = max(1, min(int(modes), (int(space_points) // 2) + 1))

        self.lift = torch.nn.Conv1d(
            context_steps, hidden_channels, kernel_size=1, device=device
        )
        self.spectral_layers = [
            SpectralConv1D(
                torch=torch, channels=hidden_channels, modes=mode_count, device=device
            )
            for _ in range(depth)
        ]
        self.local_layers = [
            torch.nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size=1, device=device
            )
            for _ in range(depth)
        ]
        self.project = torch.nn.Conv1d(
            hidden_channels, forecast_steps, kernel_size=1, device=device
        )
        self.activation = torch.nn.GELU()

        for m in [self.lift, self.project, *self.local_layers]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def parameters(self):
        return (
            list(self.lift.parameters())
            + list(self.project.parameters())
            + [p for layer in self.local_layers for p in layer.parameters()]
            + [layer.weight for layer in self.spectral_layers]
        )

    def state_dict(self):
        return {
            "lift": self.lift.state_dict(),
            "project": self.project.state_dict(),
            "local_layers": [layer.state_dict() for layer in self.local_layers],
            "spectral_layers": [{"weight": layer.weight} for layer in self.spectral_layers],
        }

    def load_state_dict(self, state):
        self.lift.load_state_dict(state["lift"])
        self.project.load_state_dict(state["project"])
        for layer, s in zip(self.local_layers, state["local_layers"]):
            layer.load_state_dict(s)
        for layer, s in zip(self.spectral_layers, state["spectral_layers"]):
            layer.weight.data.copy_(s["weight"].data)

    def __call__(self, context: Any) -> Any:
        hidden = self.activation(self.lift(context))
        for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
            hidden = self.activation(spectral_layer(hidden) + local_layer(hidden))
        return self.project(hidden)

    def train_mode(self):
        self.lift.train()
        for layer in self.local_layers:
            layer.train()
        self.project.train()

    def eval_mode(self):
        self.lift.eval()
        for layer in self.local_layers:
            layer.eval()
        self.project.eval()
