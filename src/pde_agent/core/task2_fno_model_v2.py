"""Enhanced Task 2 FNO model with FiLM, nu-estimator, coordinate channels, and residual output.

Ported from mlw67/pdeagent insights:
- Coordinate channel injection (spatial awareness)
- FiLM + nu_estimator (handle hidden nu at test time)
- Residual output (learn increment, not absolute)
- GroupNorm (better for small batch training)
"""
from __future__ import annotations

from typing import Any


class Task2FnoModelV2Error(RuntimeError):
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


class FiLM:
    """Feature-wise linear modulation for optional conditioning."""

    def __init__(self, torch: Any, cond_dim: int, width: int, device: Any = None) -> None:
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, 2 * width, device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * width, 2 * width, device=device),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def __call__(self, x: Any, cond: Any) -> Any:
        if cond.ndim == 1:
            cond = cond[:, None]
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FNOLiteRolloutV2:
    """Enhanced FNO for Task 2 with optional FiLM, coord channels, and residual output.

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
        use_film: bool = False,
        use_coord: bool = False,
        use_residual: bool = False,
        use_groupnorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        self._torch = torch
        self.forecast_steps = forecast_steps
        self.use_film = use_film
        self.use_coord = use_coord
        self.use_residual = use_residual
        mode_count = max(1, min(int(modes), (int(space_points) // 2) + 1))

        # Coordinate channel: concat spatial linspace(0,1) to input
        lift_in = context_steps + (1 if use_coord else 0)
        self.lift = torch.nn.Conv1d(
            lift_in, hidden_channels, kernel_size=1, device=device
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

        # Optional GroupNorm per block (num_groups=1 for small batch stability)
        self.norms = None
        if use_groupnorm:
            self.norms = [
                torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, device=device)
                for _ in range(depth)
            ]

        # Optional FiLM conditioning layers
        self.films = None
        if use_film:
            self.films = [FiLM(torch, cond_dim=1, width=hidden_channels, device=device) for _ in range(depth)]
            # nu_estimator: Conv1d -> GELU -> Pool -> Linear
            self.nu_estimator = torch.nn.Sequential(
                torch.nn.Conv1d(context_steps, 32, kernel_size=1, device=device),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 1, device=device),
            )

        self.project = torch.nn.Conv1d(
            hidden_channels, forecast_steps, kernel_size=1, device=device
        )
        self.activation = torch.nn.GELU()

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        for m in [self.lift, self.project, *self.local_layers]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _coord(self, x: Any) -> Any:
        """Return spatial coordinate channel [B, 1, Nx] linspace(0,1)."""
        torch = self._torch
        b, _, n = x.shape
        return torch.linspace(0.0, 1.0, n, device=x.device, dtype=x.dtype).view(1, 1, n).expand(b, -1, -1)

    def parameters(self):
        params = (
            list(self.lift.parameters())
            + list(self.project.parameters())
            + [p for layer in self.local_layers for p in layer.parameters()]
            + [layer.weight for layer in self.spectral_layers]
        )
        if self.norms is not None:
            params += [p for layer in self.norms for p in layer.parameters()]
        if self.films is not None:
            params += [p for film in self.films for p in film.net.parameters()]
            params += list(self.nu_estimator.parameters())
        return params

    def state_dict(self):
        state = {
            "lift": self.lift.state_dict(),
            "project": self.project.state_dict(),
            "local_layers": [layer.state_dict() for layer in self.local_layers],
            "spectral_layers": [{"weight": layer.weight} for layer in self.spectral_layers],
        }
        if self.norms is not None:
            state["norms"] = [layer.state_dict() for layer in self.norms]
        if self.films is not None:
            state["films"] = [{"net": film.net.state_dict()} for film in self.films]
            state["nu_estimator"] = self.nu_estimator.state_dict()
        return state

    def load_state_dict(self, state):
        self.lift.load_state_dict(state["lift"])
        self.project.load_state_dict(state["project"])
        for layer, s in zip(self.local_layers, state["local_layers"]):
            layer.load_state_dict(s)
        for layer, s in zip(self.spectral_layers, state["spectral_layers"]):
            layer.weight.data.copy_(s["weight"].data)
        if self.norms is not None and "norms" in state:
            for layer, s in zip(self.norms, state["norms"]):
                layer.load_state_dict(s)
        if self.films is not None and "films" in state:
            for film, s in zip(self.films, state["films"]):
                film.net.load_state_dict(s["net"])
        if "nu_estimator" in state:
            self.nu_estimator.load_state_dict(state["nu_estimator"])

    def __call__(self, context: Any, cond: Any = None) -> Any:
        """Predict forecast from context.

        Args:
            context: [B, T_context, X] input trajectory.
            cond: Optional [B] or [B, 1] nu values. If None and use_film=True,
                  nu is estimated from context automatically.

        Returns:
            forecast: [B, T_forecast, X] predicted future.
        """
        # Coordinate channel injection
        h = context
        if self.use_coord:
            h = self._torch.cat([h, self._coord(h)], dim=1)

        h = self.activation(self.lift(h))

        # FiLM: estimate nu if not provided
        if self.use_film and self.films is not None:
            if cond is None:
                cond = self.nu_estimator(context).squeeze(-1)
            for spectral_layer, local_layer, film in zip(
                self.spectral_layers, self.local_layers, self.films
            ):
                h = spectral_layer(h) + local_layer(h)
                h = film(h, cond)
                h = self.activation(h)
        else:
            for spectral_layer, local_layer in zip(self.spectral_layers, self.local_layers):
                h = self.activation(spectral_layer(h) + local_layer(h))

        if self.use_residual:
            # Residual output: last_frame.expand + project(features)
            last_frame = context[:, -1:, :]
            return last_frame.expand(-1, self.forecast_steps, -1) + self.project(h)
        return self.project(h)

    def train_mode(self):
        self.lift.train()
        for layer in self.local_layers:
            layer.train()
        self.project.train()
        if self.norms is not None:
            for layer in self.norms:
                layer.train()
        if self.films is not None:
            self.nu_estimator.train()

    def eval_mode(self):
        self.lift.eval()
        for layer in self.local_layers:
            layer.eval()
        self.project.eval()
        if self.norms is not None:
            for layer in self.norms:
                layer.eval()
        if self.films is not None:
            self.nu_estimator.eval()
