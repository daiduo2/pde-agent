"""Configuration dataclass for neural operator experiments.

Experiments are fully defined by a JSON-serializable config that specifies
model, dataset, loss, optimizer, training loop, and scoring parameters.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    name: str = "fno_forecast"
    modes: int = 24
    width: int = 64
    depth: int = 4
    t_in: int = 10
    t_out: int = 190
    chunk_size: int = 10
    use_film: bool = False
    dropout: float = 0.0


@dataclass
class DatasetConfig:
    """Dataset and preprocessing parameters."""

    name: str = "burgers"
    data_dir: str = ""
    hdf5_path: str = ""
    task: str = "task1"
    t_in: int = 10
    t_out: int = 190
    val_fraction: float = 0.2
    augment_shift: bool = False
    window_stride: int = 1
    chunk_size: int = 10
    train_target_horizon: int | None = None
    model_type: str = "direct"
    normalizer_mean: float | None = None
    normalizer_std: float | None = None


@dataclass
class LossConfig:
    """Loss function parameters."""

    name: str = "weighted_mse"
    multi_step_weight: float = 0.0
    grad_weight: float = 0.05
    time_diff_weight: float = 0.02
    physics_weight: float = 1e-5
    use_physics_loss: bool = False
    time_weight: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler parameters."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 220
    batch_size: int = 16
    patience: int = 35
    seed: int = 42
    device: str = "auto"
    amp: bool = False
    compile: bool = False
    num_workers: int = 0
    unroll_chunks: int = 1
    ss_start_epoch: int = 30
    ss_ramp_epochs: int = 80
    ss_max_prob: float = 0.3
    resume: str = ""


@dataclass
class ScoringConfig:
    """Scoring and evaluation parameters."""

    segment_lengths: list[int] = field(default_factory=lambda: [48, 48, 94])
    spectrum_modes: int = 32


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Fully deterministic experiments are defined by this object plus a seed.
    """

    run_id: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output_dir: str = "./runs"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Construct from a nested dictionary (e.g. loaded JSON)."""
        model = ModelConfig(**raw.get("model", {}))
        dataset = DatasetConfig(**raw.get("dataset", {}))
        loss = LossConfig(**raw.get("loss", {}))
        optimizer = OptimizerConfig(**raw.get("optimizer", {}))
        training = TrainingConfig(**raw.get("training", {}))
        scoring = ScoringConfig(**raw.get("scoring", {}))
        return cls(
            run_id=raw.get("run_id", ""),
            model=model,
            dataset=dataset,
            loss=loss,
            optimizer=optimizer,
            training=training,
            scoring=scoring,
            output_dir=raw.get("output_dir", "./runs"),
            metadata=raw.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain nested dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save as a pretty-printed JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True, default=str)
            f.write("\n")

    def validate(self) -> None:
        """Validate config values are consistent and positive where required."""
        cfg = self
        _require_positive("model.modes", cfg.model.modes)
        _require_positive("model.width", cfg.model.width)
        _require_positive("model.depth", cfg.model.depth)
        _require_positive("model.t_in", cfg.model.t_in)
        _require_positive("model.t_out", cfg.model.t_out)
        _require_positive("model.chunk_size", cfg.model.chunk_size)
        _require_non_negative("model.dropout", cfg.model.dropout)

        _require_positive("dataset.t_in", cfg.dataset.t_in)
        _require_positive("dataset.t_out", cfg.dataset.t_out)
        _require_positive("dataset.val_fraction", cfg.dataset.val_fraction)
        _require_positive("dataset.window_stride", cfg.dataset.window_stride)
        _require_positive("dataset.chunk_size", cfg.dataset.chunk_size)
        if cfg.dataset.val_fraction >= 1.0:
            raise ConfigError("dataset.val_fraction must be < 1.0")

        _require_non_negative("loss.multi_step_weight", cfg.loss.multi_step_weight)
        _require_non_negative("loss.grad_weight", cfg.loss.grad_weight)
        _require_non_negative("loss.time_diff_weight", cfg.loss.time_diff_weight)
        _require_non_negative("loss.physics_weight", cfg.loss.physics_weight)
        _require_non_negative("loss.time_weight", cfg.loss.time_weight)

        _require_positive("optimizer.lr", cfg.optimizer.lr)
        _require_non_negative("optimizer.weight_decay", cfg.optimizer.weight_decay)
        _require_non_negative("optimizer.grad_clip", cfg.optimizer.grad_clip)

        _require_positive("training.epochs", cfg.training.epochs)
        _require_positive("training.batch_size", cfg.training.batch_size)
        _require_positive("training.patience", cfg.training.patience)
        _require_non_negative("training.unroll_chunks", cfg.training.unroll_chunks)
        _require_non_negative("training.ss_start_epoch", cfg.training.ss_start_epoch)
        _require_non_negative("training.ss_ramp_epochs", cfg.training.ss_ramp_epochs)
        _require_non_negative("training.ss_max_prob", cfg.training.ss_max_prob)

        if cfg.training.device == "auto":
            try:
                import torch
                cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                cfg.training.device = "cpu"


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ConfigError(f"{name} must be non-negative, got {value}")
