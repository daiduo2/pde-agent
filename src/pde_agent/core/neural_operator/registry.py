"""Decorator-based component registry for neural operator experiments.

Supports registration of models, losses, datasets, and optimizers by name.
Registered components can be resolved at runtime from JSON config.
"""
from __future__ import annotations

from typing import Any, Callable


class RegistryError(RuntimeError):
    pass


class _ComponentRegistry:
    """Generic name-to-callable registry with conflict detection."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
        """Register a callable under ``name``.

        May be used as a decorator or as a direct call:
            @register_model("fno_forecast")
            def build_fn(...): ...

            register_model("fno_forecast", build_fn)
        """
        if name in self._entries:
            raise RegistryError(f"{self._kind} '{name}' is already registered")

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._entries[name] = func
            return func

        if fn is None:
            return _decorator
        return _decorator(fn)

    def get(self, name: str) -> Callable[..., Any]:
        """Lookup a registered callable by name."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries.keys())) or "(none)"
            raise RegistryError(f"unknown {self._kind}: '{name}'. Available: {available}")
        return self._entries[name]

    def list(self) -> list[str]:
        """Return sorted list of registered names."""
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# Global registries
_MODELS = _ComponentRegistry("model")
_LOSSES = _ComponentRegistry("loss")
_DATASETS = _ComponentRegistry("dataset")
_OPTIMIZERS = _ComponentRegistry("optimizer")


def register_model(name: str, fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    return _MODELS.register(name, fn)


def register_loss(name: str, fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    return _LOSSES.register(name, fn)


def register_dataset(name: str, fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    return _DATASETS.register(name, fn)


def register_optimizer(name: str, fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    return _OPTIMIZERS.register(name, fn)


def get_model(name: str) -> Callable[..., Any]:
    return _MODELS.get(name)


def get_loss(name: str) -> Callable[..., Any]:
    return _LOSSES.get(name)


def get_dataset(name: str) -> Callable[..., Any]:
    return _DATASETS.get(name)


def get_optimizer(name: str) -> Callable[..., Any]:
    return _OPTIMIZERS.get(name)


def list_models() -> list[str]:
    return _MODELS.list()


def list_losses() -> list[str]:
    return _LOSSES.list()


def list_datasets() -> list[str]:
    return _DATASETS.list()


def list_optimizers() -> list[str]:
    return _OPTIMIZERS.list()
