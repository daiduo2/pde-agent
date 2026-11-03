from __future__ import annotations

from collections.abc import Iterable


STATE_HINT = "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
FAMILY_SELECTION_STATE_HINT = "TASK1_SHORT_WINDOW_FAMILY_SELECTION_RAN"
SUPPORTED_NORMALIZATION_TYPES = {"none", "sample_standard"}


class _DynamicSet:
    """Set-like object that resolves its contents from the registry on demand."""

    def __init__(self, getter):
        self._getter = getter

    def _values(self):
        return self._getter()

    def __contains__(self, item):
        return item in self._values()

    def __iter__(self):
        return iter(self._values())

    def __len__(self):
        return len(self._values())

    def __sub__(self, other):
        if isinstance(other, set):
            return self._values() - other
        return self._values() - set(other)

    def __rsub__(self, other):
        if isinstance(other, set):
            return other - self._values()
        return set(other) - self._values()

    def __le__(self, other):
        return self._values() <= set(other)

    def __ge__(self, other):
        return self._values() >= set(other)

    def issubset(self, other):
        return self._values().issubset(other)

    def issuperset(self, other):
        return self._values().issuperset(other)


def _supported_model_families() -> set[str]:
    """Return registered model families from the component registry."""
    try:
        from pde_agent.core.neural_operator.registry import list_models
        return set(list_models())
    except Exception:
        return {"conv1d_residual", "fno_ar1", "fno_lite"}


def _supported_loss_types() -> set[str]:
    """Return registered loss types from the component registry."""
    try:
        from pde_agent.core.neural_operator.registry import list_losses
        return set(list_losses())
    except Exception:
        return {"mse", "early_weighted_mse", "horizon_weighted_mse", "spectral_mse"}


SUPPORTED_MODEL_FAMILIES = _DynamicSet(_supported_model_families)
SUPPORTED_FNO_LOSS_TYPES = _DynamicSet(_supported_loss_types)


class Task1ShortWindowTrainingError(RuntimeError):
    pass
