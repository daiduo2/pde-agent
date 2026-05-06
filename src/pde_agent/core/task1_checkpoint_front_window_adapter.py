"""Checkpoint front-window source adapters for Task 1 Hermes exploration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


class Task1CheckpointFrontWindowAdapterError(RuntimeError):
    pass


DEFAULT_WINDOWS = [(10, 20), (20, 35), (35, 57)]
DEFAULT_ADAPTER_KINDS = ["direct", "blend", "residual"]


def build_checkpoint_front_window_adapter_sources(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_prediction: np.ndarray,
    checkpoint_source: str,
    windows: Sequence[tuple[int, int]] | None = None,
    adapter_kinds: Sequence[str] | None = None,
    blend_weights: Sequence[float] | None = None,
    residual_scales: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    context = _as_context(context)
    base_prediction = _as_prediction(base_prediction, "base_prediction")
    checkpoint_prediction = _as_prediction(checkpoint_prediction, "checkpoint_prediction")
    if base_prediction.shape != checkpoint_prediction.shape:
        raise Task1CheckpointFrontWindowAdapterError(
            "base and checkpoint predictions must have matching shapes"
        )
    if context.shape[0] != base_prediction.shape[0]:
        raise Task1CheckpointFrontWindowAdapterError(
            "context and prediction sample counts must match"
        )
    window_rows = [_normalize_window(window) for window in (windows or DEFAULT_WINDOWS)]
    kinds = [str(kind) for kind in (adapter_kinds or DEFAULT_ADAPTER_KINDS)]
    for kind in kinds:
        if kind not in {"direct", "blend", "residual"}:
            raise Task1CheckpointFrontWindowAdapterError(
                "adapter kind must be direct, blend, or residual"
            )
    blend_rows = [float(value) for value in (blend_weights or [0.25, 0.5, 0.75])]
    residual_rows = [float(value) for value in (residual_scales or [0.25, 0.5, 0.75])]
    sources: list[dict[str, Any]] = []
    for start, stop in window_rows:
        if "direct" in kinds:
            sources.append(
                _source(
                    context=context,
                    base_prediction=base_prediction,
                    checkpoint_prediction=checkpoint_prediction,
                    checkpoint_source=checkpoint_source,
                    adapter_kind="direct",
                    start=start,
                    stop=stop,
                    weight=1.0,
                    prediction=_patch_window(
                        base_prediction=base_prediction,
                        patch=checkpoint_prediction[:, start:stop, :],
                        start=start,
                        stop=stop,
                    ),
                )
            )
        if "blend" in kinds:
            for weight in blend_rows:
                patch = (
                    (1.0 - weight) * base_prediction[:, start:stop, :]
                    + weight * checkpoint_prediction[:, start:stop, :]
                )
                sources.append(
                    _source(
                        context=context,
                        base_prediction=base_prediction,
                        checkpoint_prediction=checkpoint_prediction,
                        checkpoint_source=checkpoint_source,
                        adapter_kind="blend",
                        start=start,
                        stop=stop,
                        weight=weight,
                        prediction=_patch_window(
                            base_prediction=base_prediction,
                            patch=patch,
                            start=start,
                            stop=stop,
                        ),
                    )
                )
        if "residual" in kinds:
            residual = checkpoint_prediction[:, start:stop, :] - base_prediction[:, start:stop, :]
            for scale in residual_rows:
                patch = base_prediction[:, start:stop, :] + scale * residual
                sources.append(
                    _source(
                        context=context,
                        base_prediction=base_prediction,
                        checkpoint_prediction=checkpoint_prediction,
                        checkpoint_source=checkpoint_source,
                        adapter_kind="residual",
                        start=start,
                        stop=stop,
                        weight=scale,
                        prediction=_patch_window(
                            base_prediction=base_prediction,
                            patch=patch,
                            start=start,
                            stop=stop,
                        ),
                    )
                )
    return sources


def _source(
    *,
    context: np.ndarray,
    base_prediction: np.ndarray,
    checkpoint_prediction: np.ndarray,
    checkpoint_source: str,
    adapter_kind: str,
    start: int,
    stop: int,
    weight: float,
    prediction: np.ndarray,
) -> dict[str, Any]:
    prediction[:, :CONTEXT_STEPS, :] = context
    diff = checkpoint_prediction[:, start:stop, :] - base_prediction[:, start:stop, :]
    metadata = {
        "candidate_family": "task1_checkpoint_front_window_adapter",
        "adapter_kind": adapter_kind,
        "checkpoint_source": str(checkpoint_source),
        "uses_target_as_reference": False,
        "window": f"{start}:{stop}",
        "window_start": start,
        "window_stop": stop,
        "weight": float(weight),
        "mean_abs_checkpoint_delta": float(np.mean(np.abs(diff))),
        "rmse_checkpoint_delta": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
    }
    candidate_id = (
        "task1_checkpoint_front_window_adapter"
        f"_{adapter_kind}_w{start}_{stop}_a{float(weight):g}"
    )
    return {
        "candidate_id": candidate_id,
        "prediction": prediction.astype(np.float32, copy=False),
        "metadata": metadata,
    }


def _patch_window(
    *,
    base_prediction: np.ndarray,
    patch: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    prediction = np.array(base_prediction, copy=True, dtype=np.float32)
    prediction[:, start:stop, :] = patch
    return prediction


def _normalize_window(window: tuple[int, int]) -> tuple[int, int]:
    start, stop = int(window[0]), int(window[1])
    if start < CONTEXT_STEPS or stop <= start or stop > PREDICTION_STEPS:
        raise Task1CheckpointFrontWindowAdapterError(
            "window must satisfy context_steps <= start < stop <= prediction_steps"
        )
    return start, stop


def _as_context(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise Task1CheckpointFrontWindowAdapterError(
            f"context must have shape (N, {CONTEXT_STEPS}, {SPACE_POINTS})"
        )
    _require_finite(array, "context")
    return array


def _as_prediction(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1CheckpointFrontWindowAdapterError(
            f"{name} must have shape (N, {PREDICTION_STEPS}, {SPACE_POINTS})"
        )
    _require_finite(array, name)
    return array


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise Task1CheckpointFrontWindowAdapterError(
            f"{name} must contain only finite values"
        )
