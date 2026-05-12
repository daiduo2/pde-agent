"""Manual Burgers operator tuning utilities for validation-set diagnosis."""


import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pde_agent.core.burgers_hdf5_io import load_hdf5_tensor
from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


FUTURE_STEPS = PREDICTION_STEPS - CONTEXT_STEPS
EARLY_FUTURE_STOP = 47
MIDDLE_FUTURE_STOP = 95


class ManualOperatorTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    family: str
    params: dict[str, Any]
    elapsed_seconds: float
    segment_score: float
    task1_prediction_score: float
    task1_total_fast_proxy: float
    segments: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "params": self.params,
            "elapsed_seconds": self.elapsed_seconds,
            "segment_score": self.segment_score,
            "task1_prediction_score": self.task1_prediction_score,
            "task1_total_fast_proxy": self.task1_total_fast_proxy,
            "segments": self.segments,
        }


def score_burgers_segments(*, prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = _as_prediction_array(prediction, "prediction")
    target = _as_prediction_array(target, "target")
    if prediction.shape != target.shape:
        raise ManualOperatorTuningError(
            f"shape mismatch: prediction {prediction.shape} != target {target.shape}"
        )

    squared_error = (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    target_squared = target.astype(np.float64) ** 2
    early = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=0,
        future_stop=EARLY_FUTURE_STOP,
    )
    middle = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=EARLY_FUTURE_STOP,
        future_stop=MIDDLE_FUTURE_STOP,
    )
    late = _late_segment(
        squared_error=squared_error,
        future_start=MIDDLE_FUTURE_STOP,
        future_stop=FUTURE_STEPS,
    )
    segment_score = float(
        0.25 * early["score"] + 0.25 * middle["score"] + 0.5 * late["score"]
    )
    return {
        "segment_score": segment_score,
        "task1_prediction_score": float(segment_score * 0.75),
        "segments": {
            "early": early,
            "middle": middle,
            "late": late,
        },
    }


def make_persistence_prediction(context: np.ndarray) -> np.ndarray:
    context = _as_context_array(context)
    prediction = np.empty(
        (context.shape[0], PREDICTION_STEPS, SPACE_POINTS), dtype=np.float32
    )
    prediction[:, :CONTEXT_STEPS, :] = context
    prediction[:, CONTEXT_STEPS:, :] = context[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :]
    return prediction


def spectral_smooth_prediction(*, prediction: np.ndarray, strength: float) -> np.ndarray:
    prediction = _as_prediction_array(prediction, "prediction")
    if strength < 0.0:
        raise ManualOperatorTuningError("strength must be non-negative")
    transformed = np.array(prediction, copy=True)
    transformed[:, CONTEXT_STEPS:, :] = _spectral_smooth_future(
        future=transformed[:, CONTEXT_STEPS:, :],
        strength=strength,
    )
    return np.asarray(transformed, dtype=np.float32)


def linear_trend_prediction(
    *,
    context: np.ndarray,
    gain: float,
    slope_window: int,
    decay: float | None,
    clip: tuple[float, float] | None,
) -> np.ndarray:
    context = _as_context_array(context)
    if slope_window <= 0 or slope_window >= CONTEXT_STEPS:
        raise ManualOperatorTuningError("slope_window must be in [1, 9]")
    last = context[:, -1:, :]
    slope = (context[:, -1:, :] - context[:, -1 - slope_window : -slope_window, :]) / float(
        slope_window
    )
    horizon = np.arange(1, FUTURE_STEPS + 1, dtype=np.float32).reshape(1, -1, 1)
    if decay is not None:
        if decay <= 0.0:
            raise ManualOperatorTuningError("decay must be positive")
        horizon = horizon * np.exp(-horizon / float(decay))
    future = last + float(gain) * horizon * slope
    if clip is not None:
        future = np.clip(future, clip[0], clip[1])
    prediction = make_persistence_prediction(context)
    prediction[:, CONTEXT_STEPS:, :] = future
    return prediction.astype(np.float32)


def burgers_fd_prediction(
    *,
    context: np.ndarray,
    nu: float,
    dt: float,
    method: str,
    blend_with_persistence: float,
    clip: tuple[float, float] | None,
) -> np.ndarray:
    context = _as_context_array(context)
    if nu < 0.0:
        raise ManualOperatorTuningError("nu must be non-negative")
    if dt <= 0.0:
        raise ManualOperatorTuningError("dt must be positive")
    if method not in {"central", "upwind"}:
        raise ManualOperatorTuningError("method must be central or upwind")
    if not 0.0 <= blend_with_persistence <= 1.0:
        raise ManualOperatorTuningError("blend_with_persistence must be in [0, 1]")

    u = context[:, -1, :].astype(np.float64)
    dx = 1.0 / float(SPACE_POINTS)
    future = np.empty((context.shape[0], FUTURE_STEPS, SPACE_POINTS), dtype=np.float64)
    for step in range(FUTURE_STEPS):
        right = np.roll(u, -1, axis=1)
        left = np.roll(u, 1, axis=1)
        if method == "central":
            ux = (right - left) / (2.0 * dx)
        else:
            backward = (u - left) / dx
            forward = (right - u) / dx
            ux = np.where(u >= 0.0, backward, forward)
        uxx = (right - 2.0 * u + left) / (dx * dx)
        u = u - float(dt) * u * ux + float(nu) * float(dt) * uxx
        if clip is not None:
            u = np.clip(u, clip[0], clip[1])
        future[:, step, :] = u

    persistence_future = np.repeat(context[:, -1:, :], FUTURE_STEPS, axis=1).astype(
        np.float64
    )
    future = (
        float(blend_with_persistence) * persistence_future
        + (1.0 - float(blend_with_persistence)) * future
    )
    prediction = make_persistence_prediction(context)
    prediction[:, CONTEXT_STEPS:, :] = future
    return prediction.astype(np.float32)


def blend_predictions(
    *,
    left: np.ndarray,
    right: np.ndarray,
    weight_right: float,
) -> np.ndarray:
    left = _as_prediction_array(left, "left")
    right = _as_prediction_array(right, "right")
    if left.shape != right.shape:
        raise ManualOperatorTuningError(f"shape mismatch: {left.shape} != {right.shape}")
    if not 0.0 <= weight_right <= 1.0:
        raise ManualOperatorTuningError("weight_right must be in [0, 1]")
    blended = np.array(left, copy=True, dtype=np.float32)
    blended[:, CONTEXT_STEPS:, :] = (
        (1.0 - float(weight_right)) * left[:, CONTEXT_STEPS:, :]
        + float(weight_right) * right[:, CONTEXT_STEPS:, :]
    )
    return blended.astype(np.float32)


def evaluate_manual_candidate_grid(
    *,
    context: np.ndarray,
    target: np.ndarray,
    external_predictions: dict[str, np.ndarray] | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    context = _as_context_array(context)
    target = _as_prediction_array(target, "target")
    if context.shape[0] != target.shape[0]:
        raise ManualOperatorTuningError("context and target sample counts differ")

    clip = (float(np.min(context)), float(np.max(context)))
    candidates = list(_iter_manual_candidates(context=context, clip=clip))
    if external_predictions:
        candidates.extend(_iter_external_blends(context=context, external_predictions=external_predictions))

    results: list[CandidateResult] = []
    for candidate_id, family, params, prediction_factory in candidates:
        start = time.perf_counter()
        prediction = prediction_factory()
        elapsed = time.perf_counter() - start
        score = score_burgers_segments(prediction=prediction, target=target)
        results.append(
            CandidateResult(
                candidate_id=candidate_id,
                family=family,
                params=params,
                elapsed_seconds=float(elapsed),
                segment_score=float(score["segment_score"]),
                task1_prediction_score=float(score["task1_prediction_score"]),
                task1_total_fast_proxy=_task1_total_fast_proxy(
                    task1_prediction_score=float(score["task1_prediction_score"]),
                    inference_seconds=float(elapsed),
                ),
                segments=score["segments"],
            )
        )

    ranked = sorted(results, key=lambda row: row.task1_total_fast_proxy, reverse=True)
    best = ranked[0]
    return {
        "schema_version": 1,
        "state_hint": "MANUAL_OPERATOR_TUNING_COMPLETE",
        "sample_count": int(context.shape[0]),
        "candidate_count": len(results),
        "best_candidate": best.to_json(),
        "top_candidates": [row.to_json() for row in ranked[:top_k]],
        "all_candidates": [row.to_json() for row in ranked],
        "notes": (
            "task1_total_fast_proxy uses train_time=0 and the measured numpy "
            "candidate generation time as inference time. It is a local proxy "
            "for comparing lightweight operator/postprocess ideas, not a "
            "replacement for a packaged submission timing audit."
        ),
    }


def write_report_artifacts(*, report: dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manual_operator_tuning_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "manual_operator_tuning_top.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "candidate_id",
                "family",
                "segment_score",
                "task1_prediction_score",
                "task1_total_fast_proxy",
                "elapsed_seconds",
                "early_rel_mse",
                "middle_rel_mse",
                "late_rmse",
                "params",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(report["top_candidates"], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "segment_score": row["segment_score"],
                    "task1_prediction_score": row["task1_prediction_score"],
                    "task1_total_fast_proxy": row["task1_total_fast_proxy"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "early_rel_mse": row["segments"]["early"]["rel_mse"],
                    "middle_rel_mse": row["segments"]["middle"]["rel_mse"],
                    "late_rmse": row["segments"]["late"]["rmse"],
                    "params": json.dumps(row["params"], sort_keys=True),
                }
            )


def _iter_manual_candidates(
    *,
    context: np.ndarray,
    clip: tuple[float, float],
) -> Iterable[tuple[str, str, dict[str, Any], Any]]:
    persistence = make_persistence_prediction(context)
    yield "persistence", "fast_baseline", {}, lambda: persistence

    for strength in [
        0.015625,
        0.03125,
        0.0625,
        0.09375,
        0.125,
        0.15625,
        0.1875,
        0.25,
        0.375,
        0.5,
        0.75,
        1.0,
    ]:
        yield (
            f"persistence_spectral_smooth_w{strength:g}",
            "spectral_smooth",
            {"strength": strength},
            lambda strength=strength: spectral_smooth_prediction(
                prediction=persistence,
                strength=strength,
            ),
        )

    for slope_window in [1, 2, 3, 5, 8]:
        for gain in [-0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]:
            for decay in [10.0, 25.0, 50.0, 100.0, None]:
                decay_id = "none" if decay is None else f"{decay:g}"
                yield (
                    f"linear_trend_w{slope_window}_g{gain:g}_d{decay_id}",
                    "linear_trend",
                    {"slope_window": slope_window, "gain": gain, "decay": decay},
                    lambda slope_window=slope_window, gain=gain, decay=decay: linear_trend_prediction(
                        context=context,
                        gain=gain,
                        slope_window=slope_window,
                        decay=decay,
                        clip=clip,
                    ),
                )

    for slope_window in [4, 5, 6, 7]:
        for gain in [-0.2, -0.175, -0.15, -0.125, -0.1, -0.075]:
            for decay in [35.0, 40.0, 45.0, 50.0, 60.0, 70.0]:
                yield (
                    f"linear_trend_fine_w{slope_window}_g{gain:g}_d{decay:g}",
                    "linear_trend_fine",
                    {"slope_window": slope_window, "gain": gain, "decay": decay},
                    lambda slope_window=slope_window, gain=gain, decay=decay: linear_trend_prediction(
                        context=context,
                        gain=gain,
                        slope_window=slope_window,
                        decay=decay,
                        clip=clip,
                    ),
                )

    for slope_window in [4, 5, 6]:
        for gain in [-0.15, -0.125, -0.1]:
            for decay in [40.0, 50.0, 60.0]:
                for strength in [0.125, 0.25, 0.5, 0.75, 1.0]:
                    yield (
                        "linear_trend_spectral"
                        f"_w{slope_window}_g{gain:g}_d{decay:g}_s{strength:g}",
                        "linear_trend_spectral",
                        {
                            "slope_window": slope_window,
                            "gain": gain,
                            "decay": decay,
                            "spectral_strength": strength,
                        },
                        lambda slope_window=slope_window, gain=gain, decay=decay, strength=strength: spectral_smooth_prediction(
                            prediction=linear_trend_prediction(
                                context=context,
                                gain=gain,
                                slope_window=slope_window,
                                decay=decay,
                                clip=clip,
                            ),
                            strength=strength,
                        ),
                    )

    for method in ["central", "upwind"]:
        for nu in [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            for dt in [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]:
                for blend in [0.0, 0.5, 0.75, 0.9, 0.97]:
                    yield (
                        f"burgers_fd_{method}_nu{nu:g}_dt{dt:g}_blend{blend:g}",
                        "burgers_fd",
                        {
                            "method": method,
                            "nu": nu,
                            "dt": dt,
                            "blend_with_persistence": blend,
                        },
                        lambda method=method, nu=nu, dt=dt, blend=blend: burgers_fd_prediction(
                            context=context,
                            nu=nu,
                            dt=dt,
                            method=method,
                            blend_with_persistence=blend,
                            clip=clip,
                        ),
                    )


def _iter_external_blends(
    *,
    context: np.ndarray,
    external_predictions: dict[str, np.ndarray],
) -> Iterable[tuple[str, str, dict[str, Any], Any]]:
    persistence = make_persistence_prediction(context)
    for name, external in external_predictions.items():
        external = _as_prediction_array(external, name)
        for weight in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
            yield (
                f"blend_persistence_{name}_w{weight:g}",
                "external_blend",
                {"external": name, "weight_right": weight},
                lambda external=external, weight=weight: blend_predictions(
                    left=persistence,
                    right=external,
                    weight_right=weight,
                ),
            )


def _spectral_smooth_future(*, future: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.array(future, copy=True)
    spectrum = np.fft.rfft(future, axis=2)
    freqs = np.linspace(0.0, 1.0, spectrum.shape[2], dtype=np.float64)
    damping = np.exp(-float(strength) * 8.0 * freqs**2)
    spectrum = spectrum * damping.reshape(1, 1, -1)
    smoothed = np.fft.irfft(spectrum, n=future.shape[2], axis=2)
    return np.asarray(smoothed, dtype=np.float32)


def _rel_mse_segment(
    *,
    squared_error: np.ndarray,
    target_squared: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    absolute_start = CONTEXT_STEPS + future_start
    absolute_stop = CONTEXT_STEPS + future_stop
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_by_sample_time = (
            np.sum(squared_error[:, absolute_start:absolute_stop, :], axis=2)
            / (
                np.sum(target_squared[:, absolute_start:absolute_stop, :], axis=2)
                + 1e-12
            )
        )
    _raise_if_non_finite(rel_by_sample_time, "segment relative MSE")
    rel_by_sample = np.mean(rel_by_sample_time, axis=1)
    rel_mse = float(np.mean(np.minimum(rel_by_sample, 5.0)))
    decay = 20.0 if future_start == 0 else 10.0
    score = float(100.0 * math.exp(-decay * rel_mse))
    _raise_if_non_finite([rel_mse, score], "segment score")
    return {
        "time_start": absolute_start,
        "time_stop": absolute_stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rel_mse": rel_mse,
        "score": score,
    }


def _late_segment(
    *,
    squared_error: np.ndarray,
    future_start: int,
    future_stop: int,
) -> dict[str, Any]:
    absolute_start = CONTEXT_STEPS + future_start
    absolute_stop = CONTEXT_STEPS + future_stop
    rmse = float(np.sqrt(np.mean(squared_error[:, absolute_start:absolute_stop, :])))
    score = float(100.0 / (1.0 + 10.0 * rmse))
    _raise_if_non_finite([rmse, score], "late segment score")
    return {
        "time_start": absolute_start,
        "time_stop": absolute_stop,
        "future_start": future_start,
        "future_stop": future_stop,
        "rmse": rmse,
        "lorentzian_score": score,
        "frechet_score": None,
        "score": score,
    }


def _task1_total_fast_proxy(
    *,
    task1_prediction_score: float,
    inference_seconds: float,
) -> float:
    inference_score = 0.0
    if inference_seconds <= 120.0:
        inference_score = 40.0 * max(0.0, 1.0 - inference_seconds / 120.0)
    # Keep the existing local judge convention: stated rules say 45 max, but
    # threshold table gives 35 points for <=60 minutes.
    return float(task1_prediction_score + 35.0 + inference_score)


def _as_context_array(context: np.ndarray) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3 or context.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise ManualOperatorTuningError(
            f"expected context shape [N, 10, 256], got {context.shape}"
        )
    if context.shape[0] <= 0:
        raise ManualOperatorTuningError("context batch is empty")
    if not np.isfinite(context).all():
        raise ManualOperatorTuningError("context contains NaN or Inf")
    return context


def _as_prediction_array(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise ManualOperatorTuningError(
            f"expected {label} shape [N, 200, 256], got {value.shape}"
        )
    if value.shape[0] <= 0:
        raise ManualOperatorTuningError(f"{label} batch is empty")
    if not np.isfinite(value).all():
        raise ManualOperatorTuningError(f"{label} contains NaN or Inf")
    return value


def _raise_if_non_finite(value: Any, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ManualOperatorTuningError(f"{label} contains non-finite values")
