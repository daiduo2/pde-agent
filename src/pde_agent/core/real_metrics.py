from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pde_agent.core.array_contract import BurgersArrayBundle, validate_burgers_bundle


REAL_DIAGNOSTIC_MODE = "real_array_diagnostic"
EPS = 1e-12


def rollout_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    validated = validate_burgers_bundle(bundle)
    prediction = validated.prediction
    target = validated.target
    time_count = prediction.shape[1]

    return {
        "mode": REAL_DIAGNOSTIC_MODE,
        "source_mode": str(validated.metadata["source_mode"]),
        "one_step": {"nrmse": _nrmse(prediction[:, :1, :], target[:, :1, :])},
        "long_horizon": {"nrmse": _nrmse(prediction, target)},
        "segments": _segment_metrics(prediction, target),
        "curve": {
            "time_index": list(range(time_count)),
            "nrmse": [
                _nrmse(prediction[:, index, :], target[:, index, :])
                for index in range(time_count)
            ],
        },
        "prediction": {"path": None},
    }


def spectrum_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    validated = validate_burgers_bundle(bundle)
    prediction_spectrum = np.abs(np.fft.rfft(validated.prediction, axis=-1))
    target_spectrum = np.abs(np.fft.rfft(validated.target, axis=-1))
    error_spectrum = np.abs(prediction_spectrum - target_spectrum)
    bands = _frequency_bands(error_spectrum.shape[-1])

    low_error = _relative_band_error(error_spectrum, target_spectrum, bands["low"])
    mid_error = _relative_band_error(error_spectrum, target_spectrum, bands["mid"])
    high_error = _relative_band_error(error_spectrum, target_spectrum, bands["high"])
    high_to_low_ratio = _safe_float(high_error / (low_error + EPS))
    spectral_drift = _safe_float(
        np.linalg.norm(prediction_spectrum - target_spectrum)
        / (np.linalg.norm(target_spectrum) + EPS)
    )

    band_errors = {
        "low_frequency_error": low_error,
        "mid_frequency_error": mid_error,
        "high_frequency_error": high_error,
    }
    dominant_issue = max(band_errors, key=band_errors.get)

    return {
        **band_errors,
        "high_to_low_error_ratio": high_to_low_ratio,
        "spectral_drift_score": spectral_drift,
        "dominant_spectral_issue": dominant_issue,
        "metric_definition": (
            "Deterministic diagnostic proxy using np.fft.rfft over the spatial "
            "axis; band errors are relative RMS spectral-amplitude errors."
        ),
    }


def physics_metrics(bundle: BurgersArrayBundle) -> dict[str, object]:
    validated = validate_burgers_bundle(bundle)
    prediction = validated.prediction
    target = validated.target

    target_mass = np.mean(target, axis=-1)
    prediction_mass = np.mean(prediction, axis=-1)
    mass_drift = _safe_float(
        np.max(np.abs(prediction_mass - target_mass))
        / (np.max(np.abs(target_mass)) + EPS)
    )

    prediction_energy = np.mean(prediction**2, axis=-1)
    target_energy = np.mean(target**2, axis=-1)
    energy_decay_ratio = _safe_float(
        np.max(prediction_energy) / (np.mean(target_energy) + EPS)
    )

    prediction_tv = _total_variation(prediction)
    target_tv = _total_variation(target)
    total_variation_ratio = _safe_float(
        np.mean(prediction_tv) / (np.mean(target_tv) + EPS)
    )

    sign_changes = np.diff(np.signbit(prediction), axis=-1)
    oscillation_fraction = _safe_float(np.mean(sign_changes))
    nonphysical_oscillation = bool(oscillation_fraction > 0.35)
    blowup_like_behavior = bool(
        energy_decay_ratio > 4.0
        or np.max(np.abs(prediction)) > 4.0 * (np.max(np.abs(target)) + EPS)
    )
    excessive_smoothing = bool(
        total_variation_ratio < 0.35 and energy_decay_ratio < 0.5
    )

    physics_issue = "none"
    if blowup_like_behavior:
        physics_issue = "blowup_like_behavior"
    elif nonphysical_oscillation:
        physics_issue = "nonphysical_oscillation"
    elif excessive_smoothing:
        physics_issue = "excessive_smoothing"
    elif mass_drift > 0.5:
        physics_issue = "mass_drift"

    return {
        "mass_drift": mass_drift,
        "energy_decay_ratio": energy_decay_ratio,
        "total_variation_ratio": total_variation_ratio,
        "nonphysical_oscillation": nonphysical_oscillation,
        "blowup_like_behavior": blowup_like_behavior,
        "excessive_smoothing": excessive_smoothing,
        "physics_issue": physics_issue,
        "proxy_notice": (
            "These are deterministic diagnostic proxy metrics, not formal PDE "
            "residual validation."
        ),
    }


def _nrmse(prediction: NDArray[np.floating], target: NDArray[np.floating]) -> float:
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    scale = np.sqrt(np.mean(target**2))
    return _safe_float(rmse / (scale + EPS))


def _segment_metrics(
    prediction: NDArray[np.floating], target: NDArray[np.floating]
) -> dict[str, dict[str, float]]:
    slices = _time_segments(prediction.shape[1])
    return {
        name: {"nrmse": _nrmse(prediction[:, segment, :], target[:, segment, :])}
        for name, segment in slices.items()
    }


def _time_segments(time_count: int) -> dict[str, slice]:
    split_points = np.array_split(np.arange(time_count), 3)
    names = ("early", "middle", "late")
    segments: dict[str, slice] = {}
    fallback = np.arange(max(time_count - 1, 0), time_count)
    for name, indexes in zip(names, split_points, strict=True):
        segment_indexes = indexes if indexes.size else fallback
        start = int(segment_indexes[0])
        stop = int(segment_indexes[-1]) + 1
        segments[name] = slice(start, stop)
    return segments


def _frequency_bands(bin_count: int) -> dict[str, NDArray[np.integer]]:
    split_bins = np.array_split(np.arange(bin_count), 3)
    fallback = np.arange(max(bin_count - 1, 0), bin_count)
    return {
        "low": split_bins[0] if split_bins[0].size else fallback,
        "mid": split_bins[1] if split_bins[1].size else fallback,
        "high": split_bins[2] if split_bins[2].size else fallback,
    }


def _relative_band_error(
    error_spectrum: NDArray[np.floating],
    target_spectrum: NDArray[np.floating],
    band: NDArray[np.integer],
) -> float:
    error = np.sqrt(np.mean(error_spectrum[..., band] ** 2))
    scale = np.sqrt(np.mean(target_spectrum[..., band] ** 2))
    return _safe_float(error / (scale + EPS))


def _total_variation(array: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.sum(np.abs(np.diff(array, axis=-1)), axis=-1)


def _safe_float(value: object) -> float:
    return float(
        np.nan_to_num(value, nan=0.0, posinf=1.0 / EPS, neginf=-1.0 / EPS)
    )
