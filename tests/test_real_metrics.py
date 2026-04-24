import math

import numpy as np

from pde_agent.core.array_contract import BurgersArrayBundle
from pde_agent.core.real_metrics import (
    REAL_DIAGNOSTIC_MODE,
    physics_metrics,
    rollout_metrics,
    spectrum_metrics,
)


def _bundle(
    prediction: np.ndarray,
    target: np.ndarray,
    context: np.ndarray | None = None,
) -> BurgersArrayBundle:
    if context is None:
        context = np.zeros((prediction.shape[0], 2, prediction.shape[2]))
    return BurgersArrayBundle(
        context=context,
        prediction=prediction,
        target=target,
        full_reference=None,
        metadata={"problem_id": "burgers_1d", "source_mode": "pdebench_hdf5"},
    )


def _nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((prediction - target) ** 2))
        / (np.sqrt(np.mean(target**2)) + 1e-12)
    )


def test_rollout_metrics_compute_nrmse_and_curve() -> None:
    target = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )
    prediction = target + np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 1.0, -1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ]
    )

    metrics = rollout_metrics(_bundle(prediction, target))

    assert metrics["mode"] == REAL_DIAGNOSTIC_MODE
    assert metrics["source_mode"] == "pdebench_hdf5"
    assert metrics["prediction"] == {"path": None}
    assert metrics["one_step"]["nrmse"] == _nrmse(
        prediction[:, :1, :], target[:, :1, :]
    )
    assert metrics["long_horizon"]["nrmse"] == _nrmse(prediction, target)
    assert metrics["curve"]["time_index"] == [0, 1, 2]
    assert metrics["curve"]["nrmse"] == [
        _nrmse(prediction[:, 0, :], target[:, 0, :]),
        _nrmse(prediction[:, 1, :], target[:, 1, :]),
        _nrmse(prediction[:, 2, :], target[:, 2, :]),
    ]
    assert metrics["segments"] == {
        "early": {"nrmse": _nrmse(prediction[:, :1, :], target[:, :1, :])},
        "middle": {"nrmse": _nrmse(prediction[:, 1:2, :], target[:, 1:2, :])},
        "late": {"nrmse": _nrmse(prediction[:, 2:, :], target[:, 2:, :])},
    }


def test_spectrum_metrics_identify_deterministic_band_errors() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target = np.sin(x)[None, None, :]
    prediction = target + 0.5 * np.sin(7.0 * x)[None, None, :]

    metrics = spectrum_metrics(_bundle(prediction, target))

    assert metrics["dominant_spectral_issue"] == "high_frequency_error"
    assert metrics["high_frequency_error"] > metrics["low_frequency_error"]
    assert metrics["high_frequency_error"] > metrics["mid_frequency_error"]
    assert metrics["high_to_low_error_ratio"] > 1.0
    assert metrics["spectral_drift_score"] > 0.0
    assert "rfft" in metrics["metric_definition"]


def test_physics_metrics_compute_proxy_flags() -> None:
    target = np.ones((1, 4, 8))
    prediction = np.array(
        [
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
                [4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0],
                [8.0, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0, -8.0],
            ]
        ]
    )

    metrics = physics_metrics(_bundle(prediction, target))

    assert metrics["mass_drift"] > 0.9
    assert metrics["energy_decay_ratio"] > 1.0
    assert metrics["total_variation_ratio"] > 1.0
    assert metrics["nonphysical_oscillation"] is True
    assert metrics["blowup_like_behavior"] is True
    assert metrics["excessive_smoothing"] is False
    assert metrics["physics_issue"] == "blowup_like_behavior"
    assert "diagnostic proxy" in metrics["proxy_notice"]


def test_metrics_remain_finite_for_zero_target_scale() -> None:
    target = np.zeros((1, 3, 8))
    prediction = np.ones((1, 3, 8))

    metric_groups = [
        rollout_metrics(_bundle(prediction, target)),
        spectrum_metrics(_bundle(prediction, target)),
        physics_metrics(_bundle(prediction, target)),
    ]

    def assert_finite(value: object) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, float):
            assert math.isfinite(value)
            return
        if isinstance(value, int):
            return
        if isinstance(value, list):
            for item in value:
                assert_finite(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                assert_finite(item)
            return
        raise AssertionError(f"unexpected value type {type(value)!r}")

    for metrics in metric_groups:
        assert_finite(metrics)
