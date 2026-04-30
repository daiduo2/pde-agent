from __future__ import annotations

import numpy as np


def test_score_prediction_arrays_matches_perfect_prediction() -> None:
    from pde_agent.core.manual_operator_tuning import score_prediction_arrays

    target = np.ones((2, 200, 256), dtype=np.float32)
    report = score_prediction_arrays(prediction=target.copy(), target=target)

    assert report["segment_score"] == 100.0
    assert report["task1_prediction_score"] == 75.0
    assert report["segments"]["early"]["rel_mse"] == 0.0


def test_make_persistence_prediction_preserves_context_and_repeats_last_step() -> None:
    from pde_agent.core.manual_operator_tuning import make_persistence_prediction

    context = np.arange(2 * 10 * 256, dtype=np.float32).reshape(2, 10, 256)
    prediction = make_persistence_prediction(context)

    assert prediction.shape == (2, 200, 256)
    np.testing.assert_array_equal(prediction[:, :10, :], context)
    np.testing.assert_array_equal(
        prediction[:, 10:, :],
        np.repeat(context[:, 9:10, :], 190, axis=1),
    )


def test_spectral_smooth_prediction_is_finite_shape_preserving_postprocess() -> None:
    from pde_agent.core.manual_operator_tuning import (
        make_persistence_prediction,
        spectral_smooth_prediction,
    )

    context = np.random.default_rng(7).normal(size=(3, 10, 256)).astype(np.float32)
    prediction = make_persistence_prediction(context)
    smoothed = spectral_smooth_prediction(prediction=prediction, strength=0.25)

    assert smoothed.shape == prediction.shape
    np.testing.assert_array_equal(smoothed[:, :10, :], prediction[:, :10, :])
    assert np.isfinite(smoothed).all()
