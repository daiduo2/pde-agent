from __future__ import annotations

import numpy as np


def _linear_transition_tensor(*, sample_count: int, phase_offset: float) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    tensor = np.empty((sample_count, 200, 256), dtype=np.float32)
    for sample in range(sample_count):
        phase = np.float32(phase_offset + 0.07 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (
            np.float32(0.006 + 0.001 * sample) * np.cos(x + phase)
            + np.float32(0.002)
        ).astype(np.float32)
        for step in range(200):
            tensor[sample, step, :] = base + np.float32(step) * velocity
    return tensor


def test_task1_early_transition_expert_learns_short_window_velocity_map() -> None:
    from pde_agent.core.manual_operator_tuning import (
        make_persistence_prediction,
        score_prediction_arrays,
    )
    from pde_agent.core.task1_early_transition_expert import (
        fit_task1_early_transition_expert,
    )

    target = _linear_transition_tensor(sample_count=3, phase_offset=0.01)
    reference = _linear_transition_tensor(sample_count=8, phase_offset=0.0)
    context = target[:, :10, :]
    base_prediction = make_persistence_prediction(context)

    result = fit_task1_early_transition_expert(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        k=4,
        horizon_steps=47,
        ridge_alpha=1e-8,
        correction_weight=1.0,
        transition_mode="velocity_ridge",
    )

    prediction = result["prediction"]
    persistence_score = score_prediction_arrays(
        prediction=base_prediction,
        target=target,
    )
    expert_score = score_prediction_arrays(prediction=prediction, target=target)

    assert np.allclose(prediction[:, :10, :], context, atol=1e-6)
    assert expert_score["segments"]["early"]["rel_mse"] < (
        persistence_score["segments"]["early"]["rel_mse"] * 0.05
    )
    assert result["metadata"]["transition_mode"] == "velocity_ridge"
    assert result["metadata"]["horizon_steps"] == 47
    assert result["metadata"]["reference_count"] == 8


def test_task1_early_transition_expert_accepts_reduced_time_reference() -> None:
    from pde_agent.core.manual_operator_tuning import make_persistence_prediction
    from pde_agent.core.task1_early_transition_expert import (
        fit_task1_early_transition_expert,
    )

    target = _linear_transition_tensor(sample_count=3, phase_offset=0.01)
    reference = _linear_transition_tensor(sample_count=8, phase_offset=0.0)[:, :41, :]
    context = target[:, :10, :]
    base_prediction = make_persistence_prediction(context)

    result = fit_task1_early_transition_expert(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        k=4,
        horizon_steps=47,
        ridge_alpha=1e-8,
        correction_weight=1.0,
        transition_mode="velocity_ridge",
    )

    prediction = result["prediction"]

    assert np.allclose(prediction[:, :10, :], context, atol=1e-6)
    assert np.allclose(prediction[:, 41:, :], base_prediction[:, 41:, :], atol=1e-6)
    assert result["metadata"]["horizon_steps"] == 31
    assert result["metadata"]["reference_time_steps"] == 41
