from __future__ import annotations

import numpy as np


def test_sem_burgers_rollout_preserves_constant_state_and_context() -> None:
    from pde_agent.core.burgers_spectral_element import sem_burgers_rollout

    context = np.full((2, 10, 256), 0.37, dtype=np.float32)

    prediction = sem_burgers_rollout(
        context=context,
        viscosity=0.001,
        dt=0.05,
        element_count=16,
        polynomial_order=8,
        substeps=2,
        filter_strength=0.0,
    )

    assert prediction.shape == (2, 200, 256)
    assert prediction.dtype == np.float32
    assert np.all(np.isfinite(prediction))
    assert np.allclose(prediction[:, :10, :], context, atol=1e-7)
    assert np.allclose(prediction[:, 10:, :], 0.37, atol=1e-5)


def test_sem_burgers_rollout_moves_nonconstant_state() -> None:
    from pde_agent.core.burgers_spectral_element import sem_burgers_rollout

    x = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    context = np.empty((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[0, step, :] = 0.4 + 0.15 * np.sin(2.0 * np.pi * (x - 0.01 * step))

    prediction = sem_burgers_rollout(
        context=context,
        viscosity=0.001,
        dt=0.02,
        element_count=16,
        polynomial_order=8,
        substeps=4,
        filter_strength=0.02,
    )

    assert np.allclose(prediction[:, :10, :], context, atol=1e-7)
    assert np.all(np.isfinite(prediction))
    assert not np.allclose(prediction[:, 10, :], context[:, -1, :])
