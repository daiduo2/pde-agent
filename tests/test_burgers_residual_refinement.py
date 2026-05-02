from __future__ import annotations

import numpy as np


def test_residual_guided_refinement_preserves_context_and_reduces_residual() -> None:
    from pde_agent.core.burgers_residual_refinement import (
        burgers_residual_summary,
        residual_guided_refinement,
    )

    context = np.full((2, 10, 256), 0.25, dtype=np.float32)
    base = np.empty((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    for step in range(190):
        base[:, 10 + step, :] = context[:, -1, :] + np.float32(0.002 * (step + 1))

    refined = residual_guided_refinement(
        context=context,
        base_prediction=base,
        viscosity=0.001,
        dt=0.05,
        dx=1.0 / 256.0,
        correction_weight=0.5,
        stop_step=200,
        spectral_strength=0.0,
    )

    before = burgers_residual_summary(
        base,
        viscosity=0.001,
        dt=0.05,
        dx=1.0 / 256.0,
    )
    after = burgers_residual_summary(
        refined,
        viscosity=0.001,
        dt=0.05,
        dx=1.0 / 256.0,
    )

    assert refined.shape == base.shape
    assert np.allclose(refined[:, :10, :], context, atol=1e-7)
    assert np.all(np.isfinite(refined))
    assert after["residual_rmse"] < before["residual_rmse"]
    assert after["early_residual_rmse"] < before["early_residual_rmse"]
