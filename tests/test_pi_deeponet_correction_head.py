from __future__ import annotations

import numpy as np


def test_pi_deeponet_correction_head_preserves_context_and_learns_low_rank_correction() -> None:
    from pde_agent.core.pi_deeponet_correction_head import (
        fit_pi_deeponet_correction_head,
    )

    x = np.linspace(0.0, 1.0, 32, endpoint=False, dtype=np.float32)
    context = np.empty((4, 10, 32), dtype=np.float32)
    for sample in range(4):
        amplitude = np.float32(0.2 + 0.05 * sample)
        for step in range(10):
            context[sample, step, :] = 0.3 + amplitude * np.sin(
                2.0 * np.pi * (x + 0.01 * step)
            )
    base = np.repeat(context[:, -1:, :], 200, axis=1).astype(np.float32)
    base[:, :10, :] = context
    target = base.copy()
    horizon = np.linspace(0.0, 1.0, 190, dtype=np.float32).reshape(1, 190, 1)
    spatial = np.sin(2.0 * np.pi * x).reshape(1, 1, 32)
    sample_weight = np.mean(context[:, -1:, :], axis=2, keepdims=True)
    target[:, 10:, :] += np.float32(0.08) * sample_weight * horizon * spatial

    result = fit_pi_deeponet_correction_head(
        context=context,
        base_prediction=base,
        target=target,
        ridge_alpha=1e-6,
        correction_weight=1.0,
        physics_weight=0.0,
        viscosity=0.001,
        dt=0.05,
        dx=1.0 / 32.0,
    )

    base_mse = float(np.mean((base[:, 10:, :] - target[:, 10:, :]) ** 2))
    pred_mse = float(
        np.mean((result["prediction"][:, 10:, :] - target[:, 10:, :]) ** 2)
    )
    assert result["prediction"].shape == base.shape
    assert np.allclose(result["prediction"][:, :10, :], context, atol=1e-7)
    assert pred_mse < base_mse * 0.25
    assert result["metadata"]["adapter_type"] == "numpy_pi_deeponet_correction_head"
    assert result["metadata"]["train_mode"] == "ridge_branch_trunk_correction"
