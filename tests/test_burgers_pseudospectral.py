import numpy as np


def test_pseudospectral_burgers_rollout_preserves_constant_context() -> None:
    from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout

    context = np.full((3, 10, 256), 0.25, dtype=np.float32)

    prediction = pseudospectral_burgers_rollout(
        context=context,
        viscosity=0.001,
        dt=0.05,
        substeps=4,
        filter_strength=0.02,
    )

    assert prediction.shape == (3, 200, 256)
    assert np.allclose(prediction[:, :10, :], context)
    assert np.allclose(prediction[:, 10:, :], 0.25, atol=1e-5)


def test_pseudospectral_burgers_rollout_moves_nonconstant_context() -> None:
    from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout

    x = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    context = np.zeros((1, 10, 256), dtype=np.float32)
    for index in range(10):
        context[0, index, :] = 0.2 + 0.05 * np.sin(2.0 * np.pi * (x - 0.01 * index))

    prediction = pseudospectral_burgers_rollout(
        context=context,
        viscosity=0.001,
        dt=0.03,
        substeps=4,
        filter_strength=0.02,
    )

    assert prediction.shape == (1, 200, 256)
    assert np.all(np.isfinite(prediction))
    assert np.max(np.abs(prediction[:, 11, :] - prediction[:, 10, :])) > 1e-5


def test_pseudospectral_burgers_rollout_can_stop_at_short_horizon() -> None:
    from pde_agent.core.burgers_pseudospectral import pseudospectral_burgers_rollout

    x = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    context = np.zeros((1, 10, 256), dtype=np.float32)
    for index in range(10):
        context[0, index, :] = 0.2 + 0.05 * np.sin(2.0 * np.pi * (x - 0.01 * index))

    prediction = pseudospectral_burgers_rollout(
        context=context,
        prediction_steps=35,
    )

    assert prediction.shape == (1, 35, 256)
    assert np.allclose(prediction[:, :10, :], context)
