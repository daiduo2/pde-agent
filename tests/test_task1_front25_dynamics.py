import numpy as np


def test_front25_dynamics_preserves_context_and_only_replaces_short_window() -> None:
    from pde_agent.core.task1_front25_dynamics import fit_task1_front25_dynamics

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.empty((2, 10, 256), dtype=np.float32)
    reference = np.empty((8, 200, 256), dtype=np.float32)
    for sample in range(context.shape[0]):
        phase = np.float32(0.04 * sample)
        velocity = np.float32(0.003 + 0.001 * sample) * np.cos(x + phase)
        for step in range(10):
            context[sample, step, :] = np.sin(x + phase) + step * velocity
    for sample in range(reference.shape[0]):
        phase = np.float32(0.04 * sample)
        velocity = np.float32(0.003 + 0.001 * sample) * np.cos(x + phase)
        for step in range(200):
            reference[sample, step, :] = np.sin(x + phase) + step * velocity
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 35:, :] = 7.0

    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base,
        reference=reference,
        k=4,
        horizon_steps=25,
        ridge_alpha=1e-8,
        correction_weight=1.0,
        dynamics_mode="velocity_feature_ridge",
        feature_mode="last_velocity_gradient",
    )

    prediction = result["prediction"]
    assert np.allclose(prediction[:, :10, :], context)
    assert np.all(prediction[:, 35:, :] == 7.0)
    assert not np.allclose(prediction[:, 10:35, :], base[:, 10:35, :])
    assert result["metadata"]["horizon_steps"] == 25
    assert result["metadata"]["dynamics_mode"] == "velocity_feature_ridge"


def test_front25_dynamics_context_pod_ridge_learns_reference_short_residual() -> None:
    from pde_agent.core.task1_front25_dynamics import fit_task1_front25_dynamics

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    reference = np.empty((12, 200, 256), dtype=np.float32)
    for sample in range(reference.shape[0]):
        phase = np.float32(0.03 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (np.float32(0.002 + 0.0005 * sample) * np.cos(x + phase)).astype(
            np.float32
        )
        for step in range(200):
            reference[sample, step, :] = base + step * velocity
    context = reference[2:5, :10, :].copy()
    base_prediction = np.zeros((3, 200, 256), dtype=np.float32)
    base_prediction[:, :10, :] = context
    base_prediction[:, 35:, :] = -3.0

    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        dynamics_mode="context_pod_ridge",
        feature_mode="last_velocity_gradient",
        feature_count=12,
        ridge_alpha=1e-8,
        correction_weight=1.0,
    )

    expected = reference[2:5, 10:35, :]
    prediction = result["prediction"]
    assert np.mean((prediction[:, 10:35, :] - expected) ** 2) < 1e-5
    assert np.all(prediction[:, 35:, :] == -3.0)
    assert result["metadata"]["dynamics_mode"] == "context_pod_ridge"
    assert result["metadata"]["feature_count"] == 12


def test_front25_dynamics_can_use_time_reduced_reference_grid() -> None:
    from pde_agent.core.task1_front25_dynamics import fit_task1_front25_dynamics

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    raw_reference = np.empty((6, 200, 256), dtype=np.float32)
    reduced_reference = np.empty((6, 40, 256), dtype=np.float32)
    for sample in range(raw_reference.shape[0]):
        phase = np.float32(0.05 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (np.float32(0.004 + 0.0005 * sample) * np.cos(x + phase)).astype(
            np.float32
        )
        for raw_step in range(200):
            raw_reference[sample, raw_step, :] = base + (raw_step / 5.0) * velocity
        reduced_reference[sample] = raw_reference[sample, ::5, :]

    context = reduced_reference[2:4, :10, :].copy()
    base_prediction = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction[:, :10, :] = context
    base_prediction[:, 35:, :] = 11.0

    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base_prediction,
        reference=raw_reference,
        k=1,
        horizon_steps=25,
        dynamics_mode="delta_mean",
        feature_mode="last_velocity_gradient",
        reference_time_stride=5,
    )

    prediction = result["prediction"]
    assert np.mean((prediction[:, 10:35, :] - reduced_reference[2:4, 10:35, :]) ** 2) < 1e-7
    assert np.all(prediction[:, 35:, :] == 11.0)
    assert result["metadata"]["reference_time_stride"] == 5


def test_front25_dynamics_accepts_pre_reduced_reference_grid() -> None:
    from pde_agent.core.task1_front25_dynamics import fit_task1_front25_dynamics

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    reference = np.empty((6, 41, 256), dtype=np.float32)
    for sample in range(reference.shape[0]):
        phase = np.float32(0.05 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (np.float32(0.004 + 0.0005 * sample) * np.cos(x + phase)).astype(
            np.float32
        )
        for step in range(reference.shape[1]):
            reference[sample, step, :] = base + np.float32(step) * velocity

    context = reference[2:4, :10, :].copy()
    base_prediction = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction[:, :10, :] = context
    base_prediction[:, 35:, :] = 11.0

    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        k=1,
        horizon_steps=25,
        dynamics_mode="delta_mean",
        feature_mode="last_velocity_gradient",
    )

    prediction = result["prediction"]

    assert np.mean((prediction[:, 10:35, :] - reference[2:4, 10:35, :]) ** 2) < 1e-7
    assert np.all(prediction[:, 35:, :] == 11.0)
    assert result["metadata"]["reference_time_steps"] == 41


def test_front25_dynamics_pseudospectral_mode_only_replaces_short_window() -> None:
    from pde_agent.core.task1_front25_dynamics import fit_task1_front25_dynamics

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.empty((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[0, step, :] = np.sin(x + np.float32(0.02 * step))
    reference = np.zeros((2, 200, 256), dtype=np.float32)
    reference[:, :10, :] = context
    base_prediction = np.zeros((1, 200, 256), dtype=np.float32)
    base_prediction[:, :10, :] = context
    base_prediction[:, 35:, :] = 9.0

    result = fit_task1_front25_dynamics(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        dynamics_mode="pseudospectral",
        horizon_steps=25,
        correction_weight=1.0,
        pseudospectral_dt=0.03,
        pseudospectral_substeps=2,
        pseudospectral_filter_strength=0.01,
    )

    prediction = result["prediction"]
    assert np.allclose(prediction[:, :10, :], context)
    assert np.all(prediction[:, 35:, :] == 9.0)
    assert not np.allclose(prediction[:, 10:35, :], base_prediction[:, 10:35, :])
    assert result["metadata"]["dynamics_mode"] == "pseudospectral"
    assert result["metadata"]["pseudospectral_dt"] == 0.03
    assert result["metadata"]["pseudospectral_substeps"] == 2
