import numpy as np


def test_scalar_source_gate_learns_observable_threshold_and_improves() -> None:
    from pde_agent.core.manual_operator_tuning import score_prediction_arrays
    from pde_agent.core.task1_scalar_source_gate import fit_task1_scalar_source_gate

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    context = np.empty((6, 10, 256), dtype=np.float32)
    for sample in range(context.shape[0]):
        velocity_scale = np.float32(0.001 if sample < 3 else 0.008)
        base_wave = np.sin(x + np.float32(sample) * 0.01)
        for step in range(10):
            context[sample, step, :] = base_wave + np.float32(step) * velocity_scale

    target = np.zeros((6, 200, 256), dtype=np.float32)
    base = np.zeros((6, 200, 256), dtype=np.float32)
    source = np.zeros((6, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    base[:, :10, :] = context
    source[:, :10, :] = context
    target[:, 10:, :] = context[:, 9:10, :]
    base[:, 10:, :] = context[:, 9:10, :] + 0.3
    source[:, 10:, :] = context[:, 9:10, :] + 0.6
    high_velocity = np.arange(3, 6)
    target[high_velocity, 10:57, :] = context[high_velocity, 9:10, :] + 0.6

    base_score = score_prediction_arrays(prediction=base, target=target)
    result = fit_task1_scalar_source_gate(
        context=context,
        base_prediction=base,
        source_predictions=[("high_velocity_source", source)],
        target=target,
        max_steps=1,
        windows=[(10, 57)],
        feature_names=["context_velocity_energy"],
        quantiles=[0.5],
        directions=["gt"],
        blend_weights=[1.0],
        min_selected=1,
    )

    gated_score = score_prediction_arrays(prediction=result["prediction"], target=target)
    assert gated_score["segment_score"] > base_score["segment_score"]
    assert np.allclose(result["prediction"][:, :10, :], context)
    assert result["selected_specs"][0]["source_label"] == "high_velocity_source"
    assert result["selected_specs"][0]["selected_count"] == 3


def test_scalar_source_gate_keeps_base_when_no_spec_improves() -> None:
    from pde_agent.core.task1_scalar_source_gate import fit_task1_scalar_source_gate

    context = np.ones((2, 10, 256), dtype=np.float32)
    base = np.ones((2, 200, 256), dtype=np.float32)
    source = np.ones((2, 200, 256), dtype=np.float32) * 2.0
    target = np.ones((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    source[:, :10, :] = context

    result = fit_task1_scalar_source_gate(
        context=context,
        base_prediction=base,
        source_predictions=[("bad_source", source)],
        target=target,
        max_steps=2,
        windows=[(10, 57)],
        feature_names=["source_base_disagreement"],
        quantiles=[0.5],
        directions=["gt"],
        blend_weights=[1.0],
        min_selected=1,
    )

    assert np.allclose(result["prediction"], base)
    assert result["selected_specs"] == []
