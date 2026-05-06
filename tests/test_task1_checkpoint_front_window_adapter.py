import numpy as np
import pytest


def test_checkpoint_front_window_adapter_builds_direct_blend_and_residual_sources() -> None:
    from pde_agent.core.task1_checkpoint_front_window_adapter import (
        build_checkpoint_front_window_adapter_sources,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 20:35, :] = 1.0
    checkpoint = np.array(base, copy=True)
    checkpoint[:, 20:35, :] = 3.0

    sources = build_checkpoint_front_window_adapter_sources(
        context=context,
        base_prediction=base,
        checkpoint_prediction=checkpoint,
        checkpoint_source="official_fno_checkpoint",
        windows=[(20, 35)],
        adapter_kinds=["direct", "blend", "residual"],
        blend_weights=[0.25],
        residual_scales=[0.5],
    )

    by_kind = {source["metadata"]["adapter_kind"]: source for source in sources}
    direct = by_kind["direct"]["prediction"]
    blend = by_kind["blend"]["prediction"]
    residual = by_kind["residual"]["prediction"]

    assert np.allclose(direct[:, :10, :], context)
    assert np.all(direct[:, 10:20, :] == 0.0)
    assert np.all(direct[:, 35:, :] == 0.0)
    assert np.all(direct[:, 20:35, :] == 3.0)
    assert np.all(blend[:, 20:35, :] == 1.5)
    assert np.all(residual[:, 20:35, :] == 2.0)
    assert all(
        source["metadata"]["uses_target_as_reference"] is False for source in sources
    )
    assert {source["metadata"]["window"] for source in sources} == {"20:35"}


def test_checkpoint_front_window_adapter_rejects_shape_mismatch_and_nonfinite() -> None:
    from pde_agent.core.task1_checkpoint_front_window_adapter import (
        Task1CheckpointFrontWindowAdapterError,
        build_checkpoint_front_window_adapter_sources,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    base = np.zeros((1, 200, 256), dtype=np.float32)
    checkpoint = np.zeros((1, 199, 256), dtype=np.float32)

    with pytest.raises(Task1CheckpointFrontWindowAdapterError, match="shape"):
        build_checkpoint_front_window_adapter_sources(
            context=context,
            base_prediction=base,
            checkpoint_prediction=checkpoint,
            checkpoint_source="checkpoint",
        )

    checkpoint = np.zeros((1, 200, 256), dtype=np.float32)
    checkpoint[0, 20, 0] = np.nan
    with pytest.raises(Task1CheckpointFrontWindowAdapterError, match="finite"):
        build_checkpoint_front_window_adapter_sources(
            context=context,
            base_prediction=base,
            checkpoint_prediction=checkpoint,
            checkpoint_source="checkpoint",
        )


def test_checkpoint_front_window_adapter_rejects_invalid_window() -> None:
    from pde_agent.core.task1_checkpoint_front_window_adapter import (
        Task1CheckpointFrontWindowAdapterError,
        build_checkpoint_front_window_adapter_sources,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    base = np.zeros((1, 200, 256), dtype=np.float32)
    checkpoint = np.zeros((1, 200, 256), dtype=np.float32)

    with pytest.raises(Task1CheckpointFrontWindowAdapterError, match="window"):
        build_checkpoint_front_window_adapter_sources(
            context=context,
            base_prediction=base,
            checkpoint_prediction=checkpoint,
            checkpoint_source="checkpoint",
            windows=[(5, 20)],
        )
