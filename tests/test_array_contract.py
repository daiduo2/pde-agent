import numpy as np
import pytest

from pde_agent.core.array_contract import (
    ArrayContractError,
    BurgersArrayBundle,
    validate_burgers_bundle,
)


def test_validate_burgers_bundle_accepts_matching_arrays() -> None:
    context = np.ones((2, 3, 5), dtype=np.float32)
    prediction = np.ones((2, 4, 5), dtype=np.int32)
    target = np.zeros((2, 4, 5), dtype=np.float64)
    full_reference = np.ones((2, 7, 5), dtype=np.float32)
    bundle = BurgersArrayBundle(
        context=context,
        prediction=prediction,
        target=target,
        full_reference=full_reference,
        metadata={
            "problem_id": "burgers_1d",
            "source": "fixture",
            "source_mode": "npz_fixture",
            "sample_count": 999,
        },
    )

    validated = validate_burgers_bundle(bundle)

    assert validated.context.dtype == np.float64
    assert validated.prediction.dtype == np.float64
    assert validated.target is target
    assert validated.full_reference is not None
    assert validated.full_reference.dtype == np.float64
    assert validated.metadata == {
        "problem_id": "burgers_1d",
        "source": "fixture",
        "source_mode": "npz_fixture",
        "sample_count": 2,
        "context_time": 3,
        "forecast_time": 4,
        "space_points": 5,
    }
    assert bundle.metadata == {
        "problem_id": "burgers_1d",
        "source": "fixture",
        "source_mode": "npz_fixture",
        "sample_count": 999,
    }


def test_validate_burgers_bundle_rejects_prediction_target_shape_mismatch() -> None:
    bundle = BurgersArrayBundle(
        context=np.ones((2, 3, 5)),
        prediction=np.ones((2, 4, 5)),
        target=np.ones((2, 5, 5)),
        full_reference=None,
        metadata={},
    )

    with pytest.raises(
        ArrayContractError, match="prediction shape must match target shape"
    ):
        validate_burgers_bundle(bundle)


def test_validate_burgers_bundle_rejects_non_3d_arrays() -> None:
    bundle = BurgersArrayBundle(
        context=np.ones((2, 3)),
        prediction=np.ones((2, 4, 5)),
        target=np.ones((2, 4, 5)),
        full_reference=None,
        metadata={},
    )

    with pytest.raises(ArrayContractError, match="context must be rank 3"):
        validate_burgers_bundle(bundle)


def test_validate_burgers_bundle_rejects_missing_required_metadata() -> None:
    bundle = BurgersArrayBundle(
        context=np.ones((2, 3, 5)),
        prediction=np.ones((2, 4, 5)),
        target=np.ones((2, 4, 5)),
        full_reference=None,
        metadata={"problem_id": "burgers_1d"},
    )

    with pytest.raises(
        ArrayContractError, match="metadata.source_mode is required"
    ):
        validate_burgers_bundle(bundle)


def test_validate_burgers_bundle_rejects_complex_dtype() -> None:
    bundle = BurgersArrayBundle(
        context=np.ones((2, 3, 5), dtype=np.complex128),
        prediction=np.ones((2, 4, 5)),
        target=np.ones((2, 4, 5)),
        full_reference=None,
        metadata={"problem_id": "burgers_1d", "source_mode": "npz_fixture"},
    )

    with pytest.raises(
        ArrayContractError,
        match="context dtype cannot be converted to float safely",
    ):
        validate_burgers_bundle(bundle)
