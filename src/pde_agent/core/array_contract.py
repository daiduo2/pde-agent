
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


SourceMode = Literal["synthetic", "npz_fixture", "pdebench_hdf5"]


class ArrayContractError(ValueError):
    pass


@dataclass(frozen=True)
class BurgersArrayBundle:
    context: NDArray[np.floating]
    prediction: NDArray[np.floating]
    target: NDArray[np.floating]
    full_reference: NDArray[np.floating] | None
    metadata: dict[str, object]


def as_float_array(name: str, value: object) -> NDArray[np.floating]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ArrayContractError(f"{name} dtype cannot be converted to float safely")
    return array.astype(np.float64, copy=False)


def validate_burgers_bundle(bundle: BurgersArrayBundle) -> BurgersArrayBundle:
    context = as_float_array("context", bundle.context)
    prediction = as_float_array("prediction", bundle.prediction)
    target = as_float_array("target", bundle.target)
    full_reference = (
        None
        if bundle.full_reference is None
        else as_float_array("full_reference", bundle.full_reference)
    )

    _require_rank_3("context", context)
    _require_rank_3("prediction", prediction)
    _require_rank_3("target", target)
    if full_reference is not None:
        _require_rank_3("full_reference", full_reference)

    if prediction.shape != target.shape:
        raise ArrayContractError("prediction shape must match target shape")
    if context.shape[0] != prediction.shape[0]:
        raise ArrayContractError("context sample count must match prediction")
    if context.shape[2] != prediction.shape[2]:
        raise ArrayContractError("context space dimension must match prediction")

    metadata = dict(bundle.metadata)
    _require_metadata_value(metadata, "problem_id")
    _require_metadata_value(metadata, "source_mode")
    metadata.update(
        {
            "sample_count": prediction.shape[0],
            "context_time": context.shape[1],
            "forecast_time": prediction.shape[1],
            "space_points": prediction.shape[2],
        }
    )
    return BurgersArrayBundle(
        context=context,
        prediction=prediction,
        target=target,
        full_reference=full_reference,
        metadata=metadata,
    )


def _require_rank_3(name: str, array: NDArray[np.floating]) -> None:
    if array.ndim != 3:
        raise ArrayContractError(f"{name} must be rank 3")


def _require_metadata_value(metadata: dict[str, object], key: str) -> None:
    if not metadata.get(key):
        raise ArrayContractError(f"metadata.{key} is required")
