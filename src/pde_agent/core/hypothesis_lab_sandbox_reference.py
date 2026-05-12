"""Reference loading and leakage audit helpers for Hypothesis Lab sandboxes."""

from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.hypothesis_lab_utils import HypothesisLabError


def _load_sandbox_reference(
    *,
    reference_path: str | Path | None,
    target_path: str | Path,
    target: np.ndarray,
    sample_limit: int | None,
) -> np.ndarray:
    if sample_limit is not None and sample_limit < 2:
        raise HypothesisLabError("reference_sample_limit must be at least 2")
    if reference_path is None:
        reference = np.asarray(target, dtype=np.float32)
    else:
        reference = _load_reference_tensor(
            path=Path(reference_path),
            sample_limit=sample_limit,
        )
    if reference.ndim != 3:
        raise HypothesisLabError(f"expected rank-3 reference tensor, got {reference.shape}")
    if reference.shape[1] <= 10:
        raise HypothesisLabError(
            f"reference must contain more than 10 time steps, got {reference.shape}"
        )
    reference = np.asarray(reference[:, :200, :], dtype=np.float32)
    if reference.shape[2] != 256:
        raise HypothesisLabError(
            f"reference spatial width must be 256 after reduction, got {reference.shape[2]}"
        )
    if sample_limit is not None:
        reference = reference[: int(sample_limit)]
    if reference.shape[0] < 1:
        raise HypothesisLabError("reference library must contain at least 1 sample")
    if not np.isfinite(reference).all():
        raise HypothesisLabError("reference contains NaN or Inf")
    _ = target_path
    return reference


def _reference_audit(
    *,
    reference_path: str | Path | None,
    target_path: str | Path,
    reference: np.ndarray,
    reference_sample_limit: int | None,
) -> dict[str, Any]:
    target_file = Path(target_path)
    reference_file = Path(reference_path) if reference_path is not None else target_file
    uses_target_as_reference = reference_path is None or _same_existing_path(
        reference_file,
        target_file,
    )
    return {
        "reference_path": str(reference_file),
        "target_path": str(target_file),
        "uses_target_as_reference": uses_target_as_reference,
        "reference_count": int(reference.shape[0]),
        "reference_time_steps": int(reference.shape[1]),
        "reference_sample_limit": int(reference_sample_limit)
        if reference_sample_limit is not None
        else None,
        "submission_validity": "local_validation_only"
        if uses_target_as_reference
        else "submission_plausible_reference",
        "risk": "target_reference_leakage"
        if uses_target_as_reference
        else "external_reference_distribution_shift",
    }


def _load_reference_tensor(*, path: Path, sample_limit: int | None) -> np.ndarray:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if "tensor" not in handle:
            raise HypothesisLabError(f"dataset 'tensor' not found in reference: {path}")
        dataset = handle["tensor"]
        if dataset.ndim != 3:
            raise HypothesisLabError(
                f"expected rank-3 reference tensor, got {dataset.shape}"
            )
        stop = int(sample_limit) if sample_limit is not None else dataset.shape[0]
        data = np.asarray(dataset[:stop], dtype=np.float32)
    if data.ndim != 3:
        raise HypothesisLabError(f"expected rank-3 reference tensor, got {data.shape}")
    if data.shape[1] == 1000:
        data = data[:, ::5, :]
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    return data


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)
