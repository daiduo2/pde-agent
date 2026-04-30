"""Task-1 trajectory retrieval prior for Burgers rollouts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


STATE_HINT = "TASK1_RETRIEVAL_PRIOR_PREDICTION_READY"
FORECAST_STEPS = PREDICTION_STEPS - CONTEXT_STEPS


class Task1RetrievalPriorError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievalPriorPrediction:
    prediction: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray


def run_task1_retrieval_prior(
    *,
    reference_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    dataset_key: str = "tensor",
    k: int = 20,
    mode: str = "affine",
    leave_one_out: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    report_path = output_root / "candidate_execution_report.json"
    prediction_path = output_root / "candidate_prediction.npz"
    if (report_path.exists() or prediction_path.exists()) and not force:
        raise Task1RetrievalPriorError(
            f"retrieval prior artifacts already exist under {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    reference = _load_reference(Path(reference_path), dataset_key=dataset_key)
    context = _load_context(Path(test_path), dataset_key=dataset_key)

    start = time.perf_counter()
    result = predict_retrieval_prior(
        reference=reference,
        context=context,
        k=k,
        mode=mode,
        leave_one_out=leave_one_out,
    )
    inference_time = time.perf_counter() - start
    metadata = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "model_type": "retrieval_prior",
        "mode": mode,
        "k": int(k),
        "leave_one_out": bool(leave_one_out),
        "reference_path": str(reference_path),
        "test_path": str(test_path),
        "reference_count": int(reference.shape[0]),
        "sample_count": int(context.shape[0]),
        "context_steps": CONTEXT_STEPS,
        "forecast_steps": FORECAST_STEPS,
        "space_points": SPACE_POINTS,
    }
    with prediction_path.open("wb") as handle:
        np.savez(
            handle,
            prediction=result.prediction.astype(np.float32),
            neighbor_indices=result.neighbor_indices.astype(np.int64),
            neighbor_distances=result.neighbor_distances.astype(np.float64),
            metadata_json=json.dumps(metadata, allow_nan=False, sort_keys=True),
        )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "reference_path": str(reference_path),
        "test_path": str(test_path),
        "candidate": {
            "experiment_id": f"task1_retrieval_prior_k{int(k)}_{mode}",
            "model_family": "RetrievalPrior",
            "task_id": 1,
            "parameters": {
                "k": int(k),
                "mode": mode,
                "leave_one_out": bool(leave_one_out),
            },
        },
        "inputs": {
            "reference_count": int(reference.shape[0]),
            "sample_count": int(context.shape[0]),
            "context_steps": CONTEXT_STEPS,
            "forecast_steps": FORECAST_STEPS,
            "space_points": SPACE_POINTS,
        },
        "artifacts": {
            "candidate_prediction_path": str(prediction_path),
            "candidate_execution_report_path": str(report_path),
        },
        "train": {
            "mode": "reference_library_lookup",
            "elapsed_seconds": 0.0,
        },
        "inference": {
            "elapsed_seconds": float(inference_time),
            "backend": "numpy_knn_affine_retrieval",
            "checkpoint_format": "none",
        },
        "next_command": "materialize-competition-candidate",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def predict_retrieval_prior(
    *,
    reference: np.ndarray,
    context: np.ndarray,
    k: int,
    mode: str,
    leave_one_out: bool,
) -> RetrievalPriorPrediction:
    reference = _as_reference(reference)
    context = _as_context(context)
    mode = _mode(mode)
    k = _k(k=k, reference_count=reference.shape[0], leave_one_out=leave_one_out)

    reference_context = reference[:, :CONTEXT_STEPS, :]
    distances = _context_distances(query=context, reference=reference_context)
    if leave_one_out:
        if context.shape[0] != reference.shape[0]:
            raise Task1RetrievalPriorError(
                "leave_one_out requires context and reference sample counts to match"
            )
        np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    order = np.argsort(neighbor_distances, axis=1)
    neighbor_indices = np.take_along_axis(neighbor_indices, order, axis=1)
    neighbor_distances = np.take_along_axis(neighbor_distances, order, axis=1)
    weights = 1.0 / (neighbor_distances + 1e-9)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    neighbors = reference[neighbor_indices]
    prediction = _combine_neighbors(
        context=context,
        neighbors=neighbors,
        weights=weights,
        mode=mode,
    )
    if not np.isfinite(prediction).all():
        raise Task1RetrievalPriorError("retrieval prediction contains NaN or Inf")
    return RetrievalPriorPrediction(
        prediction=np.asarray(prediction, dtype=np.float32),
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )


def _combine_neighbors(
    *,
    context: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "raw":
        future = neighbors[:, :, CONTEXT_STEPS:, :]
    elif mode == "delta":
        future = context[:, None, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :] + (
            neighbors[:, :, CONTEXT_STEPS:, :]
            - neighbors[:, :, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :]
        )
    else:
        neighbor_context = neighbors[:, :, :CONTEXT_STEPS, :]
        ref_mean = neighbor_context.mean(axis=(2, 3), keepdims=True)
        ref_std = neighbor_context.std(axis=(2, 3), keepdims=True) + 1e-6
        query_mean = context.mean(axis=(1, 2), keepdims=True)[:, None, :, :]
        query_std = context.std(axis=(1, 2), keepdims=True)[:, None, :, :] + 1e-6
        future = (
            (neighbors[:, :, CONTEXT_STEPS:, :] - ref_mean) / ref_std
        ) * query_std + query_mean
    return np.sum(future * weights[:, :, None, None], axis=1)


def _context_distances(*, query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query_flat = query.reshape(query.shape[0], -1).astype(np.float64)
    reference_flat = reference.reshape(reference.shape[0], -1).astype(np.float64)
    query_flat = _zscore_rows(query_flat)
    reference_flat = _zscore_rows(reference_flat)
    return np.mean(
        (query_flat[:, None, :] - reference_flat[None, :, :]) ** 2,
        axis=2,
    )


def _zscore_rows(value: np.ndarray) -> np.ndarray:
    return (value - value.mean(axis=1, keepdims=True)) / (
        value.std(axis=1, keepdims=True) + 1e-6
    )


def _load_reference(path: Path, *, dataset_key: str) -> np.ndarray:
    data = _load_hdf5(path=path, dataset_key=dataset_key)
    if data.shape[1] < PREDICTION_STEPS:
        raise Task1RetrievalPriorError(
            f"reference must contain at least {PREDICTION_STEPS} time steps"
        )
    return np.asarray(data[:, :PREDICTION_STEPS, :], dtype=np.float32)


def _load_context(path: Path, *, dataset_key: str) -> np.ndarray:
    data = _load_hdf5(path=path, dataset_key=dataset_key)
    if data.shape[1] < CONTEXT_STEPS:
        raise Task1RetrievalPriorError(
            f"test data must contain at least {CONTEXT_STEPS} context steps"
        )
    return np.asarray(data[:, :CONTEXT_STEPS, :], dtype=np.float32)


def _load_hdf5(*, path: Path, dataset_key: str) -> np.ndarray:
    if not path.is_file():
        raise Task1RetrievalPriorError(f"missing HDF5 path: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise Task1RetrievalPriorError(f"dataset {dataset_key!r} not found")
        data = np.asarray(handle[dataset_key][:], dtype=np.float32)
    if data.ndim != 3:
        raise Task1RetrievalPriorError(f"expected rank-3 tensor, got {data.shape}")
    if data.shape[1] == 1000:
        data = data[:, ::5, :]
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    if data.shape[2] != SPACE_POINTS:
        raise Task1RetrievalPriorError(
            f"expected spatial width 256 or 1024, got {data.shape[2]}"
        )
    if not np.isfinite(data).all():
        raise Task1RetrievalPriorError("HDF5 tensor contains NaN or Inf")
    return data


def _as_reference(reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    if reference.ndim != 3 or reference.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise Task1RetrievalPriorError(
            f"expected reference shape [N, 200, 256], got {reference.shape}"
        )
    if reference.shape[0] < 2:
        raise Task1RetrievalPriorError("reference library must contain at least 2 samples")
    if not np.isfinite(reference).all():
        raise Task1RetrievalPriorError("reference contains NaN or Inf")
    return reference


def _as_context(context: np.ndarray) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3 or context.shape[1:] != (CONTEXT_STEPS, SPACE_POINTS):
        raise Task1RetrievalPriorError(
            f"expected context shape [N, 10, 256], got {context.shape}"
        )
    if context.shape[0] < 1:
        raise Task1RetrievalPriorError("context batch is empty")
    if not np.isfinite(context).all():
        raise Task1RetrievalPriorError("context contains NaN or Inf")
    return context


def _k(*, k: int, reference_count: int, leave_one_out: bool) -> int:
    if k < 1:
        raise Task1RetrievalPriorError("k must be positive")
    max_k = reference_count - 1 if leave_one_out else reference_count
    if k > max_k:
        raise Task1RetrievalPriorError(f"k must be <= {max_k}")
    return int(k)


def _mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in {"raw", "delta", "affine"}:
        raise Task1RetrievalPriorError("mode must be raw, delta, or affine")
    return mode
