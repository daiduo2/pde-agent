"""Generate lightweight postprocess submission candidates for Hermes optimization."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
    build_competition_submission,
)


STATE_HINT = "HERMES_POSTPROCESS_CANDIDATE_READY"
SUPPORTED_STRATEGIES = {
    "context_anchor_blend",
    "spectral_smooth",
}


class HermesPostprocessCandidateError(RuntimeError):
    pass


def generate_hermes_postprocess_candidate(
    *,
    base_submission_dir: str | Path,
    output_dir: str | Path,
    task_id: int,
    test_path: str | Path,
    strategy: str,
    blend_weight: float = 0.25,
    submission_id: str = "team_hermes",
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise HermesPostprocessCandidateError("task_id must be 1 or 2")
    if strategy not in SUPPORTED_STRATEGIES:
        raise HermesPostprocessCandidateError(f"unsupported strategy: {strategy}")
    if not 0.0 <= blend_weight <= 1.0:
        raise HermesPostprocessCandidateError("blend_weight must be in [0, 1]")

    root = Path(output_dir)
    report_path = root / "hermes_postprocess_candidate.json"
    if report_path.exists() and not force:
        raise HermesPostprocessCandidateError(
            f"postprocess candidate already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)

    base_timing = _read_base_timing(
        base_submission_dir=Path(base_submission_dir),
        task_id=task_id,
    )
    postprocess_start = time.perf_counter()
    base_prediction_path = Path(base_submission_dir) / f"task{task_id}_pred.hdf5"
    base_prediction = _load_prediction(
        path=base_prediction_path,
        dataset_key=prediction_dataset_key,
    )
    transformed = _transform_prediction(
        prediction=base_prediction,
        strategy=strategy,
        blend_weight=blend_weight,
    )
    candidate_prediction_path = root / "candidate_prediction.npz"
    np.savez(candidate_prediction_path, prediction=transformed[:, CONTEXT_STEPS:, :])
    postprocess_elapsed = time.perf_counter() - postprocess_start
    submission = build_competition_submission(
        task_id=task_id,
        test_path=Path(test_path),
        output_dir=root / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=candidate_prediction_path,
        source_prediction_key="prediction",
        train_time=base_timing["raw_train_time"],
        inference_time=base_timing["raw_inference_time"] + postprocess_elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=True,
    )
    candidate_id = f"postprocess_{strategy}_w{blend_weight:g}"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "candidate_report_path": str(report_path),
        "candidate_id": candidate_id,
        "strategy": strategy,
        "blend_weight": float(blend_weight),
        "task_id": task_id,
        "base_submission_dir": str(base_submission_dir),
        "timing": {
            "base_train_time": base_timing["train_time"],
            "base_inference_time": base_timing["inference_time"],
            "base_raw_train_time": base_timing["raw_train_time"],
            "base_raw_inference_time": base_timing["raw_inference_time"],
            "postprocess_elapsed_seconds": postprocess_elapsed,
            "reported_train_time": submission["metrics"]["train_time"],
            "reported_inference_time": submission["metrics"]["inference_time"],
            "timing_profile": submission["metrics"].get("timing_profile"),
            "timing_factor": submission["metrics"].get("timing_factor"),
            "timing_valid_for_submission": submission["metrics"].get(
                "timing_valid_for_submission"
            ),
        },
        "submission_dir": submission["artifacts"]["submission_dir"],
        "artifacts": {
            "candidate_report_path": str(report_path),
            "candidate_prediction_path": str(candidate_prediction_path),
            "submission_dir": submission["artifacts"]["submission_dir"],
            "submission_zip_path": submission["artifacts"]["submission_zip_path"],
            "prediction_path": submission["artifacts"]["prediction_path"],
        },
        "submission": submission,
        "next_command": "judge-hermes-submission",
    }
    _write_json(report_path, report)
    return report


def _read_base_timing(*, base_submission_dir: Path, task_id: int) -> dict[str, float]:
    path = base_submission_dir / f"task{task_id}_time.csv"
    if not path.exists():
        raise HermesPostprocessCandidateError(f"base time csv not found: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 1:
        raise HermesPostprocessCandidateError(f"invalid base time csv rows: {path}")
    try:
        train_time = float(rows[0]["train_time"])
        inference_time = float(rows[0]["inference_time"])
    except (KeyError, ValueError) as exc:
        raise HermesPostprocessCandidateError(
            f"invalid base time csv values: {path}"
        ) from exc
    if train_time < 0.0 or inference_time < 0.0:
        raise HermesPostprocessCandidateError(
            f"base time csv contains negative values: {path}"
        )
    log_values = _read_base_raw_timing_from_log(
        base_submission_dir=base_submission_dir,
        task_id=task_id,
    )
    raw_train_time = log_values.get("raw_train_time", train_time)
    raw_inference_time = log_values.get("raw_inference_time", inference_time)
    if raw_train_time < 0.0 or raw_inference_time < 0.0:
        raise HermesPostprocessCandidateError(
            f"base log contains negative raw timing values: {base_submission_dir}"
        )
    return {
        "train_time": train_time,
        "inference_time": inference_time,
        "raw_train_time": raw_train_time,
        "raw_inference_time": raw_inference_time,
    }


def _read_base_raw_timing_from_log(
    *,
    base_submission_dir: Path,
    task_id: int,
) -> dict[str, float]:
    path = base_submission_dir / f"task{task_id}_logs.log"
    if not path.exists():
        return {}
    values: dict[str, float] = {}
    field_map = {
        "raw_train_time_seconds": "raw_train_time",
        "raw_inference_time_seconds": "raw_inference_time",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        for label, key in field_map.items():
            prefix = f"- {label}:"
            if not text.startswith(prefix):
                continue
            try:
                values[key] = float(text[len(prefix) :].strip())
            except ValueError as exc:
                raise HermesPostprocessCandidateError(
                    f"invalid raw timing value in {path}: {text}"
                ) from exc
    return values


def _load_prediction(*, path: Path, dataset_key: str) -> np.ndarray:
    if not path.exists():
        raise HermesPostprocessCandidateError(f"base prediction not found: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise HermesPostprocessCandidateError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(handle[dataset_key][:], dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (PREDICTION_STEPS, SPACE_POINTS):
        raise HermesPostprocessCandidateError(
            f"expected prediction shape [N, 200, 256], got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise HermesPostprocessCandidateError("base prediction contains NaN or Inf")
    return data


def _transform_prediction(
    *,
    prediction: np.ndarray,
    strategy: str,
    blend_weight: float,
) -> np.ndarray:
    transformed = np.array(prediction, copy=True)
    if strategy == "context_anchor_blend":
        anchor = prediction[:, CONTEXT_STEPS - 1 : CONTEXT_STEPS, :]
        transformed[:, CONTEXT_STEPS:, :] = (
            (1.0 - blend_weight) * prediction[:, CONTEXT_STEPS:, :]
            + blend_weight * anchor
        )
        return transformed.astype(np.float32)
    if strategy == "spectral_smooth":
        future = prediction[:, CONTEXT_STEPS:, :]
        smoothed = _spectral_smooth(future=future, strength=blend_weight)
        transformed[:, CONTEXT_STEPS:, :] = smoothed
        return transformed.astype(np.float32)
    raise HermesPostprocessCandidateError(f"unsupported strategy: {strategy}")


def _spectral_smooth(*, future: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.array(future, copy=True)
    spectrum = np.fft.rfft(future, axis=2)
    freqs = np.linspace(0.0, 1.0, spectrum.shape[2], dtype=np.float64)
    damping = np.exp(-float(strength) * 8.0 * freqs**2)
    spectrum = spectrum * damping.reshape(1, 1, -1)
    smoothed = np.fft.irfft(spectrum, n=future.shape[2], axis=2)
    return np.asarray(smoothed, dtype=np.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
