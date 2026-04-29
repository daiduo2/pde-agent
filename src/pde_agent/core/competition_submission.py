"""Build competition-shaped PDE Burgers submission artifacts."""

from __future__ import annotations

import csv
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


STATE_HINT = "COMPETITION_SUBMISSION_READY"
PROBLEM_ID = "PDE_Burgers"
PREDICTION_STEPS = 200
CONTEXT_STEPS = 10
SPACE_POINTS = 256


class CompetitionSubmissionError(RuntimeError):
    pass


def build_competition_submission(
    *,
    task_id: int,
    test_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    method: str = "persistence",
    source_prediction_path: str | Path | None = None,
    source_prediction_key: str = "prediction",
    train_time: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    if task_id not in {1, 2}:
        raise CompetitionSubmissionError("task_id must be 1 or 2")
    if method not in {"persistence", "external_prediction"}:
        raise CompetitionSubmissionError(
            f"unsupported competition submission method: {method}"
        )
    if train_time < 0.0:
        raise CompetitionSubmissionError("train_time must be non-negative")

    output_root = Path(output_dir)
    submission_dir = output_root / "submission"
    code_dir = submission_dir / "code"
    prediction_path = submission_dir / f"task{task_id}_pred.hdf5"
    time_path = submission_dir / f"task{task_id}_time.csv"
    log_path = submission_dir / f"task{task_id}_logs.log"
    submission_json_path = submission_dir / "submission.json"
    submission_zip_path = output_root / "submission.zip"
    train_py_path = code_dir / "train.py"
    model_py_path = code_dir / "model.py"

    targets = [
        prediction_path,
        time_path,
        log_path,
        submission_json_path,
        submission_zip_path,
        train_py_path,
        model_py_path,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise CompetitionSubmissionError(
            "submission artifacts already exist: "
            + ", ".join(str(path) for path in existing)
        )

    context = _load_context(
        path=Path(test_path),
        dataset_key=dataset_key,
    )
    start = time.perf_counter()
    if method == "persistence":
        prediction = _persistence_prediction(context)
    else:
        if source_prediction_path is None:
            raise CompetitionSubmissionError(
                "external_prediction requires source_prediction_path"
            )
        prediction = _external_prediction(
            context=context,
            path=Path(source_prediction_path),
            key=source_prediction_key,
        )
    inference_time = time.perf_counter() - start
    _validate_prediction(prediction=prediction, context=context)

    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    _write_prediction(
        path=prediction_path,
        dataset_key=prediction_dataset_key,
        prediction=prediction,
    )
    _write_time_csv(
        path=time_path,
        train_time=train_time,
        inference_time=inference_time,
    )
    _write_submission_json(path=submission_json_path, submission_id=submission_id)
    _write_log(
        path=log_path,
        task_id=task_id,
        test_path=Path(test_path),
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        sample_count=int(prediction.shape[0]),
        method=method,
        source_prediction_path=Path(source_prediction_path)
        if source_prediction_path is not None
        else None,
        inference_time=inference_time,
        train_time=train_time,
    )
    _write_code_stubs(model_path=model_py_path, train_path=train_py_path)
    _write_submission_zip(
        submission_dir=submission_dir,
        zip_path=submission_zip_path,
    )

    return {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "method": method,
        "artifacts": {
            "submission_dir": str(submission_dir),
            "submission_zip_path": str(submission_zip_path),
            "prediction_path": str(prediction_path),
            "time_csv_path": str(time_path),
            "log_path": str(log_path),
            "submission_json_path": str(submission_json_path),
            "code_dir": str(code_dir),
        },
        "metrics": {
            "task_id": task_id,
            "sample_count": int(prediction.shape[0]),
            "prediction_shape": list(prediction.shape),
            "method": method,
            "train_time": train_time,
            "inference_time": inference_time,
        },
        "next_command": "submit-zip-to-competition",
    }


def _load_context(*, path: Path, dataset_key: str) -> np.ndarray:
    if not path.exists():
        raise CompetitionSubmissionError(f"test HDF5 not found: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise CompetitionSubmissionError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(handle[dataset_key][:])
    if data.ndim != 3:
        raise CompetitionSubmissionError(
            f"expected rank-3 test tensor, got shape {data.shape}"
        )
    if data.shape[1] < CONTEXT_STEPS:
        raise CompetitionSubmissionError(
            f"expected at least {CONTEXT_STEPS} context steps, got {data.shape[1]}"
        )
    if data.shape[2] == SPACE_POINTS:
        context = data[:, :CONTEXT_STEPS, :]
    elif data.shape[2] == 1024:
        context = data[:, :CONTEXT_STEPS, ::4]
    else:
        raise CompetitionSubmissionError(
            f"expected spatial width 256 or 1024, got {data.shape[2]}"
        )
    context = np.asarray(context, dtype=np.float32)
    if not np.isfinite(context).all():
        raise CompetitionSubmissionError("test context contains NaN or Inf values")
    return context


def _persistence_prediction(context: np.ndarray) -> np.ndarray:
    prediction = np.empty(
        (context.shape[0], PREDICTION_STEPS, SPACE_POINTS), dtype=np.float32
    )
    prediction[:, :CONTEXT_STEPS, :] = context
    prediction[:, CONTEXT_STEPS:, :] = context[:, -1:, :]
    return prediction


def _external_prediction(*, context: np.ndarray, path: Path, key: str) -> np.ndarray:
    source = _load_prediction_source(path=path, key=key)
    if source.shape[0] != context.shape[0]:
        raise CompetitionSubmissionError(
            f"prediction sample count {source.shape[0]} != context sample count {context.shape[0]}"
        )
    if source.shape[2] == 1024:
        source = source[:, :, ::4]
    if source.shape[2] != SPACE_POINTS:
        raise CompetitionSubmissionError(
            f"expected prediction spatial width 256 or 1024, got {source.shape[2]}"
        )
    source = np.asarray(source, dtype=np.float32)
    if not np.isfinite(source).all():
        raise CompetitionSubmissionError("source prediction contains NaN or Inf values")
    prediction = np.empty(
        (context.shape[0], PREDICTION_STEPS, SPACE_POINTS), dtype=np.float32
    )
    prediction[:, :CONTEXT_STEPS, :] = context
    if source.shape[1] == PREDICTION_STEPS:
        prediction[:, CONTEXT_STEPS:, :] = source[:, CONTEXT_STEPS:, :]
    elif source.shape[1] == PREDICTION_STEPS - CONTEXT_STEPS:
        prediction[:, CONTEXT_STEPS:, :] = source
    else:
        raise CompetitionSubmissionError(
            "expected source prediction time dimension 190 or 200, "
            f"got {source.shape[1]}"
        )
    return prediction


def _load_prediction_source(*, path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise CompetitionSubmissionError(f"source prediction not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise CompetitionSubmissionError(
                    f"source prediction NPZ missing key {key!r}"
                )
            data = np.asarray(archive[key])
    elif suffix in {".h5", ".hdf5"}:
        h5py = __import__("h5py")
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise CompetitionSubmissionError(
                    f"source prediction HDF5 missing dataset {key!r}"
                )
            data = np.asarray(handle[key][:])
    else:
        raise CompetitionSubmissionError(
            f"unsupported source prediction format: {path.suffix}"
        )
    if data.ndim != 3:
        raise CompetitionSubmissionError(
            f"expected rank-3 source prediction, got shape {data.shape}"
        )
    return data


def _validate_prediction(*, prediction: np.ndarray, context: np.ndarray) -> None:
    expected = (context.shape[0], PREDICTION_STEPS, SPACE_POINTS)
    if prediction.shape != expected:
        raise CompetitionSubmissionError(
            f"prediction shape mismatch: expected {expected}, got {prediction.shape}"
        )
    if not np.allclose(prediction[:, :CONTEXT_STEPS, :], context, atol=1e-6):
        raise CompetitionSubmissionError("first 10 prediction steps do not match input")


def _write_prediction(
    *, path: Path, dataset_key: str, prediction: np.ndarray
) -> None:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_key, data=prediction)


def _write_time_csv(*, path: Path, train_time: float, inference_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow(
            {
                "train_time": f"{train_time:.6f}",
                "inference_time": f"{inference_time:.6f}",
            }
        )


def _write_submission_json(*, path: Path, submission_id: str) -> None:
    payload = {
        "submission_id": submission_id,
        "problem_id": PROBLEM_ID,
        "code_path": "code",
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_log(
    *,
    path: Path,
    task_id: int,
    test_path: Path,
    dataset_key: str,
    prediction_dataset_key: str,
    sample_count: int,
    method: str,
    source_prediction_path: Path | None,
    train_time: float,
    inference_time: float,
) -> None:
    model_change = (
        "no learned weights; repeat the last observed context state over the 190 hidden steps."
        if method == "persistence"
        else "external predictor output wrapped into the official submission shape; first 10 steps are overwritten from the test context."
    )
    source_line = (
        f"- source_prediction_path: {source_prediction_path}"
        if source_prediction_path is not None
        else "- source_prediction_path: none"
    )
    path.write_text(
        "\n".join(
            [
                "Hermes PDE Agent Scientific Log",
                f"Task: {task_id}",
                f"Input HDF5: {test_path}",
                f"Input dataset key: {dataset_key}",
                f"Prediction dataset key: {prediction_dataset_key}",
                "",
                "Trace:",
                "- Competition requirement parsed as 200-step Burgers output with first 10 context steps preserved exactly.",
                f"- Hermes selected method={method} for this submission artifact.",
                source_line,
                "",
                "Experiment Tracking:",
                f"- version: competition_{method}_v0",
                f"- model_change: {model_change}",
                "- validation_contract: output shape is (N, 200, 256); first 10 steps are copied from the test context.",
                f"- samples: {sample_count}",
                f"- train_time_seconds: {train_time:.6f}",
                f"- inference_time_seconds: {inference_time:.6f}",
                "",
                "Conclusion:",
                "- This artifact is a structurally valid baseline submission and a stable target for subsequent Hermes strategy upgrades.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_code_stubs(*, model_path: Path, train_path: Path) -> None:
    model_path.write_text(
        '''"""Minimal persistence model used by the competition submission baseline."""

from __future__ import annotations

import numpy as np


def persistence_forecast(context: np.ndarray, *, total_steps: int = 200) -> np.ndarray:
    if context.ndim != 3:
        raise ValueError(f"expected rank-3 context tensor, got {context.shape}")
    prediction = np.empty((context.shape[0], total_steps, context.shape[2]), dtype=np.float32)
    prediction[:, : context.shape[1], :] = context.astype(np.float32)
    prediction[:, context.shape[1] :, :] = context[:, -1:, :].astype(np.float32)
    return prediction
''',
        encoding="utf-8",
    )
    train_path.write_text(
        '''"""No-op training entrypoint for the persistence submission baseline."""

from __future__ import annotations


def main() -> None:
    print("Persistence baseline has no trainable parameters.")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def _write_submission_zip(*, submission_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(submission_dir.parent))
