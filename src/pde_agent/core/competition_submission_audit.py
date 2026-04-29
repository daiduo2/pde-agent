"""Audit generated competition submission directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.competition_submission import (
    CONTEXT_STEPS,
    PREDICTION_STEPS,
    SPACE_POINTS,
)


STATE_HINT = "COMPETITION_SUBMISSION_AUDITED"


def audit_competition_submission(
    *,
    task_id: int,
    submission_dir: str | Path,
    test_path: str | Path,
    output_path: str | Path | None,
    dataset_key: str = "tensor",
) -> dict[str, Any]:
    root = Path(submission_dir)
    issues: list[dict[str, str]] = []
    paths = {
        "prediction": root / f"task{task_id}_pred.hdf5",
        "time": root / f"task{task_id}_time.csv",
        "log": root / f"task{task_id}_logs.log",
        "submission_json": root / "submission.json",
        "code_dir": root / "code",
    }

    _check_prediction(
        prediction_path=paths["prediction"],
        test_path=Path(test_path),
        dataset_key=dataset_key,
        issues=issues,
    )
    _check_time_csv(paths["time"], issues)
    _check_log(paths["log"], issues)
    _check_submission_json(paths["submission_json"], issues)
    _check_code_dir(paths["code_dir"], issues)

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "submission_dir": str(root),
        "test_path": str(test_path),
        "decision": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issue_codes": [issue["code"] for issue in issues],
        "issues": issues,
    }
    if output_path is not None:
        audit_path = Path(output_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        report["audit_path"] = str(audit_path)
        audit_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _check_prediction(
    *,
    prediction_path: Path,
    test_path: Path,
    dataset_key: str,
    issues: list[dict[str, str]],
) -> None:
    if not prediction_path.exists():
        issues.append({"code": "missing_task_prediction", "detail": str(prediction_path)})
        return
    try:
        prediction = _load_hdf5(path=prediction_path, dataset_key=dataset_key)
        context = _load_context(path=test_path, dataset_key=dataset_key)
    except RuntimeError as exc:
        issues.append({"code": "invalid_prediction_hdf5", "detail": str(exc)})
        return
    if prediction.shape != (context.shape[0], PREDICTION_STEPS, SPACE_POINTS):
        issues.append(
            {
                "code": "invalid_prediction_shape",
                "detail": f"got {prediction.shape}, expected {(context.shape[0], PREDICTION_STEPS, SPACE_POINTS)}",
            }
        )
        return
    if not np.allclose(prediction[:, :CONTEXT_STEPS, :], context, atol=1e-3):
        issues.append(
            {
                "code": "initial_context_mismatch",
                "detail": "first 10 prediction steps differ from test context",
            }
        )


def _check_time_csv(path: Path, issues: list[dict[str, str]]) -> None:
    if not path.exists():
        issues.append({"code": "missing_task_time_csv", "detail": str(path)})
        return
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    except OSError as exc:
        issues.append({"code": "invalid_task_time_csv", "detail": str(exc)})
        return
    if len(rows) != 1 or set(rows[0]) != {"train_time", "inference_time"}:
        issues.append(
            {
                "code": "invalid_task_time_csv",
                "detail": "expected one row with train_time and inference_time",
            }
        )
        return
    for key in ("train_time", "inference_time"):
        try:
            value = float(rows[0][key])
        except ValueError:
            issues.append({"code": "invalid_task_time_csv", "detail": key})
            return
        if value < 0.0:
            issues.append({"code": "invalid_task_time_csv", "detail": key})
            return


def _check_log(path: Path, issues: list[dict[str, str]]) -> None:
    if not path.exists():
        issues.append({"code": "missing_task_log", "detail": str(path)})
        return
    text = path.read_text(encoding="utf-8")
    if "Experiment Tracking" not in text:
        issues.append(
            {
                "code": "missing_experiment_tracking",
                "detail": str(path),
            }
        )


def _check_submission_json(path: Path, issues: list[dict[str, str]]) -> None:
    if not path.exists():
        issues.append({"code": "missing_submission_json", "detail": str(path)})
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append({"code": "invalid_submission_json", "detail": str(exc)})
        return
    if payload.get("problem_id") != "PDE_Burgers" or payload.get("code_path") != "code":
        issues.append(
            {
                "code": "invalid_submission_json",
                "detail": "expected problem_id=PDE_Burgers and code_path=code",
            }
        )


def _check_code_dir(path: Path, issues: list[dict[str, str]]) -> None:
    if not path.exists() or not path.is_dir() or not any(item.is_file() for item in path.iterdir()):
        issues.append({"code": "missing_code_dir", "detail": str(path)})


def _load_hdf5(*, path: Path, dataset_key: str) -> np.ndarray:
    h5py = __import__("h5py")
    try:
        with h5py.File(path, "r") as handle:
            if dataset_key not in handle:
                raise RuntimeError(f"missing dataset {dataset_key!r}")
            return np.asarray(handle[dataset_key][:], dtype=np.float32)
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def _load_context(*, path: Path, dataset_key: str) -> np.ndarray:
    data = _load_hdf5(path=path, dataset_key=dataset_key)
    if data.ndim != 3 or data.shape[1] < CONTEXT_STEPS:
        raise RuntimeError(f"invalid test context shape {data.shape}")
    if data.shape[2] == SPACE_POINTS:
        return data[:, :CONTEXT_STEPS, :]
    if data.shape[2] == 1024:
        return data[:, :CONTEXT_STEPS, ::4]
    raise RuntimeError(f"invalid test context spatial width {data.shape[2]}")
