"""Local rollout error diagnosis utilities for PDE predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class RolloutDiagnosisError(RuntimeError):
    pass


def diagnose_rollout_error_npz(
    *,
    prediction_path: str | Path,
    target_path: str | Path,
    score_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    """Diagnose rollout error patterns from prediction, target, and score files."""

    prediction_file = Path(prediction_path)
    target_file = Path(target_path)
    score_file = Path(score_path)

    prediction = _load_npz_array(prediction_file, "prediction")
    target = _load_npz_array(target_file, "target")
    score = _load_score(score_file)

    result = diagnose_rollout_error_arrays(
        prediction=prediction,
        target=target,
        score=score,
    )
    result["prediction_path"] = str(prediction_file)
    result["target_path"] = str(target_file)
    result["score_path"] = str(score_file)

    if output_path is not None:
        diagnosis_file = Path(output_path)
        result["diagnosis_path"] = str(diagnosis_file)
        diagnosis_file.parent.mkdir(parents=True, exist_ok=True)
        diagnosis_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return result


def diagnose_rollout_error_arrays(
    *,
    prediction: Any,
    target: Any,
    score: dict[str, Any],
) -> dict[str, Any]:
    """Compute rollout diagnosis metrics for arrays shaped [N, T, X]."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _validate_arrays(pred, truth)
    competition_score = _require_score(score)

    error = pred - truth
    _raise_if_non_finite(error, "derived error")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared_error = error**2
        truth_squared = truth**2
        _raise_if_non_finite(squared_error, "derived squared error")
        _raise_if_non_finite(truth_squared, "derived target squared")

        global_mse = float(np.mean(squared_error))
        global_rmse = float(np.sqrt(global_mse))
        target_mse = float(np.mean(truth_squared))
        global_rel_mse = float(global_mse / (target_mse + 1e-12))

        early_slice, middle_slice, late_slice = _segment_slices(pred.shape[1])
        early_rmse, early_rel_mse = _segment_metrics(
            squared_error,
            truth_squared,
            early_slice,
        )
        middle_rmse, middle_rel_mse = _segment_metrics(
            squared_error,
            truth_squared,
            middle_slice,
        )
        late_rmse, late_rel_mse = _segment_metrics(
            squared_error,
            truth_squared,
            late_slice,
        )

        rmse_by_time = np.sqrt(np.mean(squared_error, axis=(0, 2)))
        if pred.shape[1] == 1:
            time_error_slope = 0.0
        else:
            time_error_slope = float(
                np.polyfit(np.arange(pred.shape[1]), rmse_by_time, deg=1)[0]
            )
        max_error_time_index = int(np.argmax(rmse_by_time))

        spatial_abs_error = np.mean(np.abs(error), axis=(0, 1))
        mean_spatial_error = float(np.mean(spatial_abs_error))
        max_spatial_error = float(np.max(spatial_abs_error))

        prediction_total_variation = _total_variation(pred)
        target_total_variation = _total_variation(truth)
        shock_region_error_ratio = _shock_region_error_ratio(error, truth)
        sign_change_ratio = _sign_change_ratio(error)

        _raise_if_non_finite(
            [
                global_rmse,
                global_rel_mse,
                early_rel_mse,
                middle_rel_mse,
                late_rel_mse,
                early_rmse,
                middle_rmse,
                late_rmse,
                time_error_slope,
                mean_spatial_error,
                max_spatial_error,
                prediction_total_variation,
                target_total_variation,
                shock_region_error_ratio,
                sign_change_ratio,
            ],
            "derived diagnosis metrics",
        )

    dominant_error_pattern = _classify_pattern(
        competition_score=competition_score,
        global_rmse=global_rmse,
        early_rmse=early_rmse,
        late_rmse=late_rmse,
        prediction_total_variation=prediction_total_variation,
        target_total_variation=target_total_variation,
        shock_region_error_ratio=shock_region_error_ratio,
        sign_change_ratio=sign_change_ratio,
    )
    evidence = _build_evidence(
        dominant_error_pattern=dominant_error_pattern,
        competition_score=competition_score,
        global_rmse=global_rmse,
        early_rmse=early_rmse,
        late_rmse=late_rmse,
        prediction_total_variation=prediction_total_variation,
        target_total_variation=target_total_variation,
        shock_region_error_ratio=shock_region_error_ratio,
        sign_change_ratio=sign_change_ratio,
    )

    return {
        "sample_count": int(pred.shape[0]),
        "time_steps": int(pred.shape[1]),
        "space_points": int(pred.shape[2]),
        "competition_like_score_short": competition_score,
        "global_rmse": global_rmse,
        "global_rel_mse": global_rel_mse,
        "early_rel_mse": early_rel_mse,
        "middle_rel_mse": middle_rel_mse,
        "late_rel_mse": late_rel_mse,
        "early_rmse": early_rmse,
        "middle_rmse": middle_rmse,
        "late_rmse": late_rmse,
        "time_error_slope": time_error_slope,
        "max_error_time_index": max_error_time_index,
        "mean_spatial_error": mean_spatial_error,
        "max_spatial_error": max_spatial_error,
        "prediction_total_variation": prediction_total_variation,
        "target_total_variation": target_total_variation,
        "shock_region_error_ratio": shock_region_error_ratio,
        "sign_change_ratio": sign_change_ratio,
        "dominant_error_pattern": dominant_error_pattern,
        "evidence": evidence,
    }


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise RolloutDiagnosisError(f"missing path: {path}")
    if not path.is_file():
        raise RolloutDiagnosisError(f"path is not a file: {path}")

    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise RolloutDiagnosisError(f"{path} missing key '{key}'")
            return np.asarray(data[key])
    except RolloutDiagnosisError:
        raise
    except OSError as exc:
        raise RolloutDiagnosisError(f"failed to load NPZ file {path}: {exc}") from exc


def _load_score(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RolloutDiagnosisError(f"missing path: {path}")
    if not path.is_file():
        raise RolloutDiagnosisError(f"path is not a file: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RolloutDiagnosisError(
            f"failed to parse score JSON {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RolloutDiagnosisError("score JSON must be an object")
    _require_score(data)
    return data


def _require_score(score: dict[str, Any]) -> float:
    if "competition_like_score_short" not in score:
        raise RolloutDiagnosisError("score missing competition_like_score_short")
    value = score["competition_like_score_short"]
    if not isinstance(value, int | float):
        raise RolloutDiagnosisError("competition_like_score_short must be numeric")
    score_float = float(value)
    _raise_if_non_finite([score_float], "competition_like_score_short")
    return score_float


def _validate_arrays(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise RolloutDiagnosisError(
            "shape mismatch: "
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    if prediction.ndim != 3:
        raise RolloutDiagnosisError(
            f"expected rank 3 arrays shaped [N, T, X], got rank {prediction.ndim}"
        )
    if any(dimension <= 0 for dimension in prediction.shape):
        raise RolloutDiagnosisError(
            f"expected non-empty [N, T, X] dimensions, got shape {prediction.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise RolloutDiagnosisError("prediction contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise RolloutDiagnosisError("target contains non-finite values")


def _segment_slices(time_steps: int) -> tuple[slice, slice, slice]:
    if time_steps == 1:
        return slice(0, 1), slice(0, 1), slice(0, 1)
    if time_steps == 2:
        return slice(0, 1), slice(0, 1), slice(1, 2)

    early_stop = max(1, time_steps // 4)
    middle_stop = max(early_stop + 1, time_steps // 2)
    return (
        slice(0, early_stop),
        slice(early_stop, middle_stop),
        slice(middle_stop, time_steps),
    )


def _segment_metrics(
    squared_error: np.ndarray,
    truth_squared: np.ndarray,
    time_slice: slice,
) -> tuple[float, float]:
    segment_mse = float(np.mean(squared_error[:, time_slice, :]))
    segment_target_mse = float(np.mean(truth_squared[:, time_slice, :]))
    segment_rel_mse = float(segment_mse / (segment_target_mse + 1e-12))
    return float(np.sqrt(segment_mse)), segment_rel_mse


def _total_variation(array: np.ndarray) -> float:
    variation = 0.0
    if array.shape[1] > 1:
        variation += float(np.mean(np.abs(np.diff(array, axis=1))))
    if array.shape[2] > 1:
        variation += float(np.mean(np.abs(np.diff(array, axis=2))))
    return variation


def _shock_region_error_ratio(error: np.ndarray, target: np.ndarray) -> float:
    abs_error = np.abs(error)
    if target.shape[2] <= 1:
        return 1.0

    spatial_gradient = np.abs(np.diff(target, axis=2))
    gradient = np.pad(spatial_gradient, ((0, 0), (0, 0), (0, 1)), mode="edge")
    threshold = float(np.quantile(gradient, 0.75))
    shock_mask = gradient > threshold
    if not np.any(shock_mask):
        return 1.0

    shock_error = float(np.mean(abs_error[shock_mask]))
    non_shock_error = (
        float(np.mean(abs_error[~shock_mask])) if np.any(~shock_mask) else 0.0
    )
    return shock_error / (non_shock_error + 1e-12)


def _sign_change_ratio(error: np.ndarray) -> float:
    signs = np.sign(error)
    changes: list[np.ndarray] = []
    if error.shape[1] > 1:
        changes.append(signs[:, 1:, :] * signs[:, :-1, :] < 0)
    if error.shape[2] > 1:
        changes.append(signs[:, :, 1:] * signs[:, :, :-1] < 0)
    if not changes:
        return 0.0
    change_count = sum(int(np.count_nonzero(change)) for change in changes)
    comparison_count = sum(int(change.size) for change in changes)
    return float(change_count / comparison_count)


def _classify_pattern(
    *,
    competition_score: float,
    global_rmse: float,
    early_rmse: float,
    late_rmse: float,
    prediction_total_variation: float,
    target_total_variation: float,
    shock_region_error_ratio: float,
    sign_change_ratio: float,
) -> str:
    if competition_score >= 99.0 and global_rmse <= 1e-8:
        return "low_error"
    if late_rmse > max(early_rmse * 2.0, early_rmse + 1e-8):
        return "long_horizon_drift"
    if (
        target_total_variation > 1e-12
        and prediction_total_variation / target_total_variation < 0.5
    ):
        return "overdamped_rollout"
    if sign_change_ratio > 0.45 and global_rmse > 1e-6:
        return "oscillatory_error"
    if shock_region_error_ratio > 1.5 and global_rmse > 1e-6:
        return "shock_region_error"
    return "short_horizon_bias"


def _build_evidence(
    *,
    dominant_error_pattern: str,
    competition_score: float,
    global_rmse: float,
    early_rmse: float,
    late_rmse: float,
    prediction_total_variation: float,
    target_total_variation: float,
    shock_region_error_ratio: float,
    sign_change_ratio: float,
) -> list[str]:
    evidence = [
        f"score={competition_score:.6g}",
        f"global_rmse={global_rmse:.6g}",
        f"early_rmse={early_rmse:.6g}",
        f"late_rmse={late_rmse:.6g}",
    ]
    if target_total_variation > 1e-12:
        tv_ratio = prediction_total_variation / target_total_variation
        evidence.append(f"prediction_to_target_tv_ratio={tv_ratio:.6g}")
    else:
        evidence.append("target_total_variation_near_zero")
    evidence.append(f"sign_change_ratio={sign_change_ratio:.6g}")
    evidence.append(f"shock_region_error_ratio={shock_region_error_ratio:.6g}")
    evidence.append(f"classified_as={dominant_error_pattern}")
    return evidence


def _raise_if_non_finite(value: Any, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise RolloutDiagnosisError(f"{label} contains non-finite values")
