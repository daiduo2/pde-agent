import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_scoring import (
    LocalScoreError,
    score_prediction_arrays,
    score_prediction_npz,
)


def test_score_prediction_arrays_perfect_prediction_scores_100() -> None:
    target = np.ones((2, 8, 4), dtype=np.float32)
    result = score_prediction_arrays(target.copy(), target)

    assert result["sample_count"] == 2
    assert result["time_steps"] == 8
    assert result["space_points"] == 4
    assert result["rmse"] == 0.0
    assert result["rel_mse"] == 0.0
    assert result["competition_like_score_short"] == pytest.approx(100.0)
    assert result["segments"]["early"]["score"] == pytest.approx(100.0)
    assert result["segments"]["middle"]["score"] == pytest.approx(100.0)
    assert result["segments"]["late"]["score"] == pytest.approx(100.0)


def test_score_prediction_arrays_noisy_prediction_lowers_score() -> None:
    target = np.ones((1, 8, 4), dtype=np.float32)
    prediction = target + 0.5

    result = score_prediction_arrays(prediction, target)

    assert result["rmse"] > 0.0
    assert result["rel_mse"] > 0.0
    assert result["competition_like_score_short"] < 100.0


def test_score_prediction_arrays_31_step_input_returns_finite_json_metrics() -> None:
    target = np.linspace(0.25, 2.0, num=2 * 31 * 5, dtype=np.float64).reshape(
        2, 31, 5
    )
    prediction = target + 0.01

    result = score_prediction_arrays(prediction, target)

    assert _numeric_values_are_finite(result)
    json.dumps(result, allow_nan=False)


def test_score_prediction_arrays_rejects_non_finite_prediction() -> None:
    target = np.ones((1, 8, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 0] = np.nan

    with pytest.raises(LocalScoreError, match="non-finite"):
        score_prediction_arrays(prediction, target)


def test_score_prediction_arrays_rejects_non_finite_derived_metrics() -> None:
    target = np.ones((1, 8, 4), dtype=np.float64)
    prediction = np.full((1, 8, 4), 1.0e200, dtype=np.float64)

    with pytest.raises(LocalScoreError, match="non-finite"):
        score_prediction_arrays(prediction, target)


def test_score_prediction_arrays_rejects_shape_mismatch() -> None:
    with pytest.raises(LocalScoreError, match="shape mismatch"):
        score_prediction_arrays(
            np.ones((1, 8, 4), dtype=np.float32),
            np.ones((1, 7, 4), dtype=np.float32),
        )


def test_score_prediction_npz_loads_prediction_and_target_files(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "score.json"
    target = np.ones((2, 8, 4), dtype=np.float32)
    np.savez(prediction_path, prediction=target.copy())
    np.savez(target_path, target=target)

    result = score_prediction_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=output_path,
    )

    assert result["competition_like_score_short"] == pytest.approx(100.0)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["competition_like_score_short"] == pytest.approx(100.0)


def test_score_prediction_npz_accepts_combined_file(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.npz"
    target = np.ones((1, 8, 4), dtype=np.float32)
    np.savez(combined_path, prediction=target.copy(), target=target)

    result = score_prediction_npz(
        prediction_path=combined_path,
        target_path=combined_path,
        output_path=None,
    )

    assert result["sample_count"] == 1
    assert result["competition_like_score_short"] == pytest.approx(100.0)


def test_score_prediction_npz_rejects_missing_prediction_key(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    np.savez(prediction_path, pred=np.ones((1, 8, 4), dtype=np.float32))
    np.savez(target_path, target=np.ones((1, 8, 4), dtype=np.float32))

    with pytest.raises(LocalScoreError, match="missing key 'prediction'"):
        score_prediction_npz(
            prediction_path=prediction_path,
            target_path=target_path,
            output_path=None,
        )


def _numeric_values_are_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_numeric_values_are_finite(child) for child in value.values())
    if isinstance(value, list | tuple):
        return all(_numeric_values_are_finite(child) for child in value)
    if isinstance(value, int | float):
        return bool(np.isfinite(value))
    return True
