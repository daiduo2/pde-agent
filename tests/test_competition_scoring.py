import json
from pathlib import Path

import numpy as np
import pytest


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def test_score_competition_prediction_is_perfect_for_matching_arrays(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_scoring import score_competition_prediction

    target = np.ones((2, 200, 256), dtype=np.float32)
    prediction_path = _write_hdf5(tmp_path / "task1_pred.hdf5", target)
    target_path = _write_hdf5(tmp_path / "task1_gt.hdf5", target)

    report = score_competition_prediction(
        task_id=1,
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=tmp_path / "score.json",
        prediction_dataset_key="tensor",
        target_dataset_key="tensor",
    )

    assert report["state_hint"] == "COMPETITION_PREDICTION_SCORED"
    assert report["segment_score"] == pytest.approx(100.0)
    assert report["task_prediction_score"] == pytest.approx(75.0)
    assert report["segments"]["early"]["time_start"] == 10
    assert report["segments"]["early"]["time_stop"] == 57
    assert report["segments"]["middle"]["time_start"] == 57
    assert report["segments"]["middle"]["time_stop"] == 105
    assert report["segments"]["late"]["time_start"] == 105
    assert report["segments"]["late"]["time_stop"] == 200
    assert Path(report["score_path"]).exists()


def test_score_competition_prediction_uses_future_steps_only(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_scoring import score_competition_prediction

    target = np.ones((1, 200, 256), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :10, :] = -100.0
    prediction[:, 10:57, :] = 2.0
    prediction_path = _write_hdf5(tmp_path / "task2_pred.hdf5", prediction)
    target_path = _write_hdf5(tmp_path / "task2_gt.hdf5", target)

    report = score_competition_prediction(
        task_id=2,
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=None,
        prediction_dataset_key="tensor",
        target_dataset_key="tensor",
    )

    assert report["segments"]["early"]["rel_mse"] == pytest.approx(1.0)
    assert report["segments"]["middle"]["rel_mse"] == pytest.approx(0.0)
    assert report["segments"]["late"]["rmse"] == pytest.approx(0.0)
    assert report["task_prediction_score"] == pytest.approx(
        report["segment_score"] * 1.5
    )


def test_score_competition_prediction_truncates_task2_validation_extra_steps(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_scoring import score_competition_prediction

    prediction = np.ones((2, 200, 256), dtype=np.float32)
    target = np.ones((2, 210, 256), dtype=np.float32)
    prediction_path = _write_hdf5(tmp_path / "task2_pred.hdf5", prediction)
    target_path = _write_hdf5(tmp_path / "task2_val.hdf5", target)

    report = score_competition_prediction(
        task_id=2,
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=None,
        prediction_dataset_key="tensor",
        target_dataset_key="tensor",
    )

    assert report["prediction_shape"] == [2, 200, 256]
    assert report["target_shape"] == [2, 200, 256]
    assert report["segment_score"] == pytest.approx(100.0)


def test_score_competition_prediction_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    target = np.ones((2, 200, 256), dtype=np.float32)
    prediction_path = _write_hdf5(tmp_path / "task1_pred.hdf5", target)
    target_path = _write_hdf5(tmp_path / "task1_gt.hdf5", target)
    output_path = tmp_path / "score.json"

    result = CliRunner().invoke(
        app,
        [
            "score-competition-submission",
            "--task-id",
            "1",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--output-path",
            str(output_path),
            "--prediction-dataset-key",
            "tensor",
            "--target-dataset-key",
            "tensor",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "score-competition-submission"
    assert payload["state_hint"] == "COMPETITION_PREDICTION_SCORED"
    assert payload["metrics"]["segment_score"] == pytest.approx(100.0)
    assert payload["next_hint"] == "use-score-to-select-competition-strategy"
