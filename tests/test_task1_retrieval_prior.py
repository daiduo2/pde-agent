import json
from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def _trajectory(offset: float, slope: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    frames = []
    for t in range(200):
        frames.append(offset + slope * t + 0.1 * np.sin(2.0 * np.pi * x))
    return np.asarray(frames, dtype=np.float32)


def test_retrieval_prior_excludes_self_for_leave_one_out() -> None:
    from pde_agent.core.task1_retrieval_prior import predict_retrieval_prior

    reference = np.stack(
        [
            _trajectory(0.0, 0.01),
            _trajectory(1.0, 0.02),
            _trajectory(2.0, -0.01),
        ],
        axis=0,
    )
    context = reference[:, :10, :]

    result = predict_retrieval_prior(
        reference=reference,
        context=context,
        k=1,
        mode="affine",
        leave_one_out=True,
    )

    assert result.prediction.shape == (3, 190, 256)
    assert result.neighbor_indices.shape == (3, 1)
    assert np.all(result.neighbor_indices[:, 0] != np.arange(3))
    assert np.all(np.isfinite(result.prediction))


def test_predict_task1_retrieval_prior_cli_writes_candidate_report(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    reference = np.stack(
        [
            _trajectory(0.0, 0.01),
            _trajectory(1.0, 0.02),
            _trajectory(2.0, -0.01),
        ],
        axis=0,
    )
    test = np.stack(
        [
            reference[0, :10, :],
            reference[1, :10, :],
        ],
        axis=0,
    )
    reference_path = _write_hdf5(tmp_path / "task1_val.hdf5", reference)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", test)
    output_dir = tmp_path / "retrieval"

    result = CliRunner().invoke(
        app,
        [
            "predict-task1-retrieval-prior",
            "--reference-path",
            str(reference_path),
            "--test-path",
            str(test_path),
            "--output-dir",
            str(output_dir),
            "--k",
            "2",
            "--mode",
            "affine",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "predict-task1-retrieval-prior"
    assert payload["state_hint"] == "TASK1_RETRIEVAL_PRIOR_PREDICTION_READY"
    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["forecast_steps"] == 190
    assert payload["metrics"]["reference_count"] == 3
    report_path = output_dir / "candidate_execution_report.json"
    prediction_path = output_dir / "candidate_prediction.npz"
    assert report_path.exists()
    assert prediction_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["next_command"] == "materialize-competition-candidate"
    assert report["candidate"]["model_family"] == "RetrievalPrior"
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
        neighbor_indices = archive["neighbor_indices"]
        metadata = json.loads(str(archive["metadata_json"]))
    assert prediction.shape == (2, 190, 256)
    assert neighbor_indices.shape == (2, 2)
    assert metadata["mode"] == "affine"
    assert metadata["k"] == 2
