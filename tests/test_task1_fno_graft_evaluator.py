from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def test_task1_fno_graft_evaluator_selects_late_safe_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import task1_fno_graft_evaluator as module

    target = np.ones((2, 200, 256), dtype=np.float32)
    context = target[:, :10, :]
    baseline = target.copy()
    baseline[:, 10:20, :] = 2.0
    baseline[:, 105:, :] = target[:, 105:, :] + 0.1
    source = baseline.copy()
    source[:, 10:20, :] = target[:, 10:20, :]
    source[:, 105:, :] = target[:, 105:, :] + 9.0

    baseline_path = _write_hdf5(tmp_path / "baseline.hdf5", baseline)
    test_path = _write_hdf5(tmp_path / "test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "target.hdf5", target)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        module,
        "predict_task1_fno_checkpoint",
        lambda checkpoint_path, context: source,
    )

    report = module.run_task1_fno_graft_evaluator(
        baseline_prediction_path=baseline_path,
        checkpoint_path=checkpoint_path,
        test_path=test_path,
        target_path=target_path,
        output_dir=tmp_path / "graft",
        windows=[(10, 20), (10, 200)],
        blend_weights=[1.0],
        force=True,
    )

    best = report["best_candidate"]
    assert report["state_hint"] == "TASK1_FNO_GRAFT_EVALUATOR_RAN"
    assert best["window_start"] == 10
    assert best["window_stop"] == 20
    assert best["late_non_regression_passed"] is True
    assert best["task_prediction_score_delta"] > 0
    assert Path(report["artifacts"]["best_prediction_path"]).exists()


def test_task1_fno_graft_evaluator_cli_returns_payload(tmp_path: Path, monkeypatch) -> None:
    from pde_agent.cli import app
    from pde_agent.core import task1_fno_graft_evaluator as module
    from typer.testing import CliRunner

    target = np.ones((1, 200, 256), dtype=np.float32)
    context = target[:, :10, :]
    baseline = target.copy()
    baseline[:, 10:20, :] = 2.0
    source = target.copy()

    baseline_path = _write_hdf5(tmp_path / "baseline.hdf5", baseline)
    test_path = _write_hdf5(tmp_path / "test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "target.hdf5", target)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        module,
        "predict_task1_fno_checkpoint",
        lambda checkpoint_path, context: source,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-task1-fno-graft-evaluator",
            "--baseline-prediction-path",
            str(baseline_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--test-path",
            str(test_path),
            "--target-path",
            str(target_path),
            "--output-dir",
            str(tmp_path / "cli"),
            "--window",
            "10:20",
            "--blend-weight",
            "1.0",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TASK1_FNO_GRAFT_EVALUATOR_RAN" in result.output


def test_task1_fno_graft_from_optimization_report_uses_selected_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json

    from pde_agent.core import task1_fno_graft_evaluator as module

    target = np.ones((1, 200, 256), dtype=np.float32)
    context = target[:, :10, :]
    baseline = target.copy()
    baseline[:, 10:20, :] = 2.0
    source = target.copy()

    baseline_path = _write_hdf5(tmp_path / "baseline.hdf5", baseline)
    test_path = _write_hdf5(tmp_path / "test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "target.hdf5", target)
    checkpoint_path = tmp_path / "selected.pt"
    checkpoint_path.write_bytes(b"placeholder")
    optimization_report_path = tmp_path / "optimization.json"
    optimization_report_path.write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "candidate_id": "selected_fno",
                    "artifacts": {"checkpoint_path": str(checkpoint_path)},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "predict_task1_fno_checkpoint",
        lambda checkpoint_path, context: source,
    )

    report = module.run_task1_fno_graft_from_optimization_report(
        optimization_report_path=optimization_report_path,
        baseline_prediction_path=baseline_path,
        test_path=test_path,
        target_path=target_path,
        output_dir=tmp_path / "graft",
        windows=[(10, 20)],
        blend_weights=[1.0],
        force=True,
    )

    assert report["selected_training_candidate"]["candidate_id"] == "selected_fno"
    assert report["best_candidate"]["task_prediction_score_delta"] > 0


def test_predict_task1_fno_checkpoint_accepts_partial_forecast(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pde_agent.core import task1_fno_graft_evaluator as module

    context = np.ones((2, 10, 256), dtype=np.float32)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=False):
            return {
                "metadata": {
                    "model_family": "fno_lite",
                    "forecast_steps": 47,
                    "normalization": "none",
                    "context_steps": 10,
                    "space_points": 256,
                    "width": 8,
                    "modes": 4,
                    "depth": 1,
                },
                "model_state_dict": {},
            }

        @staticmethod
        def as_tensor(value, dtype=None):
            return value

        float32 = "float32"

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

    class FakePrediction:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full((2, 47, 256), 3.0, dtype=np.float32)

    class FakeModel:
        def eval(self):
            return None

        def __call__(self, value):
            return FakePrediction()

    monkeypatch.setattr(module, "_require_torch", lambda: FakeTorch)
    monkeypatch.setattr(module, "_build_model", lambda **kwargs: FakeModel())
    monkeypatch.setattr(module, "_load_model_state", lambda **kwargs: None)

    prediction = module.predict_task1_fno_checkpoint(
        checkpoint_path=checkpoint_path,
        context=context,
    )

    assert prediction.shape == (2, 200, 256)
    assert np.all(prediction[:, :10, :] == 1.0)
    assert np.all(prediction[:, 10:57, :] == 3.0)
    assert np.all(prediction[:, 57:, :] == 1.0)
