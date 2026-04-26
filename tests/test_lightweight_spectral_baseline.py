import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_scoring import score_prediction_arrays
from pde_agent.core.spectral_residual_baseline import (
    SpectralResidualBaselineError,
    predict_spectral_residual_baseline,
    train_spectral_residual_baseline,
)


def _write_windows(path: Path, context: np.ndarray, target: np.ndarray) -> None:
    full_reference = np.concatenate([context, target], axis=1)
    with path.open("wb") as handle:
        np.savez(
            handle,
            context=context.astype(np.float32),
            target=target.astype(np.float32),
            full_reference=full_reference.astype(np.float32),
            metadata_json=json.dumps({"fixture": True}),
        )


def _linear_trend_windows(
    *,
    sample_count: int = 6,
    context_steps: int = 4,
    forecast_steps: int = 5,
    space_points: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, space_points, endpoint=False)
    context = np.zeros((sample_count, context_steps, space_points), dtype=np.float32)
    target = np.zeros((sample_count, forecast_steps, space_points), dtype=np.float32)
    for sample in range(sample_count):
        base = np.sin(x + sample * 0.1)
        velocity = 0.05 * np.cos(x) + 0.02 * (sample + 1)
        for t in range(context_steps):
            context[sample, t, :] = base + velocity * t
        for k in range(forecast_steps):
            target[sample, k, :] = base + velocity * (context_steps + k)
    return context, target


def test_train_spectral_residual_baseline_writes_checkpoint(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)

    summary = train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        modes=6,
        ridge=1e-6,
        force=False,
    )

    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".npz").exists()
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        assert {
            "weights_real",
            "weights_imag",
            "feature_mean",
            "feature_scale",
            "metadata_json",
        } <= set(archive.files)
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["model_type"] == "spectral_residual_ridge"
    assert metadata["modes"] == 6
    assert metadata["forecast_steps"] == target.shape[1]
    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert summary["train_sample_count"] == context.shape[0]


def test_predict_spectral_residual_baseline_beats_persistence_on_linear_trend(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    prediction_path = tmp_path / "prediction"
    context, target = _linear_trend_windows()
    _write_windows(windows_path, context, target)
    train_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        modes=8,
        ridge=1e-8,
        force=False,
    )

    summary = predict_spectral_residual_baseline(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_path=prediction_path,
        force=False,
    )

    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["prediction"]
    persistence = np.repeat(context[:, -1:, :], repeats=target.shape[1], axis=1)
    model_score = score_prediction_arrays(prediction, target)
    persistence_score = score_prediction_arrays(persistence, target)
    assert prediction.shape == target.shape
    assert model_score["rmse"] < persistence_score["rmse"]
    assert summary["prediction_shape"] == list(target.shape)


def test_train_spectral_residual_baseline_rejects_invalid_modes(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    context, target = _linear_trend_windows(space_points=8)
    _write_windows(windows_path, context, target)

    with pytest.raises(SpectralResidualBaselineError, match="modes"):
        train_spectral_residual_baseline(
            windows_path=windows_path,
            checkpoint_path=tmp_path / "checkpoint.npz",
            modes=0,
            ridge=1e-3,
            force=False,
        )


def test_predict_spectral_residual_baseline_rejects_incompatible_windows(
    tmp_path: Path,
) -> None:
    train_windows = tmp_path / "train_windows.npz"
    predict_windows = tmp_path / "predict_windows.npz"
    checkpoint_path = tmp_path / "checkpoint.npz"
    context, target = _linear_trend_windows(space_points=16)
    _write_windows(train_windows, context, target)
    train_spectral_residual_baseline(
        windows_path=train_windows,
        checkpoint_path=checkpoint_path,
        modes=6,
        ridge=1e-6,
        force=False,
    )
    bad_context, bad_target = _linear_trend_windows(space_points=12)
    _write_windows(predict_windows, bad_context, bad_target)

    with pytest.raises(SpectralResidualBaselineError, match="incompatible"):
        predict_spectral_residual_baseline(
            windows_path=predict_windows,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "prediction.npz",
            force=False,
        )
