import numpy as np
import pytest


def test_short_window_corrector_preserves_context_and_replaces_configured_window() -> None:
    from pde_agent.core.task1_short_window_corrector import (
        fit_task1_short_window_corrector,
    )

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    reference = np.empty((8, 200, 256), dtype=np.float32)
    for sample in range(reference.shape[0]):
        phase = np.float32(0.04 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (np.float32(0.002 + 0.0004 * sample) * np.cos(x + phase)).astype(
            np.float32
        )
        for step in range(reference.shape[1]):
            reference[sample, step, :] = base + np.float32(step) * velocity

    context = reference[1:3, :10, :].copy()
    base_prediction = np.full((2, 200, 256), 7.0, dtype=np.float32)
    base_prediction[:, :10, :] = context

    result = fit_task1_short_window_corrector(
        context=context,
        base_prediction=base_prediction,
        reference=reference,
        start_step=20,
        stop_step=35,
        feature_count=8,
        ridge_alpha=1e-8,
        correction_weight=1.0,
    )

    prediction = result["prediction"]
    assert np.allclose(prediction[:, :10, :], context)
    assert np.all(prediction[:, 10:20, :] == 7.0)
    assert np.all(prediction[:, 35:, :] == 7.0)
    assert np.mean((prediction[:, 20:35, :] - reference[1:3, 20:35, :]) ** 2) < 1e-5
    assert result["metadata"]["candidate_family"] == "task1_short_window_corrector"
    assert result["metadata"]["start_step"] == 20
    assert result["metadata"]["stop_step"] == 35


def test_short_window_corrector_can_use_reduced_time_reference_grid() -> None:
    from pde_agent.core.task1_short_window_corrector import (
        fit_task1_short_window_corrector,
    )

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    raw_reference = np.empty((6, 200, 256), dtype=np.float32)
    for sample in range(raw_reference.shape[0]):
        phase = np.float32(0.03 * sample)
        base = np.sin(x + phase).astype(np.float32)
        velocity = (np.float32(0.003 + 0.0003 * sample) * np.cos(x + phase)).astype(
            np.float32
        )
        for raw_step in range(raw_reference.shape[1]):
            raw_reference[sample, raw_step, :] = base + np.float32(raw_step / 5) * velocity
    reduced = raw_reference[:, ::5, :]
    context = reduced[2:4, :10, :].copy()
    base_prediction = np.zeros((2, 200, 256), dtype=np.float32)
    base_prediction[:, :10, :] = context

    result = fit_task1_short_window_corrector(
        context=context,
        base_prediction=base_prediction,
        reference=raw_reference,
        start_step=20,
        stop_step=35,
        reference_time_stride=5,
        feature_count=8,
        ridge_alpha=1e-8,
        correction_weight=1.0,
    )

    prediction = result["prediction"]
    assert np.mean((prediction[:, 20:35, :] - reduced[2:4, 20:35, :]) ** 2) < 1e-5
    assert result["metadata"]["reference_time_stride"] == 5
    assert result["metadata"]["reference_time_steps"] == 40


def test_short_window_corrector_rejects_invalid_or_nonfinite_inputs() -> None:
    from pde_agent.core.task1_short_window_corrector import (
        Task1ShortWindowCorrectorError,
        fit_task1_short_window_corrector,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    base_prediction = np.zeros((1, 200, 256), dtype=np.float32)
    reference = np.zeros((4, 200, 256), dtype=np.float32)
    reference[0, 11, 0] = np.nan

    with pytest.raises(Task1ShortWindowCorrectorError, match="finite"):
        fit_task1_short_window_corrector(
            context=context,
            base_prediction=base_prediction,
            reference=reference,
        )

    with pytest.raises(Task1ShortWindowCorrectorError, match="start_step"):
        fit_task1_short_window_corrector(
            context=context,
            base_prediction=base_prediction,
            reference=np.zeros((4, 200, 256), dtype=np.float32),
            start_step=5,
            stop_step=20,
        )


def test_hypothesis_lab_sandbox_executes_task1_short_window_corrector_family(
    tmp_path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    context = np.zeros((2, 10, 256), dtype=np.float32)
    target = np.zeros((2, 200, 256), dtype=np.float32)
    reference = np.zeros((6, 200, 256), dtype=np.float32)
    for sample in range(reference.shape[0]):
        reference[sample, :, :] = np.float32(sample)
    target[:, :10, :] = context

    context_path = tmp_path / "task1_test.hdf5"
    target_path = tmp_path / "task1_val.hdf5"
    reference_path = tmp_path / "official_train.hdf5"
    _write_hdf5(context_path, context)
    _write_hdf5(target_path, target)
    _write_hdf5(reference_path, reference)

    blueprint = {
        "blueprint_id": "bp_task1_short_window_corrector_test",
        "candidate_family": "task1_short_window_corrector",
        "search_mode": "short_window_corrector_smoke",
        "parameter_grid": {
            "candidate_params": [
                {
                    "start_step": 20,
                    "stop_step": 35,
                    "feature_count": 4,
                    "ridge_alpha": 1e-4,
                    "correction_weight": 0.5,
                    "reference_time_stride": 1,
                    "corrector_mode": "context_pod_ridge",
                }
            ]
        },
        "sandbox": {
            "allowed_executor": "controlled_numpy_task1_short_window_corrector",
            "primary_metric": "task1_early_rel_mse_guarded_score",
            "selection_objective": "early_rel_mse",
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=4,
        output_dir=tmp_path / "sandbox",
        submission_id="team_short_window_test",
        force=False,
    )

    assert sandbox["candidate_family"] == "task1_short_window_corrector"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_short_window_corrector"
    )
    assert sandbox["best_candidate"]["metadata"]["uses_target_as_reference"] is False
    assert sandbox["artifacts"]["best_submission_dir"]


def _write_hdf5(path, data: np.ndarray) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
