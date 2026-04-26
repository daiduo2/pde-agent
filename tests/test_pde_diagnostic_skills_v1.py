import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app
from pde_agent.core.baseline_prediction import (
    BaselinePredictionError,
    generate_baseline_prediction,
)
from pde_agent.core.burgers_physics_local import (
    check_burgers_physics_arrays,
    check_burgers_physics_npz,
)
from pde_agent.core.next_experiment_plan import (
    NextExperimentPlanError,
    plan_next_experiment_from_files,
)
from pde_agent.core.pde_hypothesis import select_pde_hypothesis_from_files
from pde_agent.core.rollout_error_diagnosis import diagnose_rollout_error_npz


def _write_windows(path: Path, context: np.ndarray, target: np.ndarray) -> None:
    full_reference = np.concatenate([context, target], axis=1)
    np.savez(
        path,
        context=context.astype(np.float32),
        target=target.astype(np.float32),
        full_reference=full_reference.astype(np.float32),
        metadata_json=json.dumps({"fixture": True}),
    )


def _write_prediction(path: Path, prediction: np.ndarray) -> None:
    np.savez(path, prediction=prediction.astype(np.float32))


def test_generate_baseline_prediction_repeats_last_context_frame(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    context = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    target = np.zeros((2, 5, 4), dtype=np.float32)
    _write_windows(windows_path, context, target)

    summary = generate_baseline_prediction(
        windows_path=windows_path,
        output_path=output_path,
        method="persistence",
        force=False,
    )

    expected = np.repeat(context[:, -1:, :], repeats=5, axis=1)
    with np.load(output_path, allow_pickle=False) as archive:
        assert np.array_equal(archive["prediction"], expected)
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["method"] == "persistence"
    assert summary["prediction_shape"] == [2, 5, 4]
    assert summary["method"] == "persistence"


def test_generate_baseline_prediction_writes_exact_suffixless_path(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction_without_suffix"
    _write_windows(
        windows_path,
        np.ones((1, 2, 3), dtype=np.float32),
        np.zeros((1, 4, 3), dtype=np.float32),
    )

    summary = generate_baseline_prediction(
        windows_path=windows_path,
        output_path=output_path,
        method="persistence",
        force=False,
    )

    assert output_path.exists()
    assert not output_path.with_suffix(".npz").exists()
    assert summary["prediction_path"] == str(output_path)


def test_generate_baseline_prediction_refuses_overwrite_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((1, 2, 3), dtype=np.float32),
        np.ones((1, 4, 3), dtype=np.float32),
    )
    output_path.write_text("exists", encoding="utf-8")

    with pytest.raises(BaselinePredictionError, match="already exists"):
        generate_baseline_prediction(
            windows_path=windows_path,
            output_path=output_path,
            method="persistence",
            force=False,
        )


def test_generate_baseline_prediction_cli_writes_payload(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    output_path = tmp_path / "prediction.npz"
    _write_windows(
        windows_path,
        np.ones((2, 3, 4), dtype=np.float32),
        np.zeros((2, 5, 4), dtype=np.float32),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate-baseline-prediction",
            "--windows-path",
            str(windows_path),
            "--output-path",
            str(output_path),
            "--method",
            "persistence",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "generate-baseline-prediction"
    assert payload["state_hint"] == "BASELINE_PREDICTION_READY"
    assert payload["artifacts"]["prediction_npz_path"] == str(output_path)
    assert payload["metrics"] == {
        "method": "persistence",
        "sample_count": 2,
        "forecast_steps": 5,
        "space_points": 4,
    }
    assert output_path.exists()


def _write_score(path: Path, score: float = 80.0, rmse: float = 0.1) -> None:
    path.write_text(
        json.dumps(
            {
                "competition_like_score_short": score,
                "rmse": rmse,
                "rel_mse": rmse * rmse,
                "segments": {
                    "early": {"rel_mse_capped": rmse * rmse, "score": score},
                    "middle": {"rel_mse_capped": rmse * rmse, "score": score},
                    "late": {"rmse": rmse, "score": score},
                },
            }
        ),
        encoding="utf-8",
    )


def test_diagnose_rollout_error_low_error_for_perfect_prediction(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    output_path = tmp_path / "diagnosis.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)
    _write_score(score_path, score=100.0, rmse=0.0)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=output_path,
    )

    assert diagnosis["dominant_error_pattern"] == "low_error"
    assert diagnosis["global_rmse"] == pytest.approx(0.0)
    assert output_path.exists()


def test_diagnose_rollout_error_long_horizon_drift(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 4:, :] += 1.0
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)
    _write_score(score_path, score=20.0, rmse=0.5)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=None,
    )

    assert diagnosis["dominant_error_pattern"] == "long_horizon_drift"
    assert diagnosis["late_rmse"] > diagnosis["early_rmse"]


def test_diagnose_rollout_error_overdamped_rollout(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    x = np.linspace(0, 4 * np.pi, 16, dtype=np.float32)
    target = np.sin(x)[None, None, :] * np.ones((1, 8, 1), dtype=np.float32)
    prediction = np.zeros_like(target)
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)
    _write_score(score_path, score=30.0, rmse=0.5)

    diagnosis = diagnose_rollout_error_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        score_path=score_path,
        output_path=None,
    )

    assert diagnosis["dominant_error_pattern"] == "overdamped_rollout"
    assert diagnosis["prediction_total_variation"] < diagnosis["target_total_variation"]


def test_diagnose_rollout_error_cli_writes_payload(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    score_path = tmp_path / "score.json"
    output_path = tmp_path / "diagnostics" / "rollout_error.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)
    _write_score(score_path, score=100.0, rmse=0.0)

    result = CliRunner().invoke(
        app,
        [
            "diagnose-rollout-error",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--score-path",
            str(score_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "ROLLOUT_ERROR_DIAGNOSED"
    assert payload["artifacts"]["rollout_error_diagnosis_path"] == str(output_path)
    assert payload["metrics"]["dominant_error_pattern"] == "low_error"


def test_check_burgers_physics_returns_finite_metrics(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "physics.json"
    x = np.linspace(0, 2 * np.pi, 16, dtype=np.float32)
    t = np.linspace(0, 1, 8, dtype=np.float32)
    target = np.sin(x[None, None, :] + t[None, :, None]).astype(np.float32)
    _write_prediction(prediction_path, target.copy())
    np.savez(target_path, target=target)

    result = check_burgers_physics_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=output_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["physics_issue"] in {
        "stable_low_error",
        "residual_mismatch",
        "inconclusive",
    }
    assert result["prediction_residual_rmse"] >= 0.0
    assert output_path.exists()


def test_check_burgers_physics_flags_overdissipation(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    x = np.linspace(0, 4 * np.pi, 32, dtype=np.float32)
    target = np.sin(x)[None, None, :] * np.ones((1, 8, 1), dtype=np.float32)
    prediction = np.zeros_like(target)
    _write_prediction(prediction_path, prediction)
    np.savez(target_path, target=target)

    result = check_burgers_physics_npz(
        prediction_path=prediction_path,
        target_path=target_path,
        output_path=None,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["physics_issue"] == "overdissipation"
    assert result["total_variation_ratio"] < 0.5


def test_check_burgers_physics_mass_drift_uses_prediction_temporal_change() -> None:
    prediction = np.zeros((1, 3, 4), dtype=np.float32)
    prediction[:, -1, :] = 2.0
    target = prediction.copy()

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["mass_drift"] == pytest.approx(2.0)


def test_check_burgers_physics_energy_ratio_uses_final_step() -> None:
    prediction = np.ones((1, 3, 4), dtype=np.float32) * 10.0
    target = np.ones((1, 3, 4), dtype=np.float32)
    prediction[:, -1, :] = 1.0
    target[:, -1, :] = 2.0

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["energy_decay_ratio"] == pytest.approx(0.25)


def test_check_burgers_physics_shock_ratio_matches_plan_formula() -> None:
    target = np.array(
        [
            [
                [0.0, 0.0, 10.0, 10.0, 20.0],
                [0.0, 0.0, 10.0, 10.0, 20.0],
            ]
        ],
        dtype=np.float32,
    )
    abs_error = np.array(
        [
            [
                [1.0, 1.0, 4.0, 1.0, 4.0],
                [1.0, 1.0, 4.0, 1.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )
    prediction = target + abs_error

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    grad = np.abs(np.diff(target, axis=2, prepend=target[:, :, :1]))
    threshold = float(np.quantile(grad, 0.8))
    mask = grad >= threshold
    expected = float(np.mean(abs_error[mask]) / (np.mean(abs_error) + 1e-12))
    assert result["shock_region_error_ratio"] == pytest.approx(expected)


def test_check_burgers_physics_residual_ratio_and_notes_match_contract() -> None:
    prediction = np.ones((1, 3, 4), dtype=np.float32)
    target = prediction.copy()

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["residual_ratio"] == pytest.approx(0.0)
    assert (
        "Finite-difference diagnostics are relative proxies on downsampled local windows."
        in result["notes"]
    )


def test_check_burgers_physics_zero_fields_do_not_overdissipate() -> None:
    prediction = np.zeros((1, 3, 4), dtype=np.float32)
    target = prediction.copy()

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert np.isfinite(result["energy_decay_ratio"])
    assert np.isfinite(result["total_variation_ratio"])
    assert result["physics_issue"] != "overdissipation"


def test_check_burgers_physics_constant_fields_have_neutral_tv_ratio() -> None:
    prediction = np.ones((1, 3, 4), dtype=np.float32) * 3.0
    target = prediction.copy()

    result = check_burgers_physics_arrays(
        prediction=prediction,
        target=target,
        nu=0.001,
        dt=1.0,
        dx=1.0,
    )

    assert result["total_variation_ratio"] == pytest.approx(1.0)
    assert result["physics_issue"] != "overdissipation"


def test_check_burgers_physics_cli_writes_payload(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    target_path = tmp_path / "windows.npz"
    output_path = tmp_path / "diagnostics" / "physics.json"
    target = np.ones((1, 8, 4), dtype=np.float32)
    _write_prediction(prediction_path, target)
    np.savez(target_path, target=target)

    result = CliRunner().invoke(
        app,
        [
            "check-burgers-physics-local",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(target_path),
            "--output-path",
            str(output_path),
            "--nu",
            "0.001",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "BURGERS_PHYSICS_CHECKED"
    assert payload["artifacts"]["burgers_physics_check_path"] == str(output_path)
    assert "physics_issue" in payload["metrics"]


def _write_error_diagnosis(path: Path, pattern: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dominant_error_pattern": pattern,
                "global_rmse": 0.05,
                "time_error_slope": 0.01,
            }
        ),
        encoding="utf-8",
    )


def _write_physics_check(path: Path, issue: str) -> None:
    path.write_text(
        json.dumps(
            {
                "physics_issue": issue,
                "residual_ratio": 0.9,
                "total_variation_ratio": 2.7,
            }
        ),
        encoding="utf-8",
    )


def _write_hypothesis(path: Path, hypothesis: str) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_hypothesis": hypothesis,
                "next_experiment_scope": "local ablation",
            }
        ),
        encoding="utf-8",
    )


def test_select_pde_hypothesis_maps_overdamping(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    output_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=40.0, rmse=0.5)
    _write_error_diagnosis(error_path, "overdamped_rollout")
    _write_physics_check(physics_path, "overdissipation")

    result = select_pde_hypothesis_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        output_path=output_path,
    )

    assert result["selected_hypothesis"] == "add_spectral_consistency_loss"
    assert result["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert output_path.exists()


def test_select_pde_hypothesis_accepts_overdamped_alias(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    _write_score(score_path, score=40.0, rmse=0.5)
    _write_error_diagnosis(error_path, "overdamped")
    _write_physics_check(physics_path, "inconclusive")

    result = select_pde_hypothesis_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        output_path=None,
    )

    assert result["selected_hypothesis"] == "add_spectral_consistency_loss"


def test_select_pde_hypothesis_maps_long_horizon_drift(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    _write_score(score_path, score=35.0, rmse=0.6)
    _write_error_diagnosis(error_path, "long_horizon_drift")
    _write_physics_check(physics_path, "inconclusive")

    result = select_pde_hypothesis_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        output_path=None,
    )

    assert result["selected_hypothesis"] == "use_scheduled_sampling_rollout_finetune"


def test_select_pde_hypothesis_cli_writes_payload(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    output_path = tmp_path / "diagnostics" / "hypothesis.json"
    _write_score(score_path, score=40.0, rmse=0.5)
    _write_error_diagnosis(error_path, "overdamped_rollout")
    _write_physics_check(physics_path, "overdissipation")

    result = CliRunner().invoke(
        app,
        [
            "select-pde-hypothesis",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert payload["artifacts"]["selected_pde_hypothesis_path"] == str(output_path)
    assert "selected_hypothesis" in payload["metrics"]


def test_plan_next_experiment_low_pass_for_nonphysical_oscillation(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    output_path = tmp_path / "next_experiment_plan.json"
    _write_score(score_path, score=77.18, rmse=0.0506)
    _write_error_diagnosis(error_path, "shock_region_error")
    _write_physics_check(physics_path, "nonphysical_oscillation")
    _write_hypothesis(hypothesis_path, "apply_low_pass_stabilization")

    plan = plan_next_experiment_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        hypothesis_path=hypothesis_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert plan["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert plan["selected_intervention"] == "apply_low_pass_stabilization"
    assert plan["source_hypothesis"] == "apply_low_pass_stabilization"
    assert plan["failure_mode"]["score_band"] == "medium"
    assert plan["change_scope"] == "postprocess_or_inference_only"
    assert plan["budget"]["requires_training"] is False
    assert plan["budget"]["max_experiment_count"] == 3
    assert plan["intervention_plan"]["candidate_parameters"][
        "low_pass_cutoff_ratio"
    ] == [0.5, 0.625, 0.75]
    assert plan["next_command"] == "generate-experiment-config"


def test_plan_next_experiment_long_horizon_drift_requests_rollout_finetune(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=45.0, rmse=0.5)
    _write_error_diagnosis(error_path, "long_horizon_drift")
    _write_physics_check(physics_path, "inconclusive")
    _write_hypothesis(hypothesis_path, "use_scheduled_sampling_rollout_finetune")

    plan = plan_next_experiment_from_files(
        score_path=score_path,
        error_diagnosis_path=error_path,
        physics_check_path=physics_path,
        hypothesis_path=hypothesis_path,
        output_path=None,
    )

    assert plan["selected_intervention"] == "use_scheduled_sampling_rollout_finetune"
    assert plan["change_scope"] == "training_schedule_only"
    assert plan["budget"]["requires_training"] is True
    assert plan["budget"]["max_experiment_count"] == 2
    assert plan["failure_mode"]["score_band"] == "low"
    assert plan["intervention_plan"]["candidate_parameters"]["rollout_steps"] == [
        10,
        20,
        31,
    ]


def test_plan_next_experiment_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    _write_score(score_path, score=77.18, rmse=0.0506)
    _write_error_diagnosis(error_path, "shock_region_error")
    physics_path.write_text(
        json.dumps(
            {
                "residual_ratio": 0.9,
                "total_variation_ratio": 2.7,
            }
        ),
        encoding="utf-8",
    )
    _write_hypothesis(hypothesis_path, "apply_low_pass_stabilization")

    with pytest.raises(NextExperimentPlanError, match="physics_issue"):
        plan_next_experiment_from_files(
            score_path=score_path,
            error_diagnosis_path=error_path,
            physics_check_path=physics_path,
            hypothesis_path=hypothesis_path,
            output_path=None,
        )


def test_plan_next_experiment_cli_writes_payload(tmp_path: Path) -> None:
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    output_path = tmp_path / "diagnostics" / "next_experiment_plan.json"
    _write_score(score_path, score=77.18, rmse=0.0506)
    _write_error_diagnosis(error_path, "shock_region_error")
    _write_physics_check(physics_path, "nonphysical_oscillation")
    _write_hypothesis(hypothesis_path, "apply_low_pass_stabilization")

    result = CliRunner().invoke(
        app,
        [
            "plan-next-experiment",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--hypothesis-path",
            str(hypothesis_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "plan-next-experiment"
    assert payload["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert payload["artifacts"]["next_experiment_plan_path"] == str(output_path)
    assert payload["metrics"] == {
        "selected_intervention": "apply_low_pass_stabilization",
        "change_scope": "postprocess_or_inference_only",
        "max_experiment_count": 3,
        "next_command": "generate-experiment-config",
    }
    assert payload["next_hint"] == "generate-experiment-config"
    assert output_path.exists()


def test_local_pde_diagnostic_chain_cli_end_to_end(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    prediction_path = tmp_path / "prediction.npz"
    score_path = tmp_path / "score.json"
    error_path = tmp_path / "error.json"
    physics_path = tmp_path / "physics.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    context = np.ones((1, 3, 8), dtype=np.float32)
    target = np.zeros((1, 6, 8), dtype=np.float32)
    _write_windows(windows_path, context, target)

    commands = [
        [
            "generate-baseline-prediction",
            "--windows-path",
            str(windows_path),
            "--output-path",
            str(prediction_path),
        ],
        [
            "score-local-rollout",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--output-path",
            str(score_path),
        ],
        [
            "diagnose-rollout-error",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--score-path",
            str(score_path),
            "--output-path",
            str(error_path),
        ],
        [
            "check-burgers-physics-local",
            "--prediction-path",
            str(prediction_path),
            "--target-path",
            str(windows_path),
            "--output-path",
            str(physics_path),
        ],
        [
            "select-pde-hypothesis",
            "--score-path",
            str(score_path),
            "--error-diagnosis-path",
            str(error_path),
            "--physics-check-path",
            str(physics_path),
            "--output-path",
            str(hypothesis_path),
        ],
    ]

    for command in commands:
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"

    for artifact in (prediction_path, score_path, error_path, physics_path, hypothesis_path):
        assert artifact.exists()
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    assert hypothesis["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
