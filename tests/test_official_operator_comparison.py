import json
from pathlib import Path

import pytest


def _write_source(path: Path) -> Path:
    path.write_text("fixture hdf5 placeholder", encoding="utf-8")
    return path


def _baseline_summary(output_dir: Path, score: float = 12.0) -> dict[str, object]:
    windows = output_dir / "windows_f003.npz"
    prediction = output_dir / "persistence_prediction.npz"
    score_path = output_dir / "persistence_score.json"
    for path in (windows, prediction, score_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    return {
        "state_hint": "REAL_HDF5_BASELINE_SMOKE_COMPLETED",
        "artifacts": {
            "windows_npz_path": str(windows),
            "prediction_npz_path": str(prediction),
            "score_path": str(score_path),
        },
        "metrics": {
            "competition_like_score_short": score,
            "rel_mse": 0.7,
            "rmse": 0.3,
            "dominant_error_pattern": "late_horizon_drift",
            "physics_issue": "overdissipation",
        },
    }


def _score(score: float, rel_mse: float, rmse: float) -> dict[str, float]:
    return {
        "competition_like_score_short": score,
        "rel_mse": rel_mse,
        "rmse": rmse,
    }


def _rollout_error(pattern: str) -> dict[str, object]:
    return {
        "dominant_error_pattern": pattern,
        "late_rel_mse": 0.4,
        "time_error_slope": 0.01,
        "shock_region_error_ratio": 1.2,
    }


def _physics(issue: str) -> dict[str, object]:
    return {
        "physics_issue": issue,
        "residual_ratio": 0.9,
        "total_variation_ratio": 1.1,
        "gradient_error_ratio": 1.0,
    }


def _diagnosis_summary(
    *,
    model_type: str,
    adapter_type: str,
    checkpoint_path: str,
    output_dir: Path,
    score: float,
    rel_mse: float,
    rmse: float,
) -> dict[str, object]:
    prediction = output_dir / f"{model_type}_prediction.npz"
    score_path = output_dir / f"{model_type}_score.json"
    for path in (prediction, score_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    return {
        "state_hint": f"{model_type.upper()}_ROLLOUT_DIAGNOSED",
        "model": {
            "model_type": model_type,
            "adapter_type": adapter_type,
            "checkpoint_path": checkpoint_path,
            "checkpoint_format": "torch_serialized",
        },
        "artifacts": {
            "prediction_npz_path": str(prediction),
            "score_json_path": str(score_path),
        },
        "score": _score(score, rel_mse, rmse),
        "error_diagnosis": _rollout_error("short_horizon_bias"),
        "physics": _physics("dispersion_error"),
        "summary_path": str(output_dir / f"{model_type}_rollout_summary.json"),
    }


def test_run_official_operator_comparison_writes_ranked_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_agent.core.official_operator_comparison as comparison
    from pde_agent.core.official_operator_comparison import (
        run_official_operator_comparison,
    )

    source_path = _write_source(tmp_path / "burgers.hdf5")
    fno_checkpoint = _write_source(tmp_path / "burgers_FNO.tar")
    unet_checkpoint = _write_source(tmp_path / "burgers_Unet.tar")
    output_dir = tmp_path / "comparison"

    def fake_baseline(**kwargs: object) -> dict[str, object]:
        return _baseline_summary(Path(kwargs["output_dir"]), score=12.0)

    def fake_train_spectral(**kwargs: object) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
        return {
            "checkpoint_path": str(checkpoint),
            "model_type": "spectral_residual_ridge",
            "modes": kwargs["modes"],
            "ridge": kwargs["ridge"],
        }

    def fake_predict_spectral(**kwargs: object) -> dict[str, object]:
        prediction = Path(kwargs["output_path"])
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text("prediction", encoding="utf-8")
        return {"prediction_path": str(prediction)}

    def fake_score(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("score", encoding="utf-8")
        return _score(18.0, 0.5, 0.2)

    def fake_error(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_path"])
        output.write_text("error", encoding="utf-8")
        return _rollout_error("spectral_tail_error")

    def fake_physics(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_path"])
        output.write_text("physics", encoding="utf-8")
        return _physics("acceptable")

    def fake_fno(**kwargs: object) -> dict[str, object]:
        return _diagnosis_summary(
            model_type="fno",
            adapter_type="pdebench_fno1d",
            checkpoint_path=str(kwargs["checkpoint_path"]),
            output_dir=Path(kwargs["output_dir"]),
            score=23.0,
            rel_mse=0.25,
            rmse=0.12,
        )

    def fake_unet(**kwargs: object) -> dict[str, object]:
        return _diagnosis_summary(
            model_type="unet",
            adapter_type="pdebench_unet1d",
            checkpoint_path=str(kwargs["checkpoint_path"]),
            output_dir=Path(kwargs["output_dir"]),
            score=9.0,
            rel_mse=0.9,
            rmse=0.4,
        )

    monkeypatch.setattr(comparison, "run_real_hdf5_baseline_smoke", fake_baseline)
    monkeypatch.setattr(comparison, "train_spectral_residual_baseline", fake_train_spectral)
    monkeypatch.setattr(comparison, "predict_spectral_residual_baseline", fake_predict_spectral)
    monkeypatch.setattr(comparison, "score_prediction_npz", fake_score)
    monkeypatch.setattr(comparison, "diagnose_rollout_error_npz", fake_error)
    monkeypatch.setattr(comparison, "check_burgers_physics_npz", fake_physics)
    monkeypatch.setattr(comparison, "diagnose_fno_rollout", fake_fno)
    monkeypatch.setattr(comparison, "diagnose_unet_rollout", fake_unet)

    report = run_official_operator_comparison(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key="tensor",
        fno_checkpoint_path=fno_checkpoint,
        unet_checkpoint_path=unet_checkpoint,
        sample_limit=4,
        context_steps=10,
        forecast_steps=3,
        time_reduction=1,
        space_reduction=2,
        spectral_modes=4,
        spectral_ridge=0.001,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "OFFICIAL_OPERATOR_COMPARISON_COMPLETED"
    assert report["best_model_id"] == "official_fno"
    assert report["recommendation"]["next_action"] == "promote_official_model_candidate"
    assert [row["model_id"] for row in report["leaderboard"]] == [
        "official_fno",
        "spectral_residual",
        "persistence",
        "official_unet",
    ]
    assert report["model_records"][0]["model_id"] == "persistence"
    assert report["model_records"][1]["score_delta_vs_persistence"] == pytest.approx(6.0)
    assert report["model_records"][2]["score_delta_vs_persistence"] == pytest.approx(11.0)
    assert report["model_records"][3]["score_delta_vs_persistence"] == pytest.approx(-3.0)
    assert Path(report["artifacts"]["comparison_report_path"]).exists()
    written = json.loads(
        (output_dir / "official_operator_comparison_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_run_official_operator_comparison_rejects_existing_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.official_operator_comparison import (
        OfficialOperatorComparisonError,
        run_official_operator_comparison,
    )

    source_path = _write_source(tmp_path / "burgers.hdf5")
    fno_checkpoint = _write_source(tmp_path / "burgers_FNO.tar")
    unet_checkpoint = _write_source(tmp_path / "burgers_Unet.tar")
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    (output_dir / "official_operator_comparison_report.json").write_text(
        "old",
        encoding="utf-8",
    )

    with pytest.raises(OfficialOperatorComparisonError, match="already exists"):
        run_official_operator_comparison(
            source_path=source_path,
            output_dir=output_dir,
            dataset_key="tensor",
            fno_checkpoint_path=fno_checkpoint,
            unet_checkpoint_path=unet_checkpoint,
            sample_limit=4,
            context_steps=10,
            forecast_steps=3,
            time_reduction=1,
            space_reduction=2,
            spectral_modes=4,
            spectral_ridge=0.001,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )


def test_run_official_operator_comparison_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_agent.commands.run_official_operator_comparison as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    source_path = _write_source(tmp_path / "burgers.hdf5")
    fno_checkpoint = _write_source(tmp_path / "burgers_FNO.tar")
    unet_checkpoint = _write_source(tmp_path / "burgers_Unet.tar")
    output_dir = tmp_path / "comparison"

    def fake_run(**_: object) -> dict[str, object]:
        report_path = output_dir / "official_operator_comparison_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "OFFICIAL_OPERATOR_COMPARISON_COMPLETED",
            "best_model_id": "official_fno",
            "best_model_family": "FNO",
            "best_score": 23.0,
            "best_score_delta_vs_persistence": 11.0,
            "model_count": 4,
            "official_model_count": 2,
            "artifacts": {"comparison_report_path": str(report_path)},
            "recommendation": {
                "next_action": "promote_official_model_candidate",
            },
        }

    monkeypatch.setattr(command_module, "run_official_operator_comparison", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "run-official-operator-comparison",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--fno-checkpoint-path",
            str(fno_checkpoint),
            "--unet-checkpoint-path",
            str(unet_checkpoint),
            "--sample-limit",
            "4",
            "--context-steps",
            "10",
            "--forecast-steps",
            "3",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--spectral-modes",
            "4",
            "--spectral-ridge",
            "0.001",
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-official-operator-comparison"
    assert payload["state_hint"] == "OFFICIAL_OPERATOR_COMPARISON_COMPLETED"
    assert payload["metrics"]["best_model_id"] == "official_fno"
    assert payload["metrics"]["official_model_count"] == 2
    assert payload["next_hint"] == "promote_official_model_candidate"
