import json
from pathlib import Path

import pytest


def _write_source(path: Path) -> Path:
    path.write_text("fixture hdf5 placeholder", encoding="utf-8")
    return path


def _baseline_summary(output_dir: Path, start_index: int, score: float) -> dict[str, object]:
    windows = output_dir / "windows_f003.npz"
    prediction = output_dir / "persistence_prediction.npz"
    score_path = output_dir / "persistence_score.json"
    summary_path = output_dir / "real_hdf5_baseline_smoke_summary.json"
    for path in (windows, prediction, score_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{start_index}:{path.name}", encoding="utf-8")
    return {
        "state_hint": "REAL_HDF5_BASELINE_SMOKE_COMPLETED",
        "artifacts": {
            "windows_npz_path": str(windows),
            "prediction_npz_path": str(prediction),
            "score_path": str(score_path),
            "summary_path": str(summary_path),
        },
        "metrics": {
            "competition_like_score_short": score,
            "rel_mse": 0.4,
            "rmse": 0.2,
            "dominant_error_pattern": "overdamped_rollout",
            "physics_issue": "nonphysical_oscillation",
        },
    }


def test_run_spectral_residual_evidence_sweep_writes_strategy_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_agent.core.spectral_residual_evidence_sweep as sweep_module
    from pde_agent.core.spectral_residual_evidence_sweep import (
        run_spectral_residual_evidence_sweep,
    )

    source_path = _write_source(tmp_path / "burgers.hdf5")
    output_dir = tmp_path / "sweep"
    baseline_scores = {0: 10.0, 8: 12.0, 16: 11.0}
    spectral_scores = {0: 14.0, 8: 13.5, 16: 15.0}

    def fake_baseline(**kwargs: object) -> dict[str, object]:
        start_index = int(kwargs["start_index"])
        return _baseline_summary(
            Path(kwargs["output_dir"]),
            start_index=start_index,
            score=baseline_scores[start_index],
        )

    def fake_train(**kwargs: object) -> dict[str, object]:
        checkpoint = Path(kwargs["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
        return {
            "checkpoint_path": str(checkpoint),
            "model_type": "spectral_residual_ridge",
            "modes": kwargs["modes"],
            "ridge": kwargs["ridge"],
        }

    def fake_predict(**kwargs: object) -> dict[str, object]:
        prediction = Path(kwargs["output_path"])
        prediction.write_text("prediction", encoding="utf-8")
        return {"prediction_path": str(prediction)}

    def fake_score(**kwargs: object) -> dict[str, object]:
        output_path = Path(kwargs["output_path"])
        start_index = int(output_path.parent.parent.name.split("_")[1])
        output_path.write_text("score", encoding="utf-8")
        score = spectral_scores[start_index]
        return {
            "competition_like_score_short": score,
            "rel_mse": 0.1,
            "rmse": 0.05,
        }

    def fake_error(**kwargs: object) -> dict[str, object]:
        output_path = Path(kwargs["output_path"])
        output_path.write_text("error", encoding="utf-8")
        return {
            "dominant_error_pattern": "shock_region_error",
            "late_rel_mse": 0.2,
            "time_error_slope": 0.01,
            "shock_region_error_ratio": 1.3,
        }

    def fake_physics(**kwargs: object) -> dict[str, object]:
        output_path = Path(kwargs["output_path"])
        output_path.write_text("physics", encoding="utf-8")
        return {
            "physics_issue": "acceptable",
            "residual_ratio": 0.8,
            "total_variation_ratio": 1.0,
            "gradient_error_ratio": 0.9,
        }

    monkeypatch.setattr(sweep_module, "run_real_hdf5_baseline_smoke", fake_baseline)
    monkeypatch.setattr(sweep_module, "train_spectral_residual_baseline", fake_train)
    monkeypatch.setattr(sweep_module, "predict_spectral_residual_baseline", fake_predict)
    monkeypatch.setattr(sweep_module, "score_prediction_npz", fake_score)
    monkeypatch.setattr(sweep_module, "diagnose_rollout_error_npz", fake_error)
    monkeypatch.setattr(sweep_module, "check_burgers_physics_npz", fake_physics)

    evidence = run_spectral_residual_evidence_sweep(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key="tensor",
        start_indices=[0, 8, 16],
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

    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["model_family"] == "SpectralResidual"
    assert evidence["source_state_hint"] == "SPECTRAL_RESIDUAL_EVIDENCE_SWEEP_COMPLETED"
    assert evidence["evaluation_window_count"] == 3
    assert evidence["best_strategy_id"] == "spectral_residual_m04_ridge1e-03"
    assert evidence["mean_score_delta"] == pytest.approx((4.0 + 1.5 + 4.0) / 3.0)
    assert evidence["failure_rate"] == 0.0
    assert len(evidence["candidate_records"]) == 3
    assert evidence["candidate_records"][0]["start_index"] == 0
    assert evidence["candidate_records"][0]["score_delta"] == pytest.approx(4.0)
    assert evidence["strategy_ranking"][0]["record_count"] == 3
    assert Path(evidence["evidence_summary_path"]).exists()
    written = json.loads(
        (output_dir / "spectral_residual_evidence_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == evidence


def test_run_spectral_residual_evidence_sweep_rejects_existing_summary(
    tmp_path: Path,
) -> None:
    from pde_agent.core.spectral_residual_evidence_sweep import (
        SpectralResidualEvidenceSweepError,
        run_spectral_residual_evidence_sweep,
    )

    source_path = _write_source(tmp_path / "burgers.hdf5")
    output_dir = tmp_path / "sweep"
    output_dir.mkdir()
    (output_dir / "spectral_residual_evidence_summary.json").write_text(
        "old",
        encoding="utf-8",
    )

    with pytest.raises(SpectralResidualEvidenceSweepError, match="already exists"):
        run_spectral_residual_evidence_sweep(
            source_path=source_path,
            output_dir=output_dir,
            dataset_key="tensor",
            start_indices=[0, 8],
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


def test_run_spectral_residual_evidence_sweep_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_agent.commands.run_spectral_residual_evidence_sweep as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    source_path = _write_source(tmp_path / "burgers.hdf5")
    output_dir = tmp_path / "sweep"

    def fake_run(**_: object) -> dict[str, object]:
        evidence_path = output_dir / "spectral_residual_evidence_summary.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
            "model_family": "SpectralResidual",
            "evidence_summary_path": str(evidence_path),
            "evaluation_window_count": 3,
            "mean_score_delta": 3.1,
            "long_window_score_delta": 3.1,
            "failure_rate": 0.0,
            "candidate_records": [{}, {}, {}],
            "strategy_ranking": [{"strategy_id": "spectral_residual_m04_ridge1e-03"}],
            "best_strategy_id": "spectral_residual_m04_ridge1e-03",
            "next_command": "recommend-strategy-next-action",
        }

    monkeypatch.setattr(command_module, "run_spectral_residual_evidence_sweep", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "run-spectral-residual-evidence-sweep",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--start-index",
            "0",
            "--start-index",
            "8",
            "--start-index",
            "16",
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
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-spectral-residual-evidence-sweep"
    assert payload["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert payload["artifacts"]["strategy_evidence_summary_path"] == str(
        output_dir / "spectral_residual_evidence_summary.json"
    )
    assert payload["metrics"]["model_family"] == "SpectralResidual"
    assert payload["metrics"]["evaluation_window_count"] == 3
    assert payload["metrics"]["best_strategy_id"] == "spectral_residual_m04_ridge1e-03"
    assert payload["next_hint"] == "recommend-strategy-next-action"
