import json
from pathlib import Path

import numpy as np
import pytest


def _write_hdf5(path: Path, *, sample_count: int = 8) -> Path:
    h5py = pytest.importorskip("h5py")
    x = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    tensor = np.zeros((sample_count, 50, 16), dtype=np.float32)
    for sample in range(tensor.shape[0]):
        for time in range(tensor.shape[1]):
            tensor[sample, time, :] = (
                np.sin(x + 0.07 * sample + 0.04 * time)
                + 0.08 * np.sin(3.0 * x + 0.01 * time)
            )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
    return path


def test_run_burgers_spectral_viscosity_benchmark_writes_quantitative_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.burgers_spectral_viscosity_benchmark import (
        run_burgers_spectral_viscosity_benchmark,
    )

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=8)
    output_dir = tmp_path / "benchmark"

    report = run_burgers_spectral_viscosity_benchmark(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key="tensor",
        sample_limit=4,
        min_evaluation_windows=3,
        context_steps=10,
        forecast_steps=31,
        time_reduction=1,
        space_reduction=2,
        train_steps=[2],
        hidden_widths=[8],
        latent_dims=[4],
        learning_rates=[0.01],
        seeds=[5],
        spectral_weights=[0.05, 0.1],
        physics_weights=[0.005],
        proposal_id="burgers_spectral_viscosity_real_hdf5_v0",
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
    assert report["proposal_id"] == "burgers_spectral_viscosity_real_hdf5_v0"
    assert report["evaluation_window_count"] == 4
    assert report["metrics"]["baseline_score"] > 0.0
    assert report["metrics"]["candidate_record_count"] > 0
    assert report["metrics"]["combined_family_count"] == 2
    assert report["metrics"]["deeponet_record_count"] > 0
    assert report["metrics"]["unet_record_count"] > 0
    assert report["metrics"]["best_model_family"] in {"DeepONet", "UNet"}
    assert isinstance(report["metrics"]["mean_score_delta"], float)
    assert isinstance(report["metrics"]["long_window_score_delta"], float)
    assert report["decision"]["decision"] in {
        "evaluate_promotion",
        "expand_evidence",
        "retrieve_external_strategy",
    }
    assert report["decision"]["next_command"]
    assert set(report["inputs"]["training_strategies"]) == {
        "branch_trunk_supervised",
        "branch_trunk_physics_residual",
        "branch_trunk_spectral",
    }
    artifact_paths = report["artifacts"]
    for key in [
        "baseline_summary_path",
        "baseline_windows_path",
        "baseline_score_path",
        "deeponet_strategy_sweep_summary_path",
        "deeponet_evidence_summary_path",
        "unet_strategy_sweep_summary_path",
        "unet_evidence_summary_path",
        "strategy_evidence_summary_path",
        "strategy_decision_path",
        "benchmark_report_path",
    ]:
        assert Path(artifact_paths[key]).exists(), key
    written = json.loads(
        (output_dir / "burgers_spectral_viscosity_benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == report


def test_run_burgers_spectral_viscosity_benchmark_rejects_insufficient_windows(
    tmp_path: Path,
) -> None:
    from pde_agent.core.burgers_spectral_viscosity_benchmark import (
        BurgersSpectralViscosityBenchmarkError,
        run_burgers_spectral_viscosity_benchmark,
    )

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=4)
    output_dir = tmp_path / "benchmark"

    with pytest.raises(
        BurgersSpectralViscosityBenchmarkError,
        match="sample_limit must be at least min_evaluation_windows",
    ):
        run_burgers_spectral_viscosity_benchmark(
            source_path=source_path,
            output_dir=output_dir,
            dataset_key="tensor",
            sample_limit=2,
            min_evaluation_windows=3,
            context_steps=10,
            forecast_steps=31,
            time_reduction=1,
            space_reduction=2,
            train_steps=[2],
            hidden_widths=[8],
            latent_dims=[4],
            learning_rates=[0.01],
            seeds=[5],
            spectral_weights=[0.05],
            physics_weights=[0.005],
        )

    assert not (
        output_dir / "burgers_spectral_viscosity_benchmark_report.json"
    ).exists()


def test_run_burgers_spectral_viscosity_benchmark_force_overwrites_decision_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.burgers_spectral_viscosity_benchmark import (
        run_burgers_spectral_viscosity_benchmark,
    )

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=8)
    output_dir = tmp_path / "benchmark"
    kwargs = {
        "source_path": source_path,
        "output_dir": output_dir,
        "dataset_key": "tensor",
        "sample_limit": 4,
        "min_evaluation_windows": 3,
        "context_steps": 10,
        "forecast_steps": 31,
        "time_reduction": 1,
        "space_reduction": 2,
        "train_steps": [2],
        "hidden_widths": [8],
        "latent_dims": [4],
        "learning_rates": [0.01],
        "seeds": [5],
        "spectral_weights": [0.05],
        "physics_weights": [0.005],
        "force": True,
    }

    first = run_burgers_spectral_viscosity_benchmark(**kwargs)
    second = run_burgers_spectral_viscosity_benchmark(**kwargs)

    assert first["artifacts"]["strategy_decision_path"] == second["artifacts"][
        "strategy_decision_path"
    ]
    assert Path(second["artifacts"]["strategy_decision_path"]).exists()


def test_run_burgers_spectral_viscosity_benchmark_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    source_path = _write_hdf5(tmp_path / "burgers.hdf5", sample_count=8)
    output_dir = tmp_path / "benchmark"

    result = CliRunner().invoke(
        app,
        [
            "run-burgers-spectral-viscosity-benchmark",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--sample-limit",
            "4",
            "--min-evaluation-windows",
            "3",
            "--context-steps",
            "10",
            "--forecast-steps",
            "31",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--train-steps",
            "2",
            "--hidden-width",
            "8",
            "--latent-dim",
            "4",
            "--learning-rate",
            "0.01",
            "--seed",
            "5",
            "--spectral-weight",
            "0.05",
            "--spectral-weight",
            "0.1",
            "--physics-weight",
            "0.005",
            "--proposal-id",
            "burgers_spectral_viscosity_real_hdf5_v0",
            "--nu",
            "0.001",
            "--dt",
            "1.0",
            "--dx",
            "1.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-burgers-spectral-viscosity-benchmark"
    assert payload["state_hint"] == "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED"
    assert payload["artifacts"]["benchmark_report_path"] == str(
        output_dir / "burgers_spectral_viscosity_benchmark_report.json"
    )
    assert payload["metrics"]["evaluation_window_count"] == 4
    assert payload["metrics"]["best_model_family"] in {"DeepONet", "UNet"}
    assert (
        payload["metrics"]["proposal_id"]
        == "burgers_spectral_viscosity_real_hdf5_v0"
    )
    assert payload["metrics"]["decision"] in {
        "evaluate_promotion",
        "expand_evidence",
        "retrieve_external_strategy",
    }
    assert payload["next_hint"]
