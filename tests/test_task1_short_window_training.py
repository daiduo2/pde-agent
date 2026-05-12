from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def test_task1_short_window_training_loop_writes_artifacts_and_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers.hdf5"
    data = np.zeros((12, 60, 64), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        phase = 0.05 * sample
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + phase + 0.03 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "run",
        dataset_key="tensor",
        train_sample_limit=8,
        val_sample_limit=4,
        context_steps=10,
        forecast_steps=20,
        time_reduction=1,
        space_reduction=2,
        model_family="conv1d_residual",
        train_steps=2,
        batch_size=4,
        learning_rate=1e-3,
        seed=7,
        force=False,
    )

    assert report["state_hint"] == "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
    assert report["data"]["train"]["context_shape"] == [8, 10, 32]
    assert report["data"]["val"]["target_shape"] == [4, 20, 32]
    assert report["model"]["model_family"] == "conv1d_residual"
    assert report["training"]["train_steps"] == 2
    assert len(report["training"]["loss_curve"]["train"]) == 2
    assert len(report["training"]["loss_curve"]["val"]) == 2
    assert report["metrics"]["early"]["rel_mse"] >= 0.0
    assert report["metrics"]["early"]["score"] >= 0.0
    assert report["metrics"]["middle_guardrail"]["evaluated"] is False
    assert report["evidence"]["model_family"] == "conv1d_residual"
    assert report["evidence"]["horizon"]["context_steps"] == 10
    assert report["evidence"]["horizon"]["forecast_steps"] == 20
    assert report["evidence"]["decision"] in {
        "continue_training",
        "change_loss_or_horizon",
        "abandon_model_family",
    }
    assert report["next_generation_constraints"]["training_model_families"]
    assert report["next_command"] in {
        "run-task1-short-window-training-loop",
        "build-hypothesis-algorithm-design",
    }

    for key in [
        "checkpoint_path",
        "config_path",
        "training_log_path",
        "prediction_path",
        "evidence_path",
    ]:
        assert Path(report["artifacts"][key]).exists()

    evidence = json.loads(Path(report["artifacts"]["evidence_path"]).read_text())
    assert evidence["source_type"] == "task1_short_window_training"
    assert evidence["next_generation_constraints"] == report[
        "next_generation_constraints"
    ]
    training_log = json.loads(
        Path(report["artifacts"]["training_log_path"]).read_text()
    )
    assert training_log["runtime_seconds"] >= 0.0
    assert training_log["decision"] == evidence["decision"]
    assert training_log["loss_curve"]["val"] == report["training"]["loss_curve"]["val"]
    prediction_payload = np.load(report["artifacts"]["prediction_path"])
    assert prediction_payload["prediction"].shape == (4, 20, 32)

    import torch

    checkpoint = torch.load(
        report["artifacts"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["metadata"]["model_family"] == "conv1d_residual"
    assert checkpoint["metadata"]["context_steps"] == 10
    assert checkpoint["metadata"]["forecast_steps"] == 20
    assert checkpoint["model_state_dict"]


def test_task1_short_window_training_supports_competition_reduction_and_middle_guardrail(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers_reduced.hdf5"
    data = np.zeros((8, 600, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(
                x + 0.02 * sample + 0.01 * time_index
            )
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "reduced_run",
        dataset_key="tensor",
        train_sample_limit=4,
        val_sample_limit=2,
        context_steps=10,
        forecast_steps=100,
        time_reduction=5,
        space_reduction=2,
        model_family="conv1d_residual",
        train_steps=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=11,
        force=False,
    )

    assert report["data"]["train"]["metadata"]["time_reduction"] == 5
    assert report["data"]["train"]["metadata"]["space_reduction"] == 2
    assert report["data"]["train"]["context_shape"] == [4, 10, 16]
    assert report["data"]["val"]["target_shape"] == [2, 100, 16]
    assert report["metrics"]["early"]["evaluated"] is True
    assert report["metrics"]["early"]["future_stop"] == 47
    assert report["metrics"]["middle_guardrail"]["evaluated"] is True
    assert report["metrics"]["middle_guardrail"]["future_start"] == 47
    assert report["metrics"]["middle_guardrail"]["future_stop"] == 95


def test_task1_short_window_training_records_late_guardrail_for_full_horizon(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
        run_fno_training_optimization_loop,
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers_full_horizon.hdf5"
    data = np.zeros((10, 220, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.02 * sample + 0.04 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    single = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "single_full",
        dataset_key="tensor",
        train_sample_limit=4,
        val_sample_limit=2,
        context_steps=10,
        forecast_steps=190,
        time_reduction=1,
        space_reduction=2,
        model_family="conv1d_residual",
        train_steps=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=7,
        force=False,
    )

    assert single["metrics"]["late_guardrail"]["evaluated"] is True
    assert single["metrics"]["late_guardrail"]["future_start"] == 95
    assert single["metrics"]["late_guardrail"]["future_stop"] == 190
    assert single["metrics"]["late_guardrail"]["rel_mse"] >= 0.0
    assert single["metrics"]["late_guardrail"]["score"] >= 0.0

    plan_path = tmp_path / "candidate_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_CANDIDATES_PLANNED",
                "candidate_plan_path": str(plan_path),
                "candidate_count": 1,
                "candidate_configs": [
                    {
                        "candidate_id": "full_horizon_candidate",
                        "model_family": "conv1d_residual",
                        "modes": 4,
                        "width": 8,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 190,
                        "train_steps": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    optimization = run_fno_training_optimization_loop(
        candidate_plan_path=plan_path,
        source_path=source_path,
        output_dir=tmp_path / "optimization",
        dataset_key="tensor",
        train_sample_limit=4,
        val_sample_limit=2,
        context_steps=10,
        time_reduction=1,
        space_reduction=2,
        batch_size=2,
        learning_rate=1e-3,
        seed=7,
        force=False,
    )
    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=Path(
            optimization["artifacts"]["optimization_report_path"]
        ),
        output_path=tmp_path / "evidence.json",
        force=False,
    )

    selected = evidence["selected_candidate"]
    assert selected["late_guardrail"]["evaluated"] is True
    assert "late_guardrail" in evidence["next_generation_constraints"][
        "refine_training_horizons"
    ]


def test_task1_short_window_training_command_and_cli_are_callable(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.commands.hypothesis_lab_runtime import (
        run_task1_short_window_training_loop_command,
    )

    source_path = tmp_path / "burgers.hdf5"
    data = np.zeros((10, 40, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.cos(x + 0.04 * sample + 0.05 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    command_payload = run_task1_short_window_training_loop_command(
        source_path=str(source_path),
        output_dir=str(tmp_path / "command_run"),
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        forecast_steps=12,
        time_reduction=1,
        space_reduction=2,
        model_family="conv1d_residual",
        train_steps=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=3,
        force=False,
    )
    assert command_payload["command"] == "run-task1-short-window-training-loop"
    assert command_payload["state_hint"] == "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
    assert command_payload["metrics"]["model_family"] == "conv1d_residual"
    assert Path(command_payload["artifacts"]["evidence_path"]).exists()

    result = CliRunner().invoke(
        app,
        [
            "run-task1-short-window-training-loop",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(tmp_path / "cli_run"),
            "--dataset-key",
            "tensor",
            "--train-sample-limit",
            "6",
            "--val-sample-limit",
            "3",
            "--context-steps",
            "10",
            "--forecast-steps",
            "12",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--model-family",
            "conv1d_residual",
            "--train-steps",
            "1",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.001",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    cli_payload = json.loads(result.output)
    assert cli_payload["status"] == "ok"
    assert cli_payload["state_hint"] == "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
    assert Path(cli_payload["artifacts"]["report_path"]).exists()


def test_task1_short_window_training_loop_supports_fno_lite_family(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers_fno_lite.hdf5"
    data = np.zeros((10, 50, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.03 * sample + 0.02 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "fno_lite_run",
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        forecast_steps=12,
        time_reduction=1,
        space_reduction=2,
        model_family="fno_lite",
        train_steps=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=13,
        force=False,
    )

    assert report["model"]["model_family"] == "fno_lite"
    assert report["model"]["operator_type"] == "spectral_conv1d"
    assert report["model"]["parameter_count"] > 0
    assert report["evidence"]["model_family"] == "fno_lite"
    assert report["evidence"]["reason_codes"]
    checkpoint_path = Path(report["artifacts"]["checkpoint_path"])

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"]["model_family"] == "fno_lite"
    assert checkpoint["metadata"]["operator_type"] == "spectral_conv1d"

    result = CliRunner().invoke(
        app,
        [
            "run-task1-short-window-training-loop",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(tmp_path / "fno_lite_cli_run"),
            "--dataset-key",
            "tensor",
            "--train-sample-limit",
            "6",
            "--val-sample-limit",
            "3",
            "--context-steps",
            "10",
            "--forecast-steps",
            "12",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--model-family",
            "fno_lite",
            "--train-steps",
            "1",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.001",
            "--seed",
            "13",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["metrics"]["model_family"] == "fno_lite"
    assert Path(payload["artifacts"]["checkpoint_path"]).exists()


def test_task1_short_window_training_loop_supports_fno_ar1_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers_fno_ar1.hdf5"
    data = np.zeros((10, 50, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.02 * sample + 0.03 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "fno_ar1_run",
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        forecast_steps=12,
        time_reduction=1,
        space_reduction=2,
        model_family="fno_ar1",
        fno_modes=4,
        fno_width=12,
        fno_depth=1,
        train_steps=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=19,
        force=False,
    )

    assert report["model"]["model_family"] == "fno_ar1"
    assert report["model"]["operator_type"] == "spectral_conv1d_autoregressive"
    assert report["model"]["parameter_count"] > 0
    assert report["metrics"]["early"]["evaluated"] is True
    prediction_payload = np.load(report["artifacts"]["prediction_path"])
    assert prediction_payload["prediction"].shape == (3, 12, 16)

    import torch

    checkpoint = torch.load(
        report["artifacts"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["metadata"]["model_family"] == "fno_ar1"
    assert checkpoint["metadata"]["operator_type"] == "spectral_conv1d_autoregressive"


def test_task1_short_window_training_supports_sample_standard_normalization(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_training_loop,
    )

    source_path = tmp_path / "burgers_normalized.hdf5"
    data = np.zeros((10, 50, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        offset = 3.0 + 0.2 * sample
        scale = 2.0 + 0.1 * sample
        for time_index in range(data.shape[1]):
            data[sample, time_index] = offset + scale * np.sin(
                x + 0.04 * time_index
            )
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=tmp_path / "normalized_run",
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        forecast_steps=12,
        time_reduction=1,
        space_reduction=2,
        model_family="fno_lite",
        normalization="sample_standard",
        train_steps=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=17,
        force=False,
    )

    assert report["model"]["config"]["normalization"] == "sample_standard"
    assert report["evidence"]["normalization"] == "sample_standard"
    assert np.isfinite(report["metrics"]["early"]["rel_mse"])
    prediction_payload = np.load(report["artifacts"]["prediction_path"])
    assert float(np.mean(prediction_payload["prediction"])) > 1.0

    import torch

    checkpoint = torch.load(
        report["artifacts"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["metadata"]["normalization"] == "sample_standard"


def test_task1_short_window_family_selection_loop_compares_families_and_writes_evidence(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_family_selection_loop,
    )

    source_path = tmp_path / "burgers_family.hdf5"
    data = np.zeros((12, 50, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.cos(x + 0.05 * sample + 0.02 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_family_selection_loop(
        source_path=source_path,
        output_dir=tmp_path / "family_selection",
        dataset_key="tensor",
        model_families=["conv1d_residual", "fno_lite"],
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        forecast_steps=12,
        time_reduction=1,
        space_reduction=2,
        train_steps=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=17,
        force=False,
    )

    assert report["state_hint"] == "TASK1_SHORT_WINDOW_FAMILY_SELECTION_RAN"
    assert [item["model_family"] for item in report["family_records"]] == [
        "conv1d_residual",
        "fno_lite",
    ]
    assert report["selected_family"] in {"conv1d_residual", "fno_lite"}
    assert report["selection_reason_codes"]
    assert report["training_directive"]["action"] in {
        "continue_training",
        "change_loss_or_horizon",
        "expand_model_family",
        "abandon_candidate_family",
    }
    assert report["evidence"]["source_type"] == "task1_short_window_family_selection"
    assert report["evidence"]["family_records"][0]["loss_curve"]["val"]
    assert Path(report["artifacts"]["comparison_report_path"]).exists()
    assert Path(report["artifacts"]["comparison_evidence_path"]).exists()

    result = CliRunner().invoke(
        app,
        [
            "run-task1-short-window-family-selection-loop",
            "--source-path",
            str(source_path),
            "--output-dir",
            str(tmp_path / "family_selection_cli"),
            "--dataset-key",
            "tensor",
            "--model-family",
            "conv1d_residual",
            "--model-family",
            "fno_lite",
            "--train-sample-limit",
            "6",
            "--val-sample-limit",
            "3",
            "--context-steps",
            "10",
            "--forecast-steps",
            "12",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--train-steps",
            "1",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.001",
            "--seed",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "TASK1_SHORT_WINDOW_FAMILY_SELECTION_RAN"
    assert payload["metrics"]["selected_family"] in {"conv1d_residual", "fno_lite"}
    assert Path(payload["artifacts"]["comparison_evidence_path"]).exists()


def test_task1_short_window_family_selection_records_middle_guardrail(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        run_task1_short_window_family_selection_loop,
    )

    source_path = tmp_path / "burgers_family_middle.hdf5"
    data = np.zeros((8, 130, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.01 * sample + 0.01 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    report = run_task1_short_window_family_selection_loop(
        source_path=source_path,
        output_dir=tmp_path / "family_middle",
        dataset_key="tensor",
        model_families=["conv1d_residual", "fno_lite"],
        train_sample_limit=4,
        val_sample_limit=2,
        context_steps=10,
        forecast_steps=100,
        time_reduction=1,
        space_reduction=2,
        train_steps=1,
        batch_size=2,
        learning_rate=1e-3,
        seed=19,
        force=False,
    )

    assert all(
        item["middle_guardrail"]["evaluated"] for item in report["family_records"]
    )
    evidence = json.loads(
        Path(report["artifacts"]["comparison_evidence_path"]).read_text()
    )
    assert all(
        item["middle_guardrail"]["future_start"] == 47
        for item in evidence["family_records"]
    )


def test_fno_training_candidate_planner_uses_previous_evidence_and_avoids_repeats(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    evidence_path = tmp_path / "family_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "source_type": "task1_short_window_family_selection",
                "selected_family": "fno_lite",
                "training_directive": {
                    "action": "change_loss_or_horizon",
                    "target_family": "fno_lite",
                    "reason_codes": ["best_early_score"],
                },
                "family_records": [
                    {
                        "model_family": "fno_lite",
                        "config": {
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "mse",
                            "forecast_steps": 20,
                            "train_steps": 1,
                        },
                        "early_score": 1.0,
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "refine_training_horizons": ["shorter_early_window"],
                    "avoid_training_model_families": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[evidence_path],
        output_path=tmp_path / "plan.json",
        max_candidates=4,
        modes=[4, 8],
        widths=[16, 32],
        depths=[1, 2],
        loss_types=["mse", "early_weighted_mse", "spectral_mse"],
        forecast_steps_values=[12, 20],
        train_steps_values=[1, 2],
        force=False,
    )

    assert report["state_hint"] == "FNO_TRAINING_CANDIDATES_PLANNED"
    assert 1 <= len(report["candidate_configs"]) <= 4
    assert any(
        item["loss_type"] == "early_weighted_mse"
        for item in report["candidate_configs"]
    )
    assert all(item["model_family"] == "fno_lite" for item in report["candidate_configs"])
    assert {
        "modes",
        "width",
        "depth",
        "loss_type",
        "forecast_steps",
        "train_steps",
    }.issubset(report["candidate_configs"][0])
    assert not any(
        item["modes"] == 8
        and item["width"] == 32
        and item["depth"] == 1
        and item["loss_type"] == "mse"
        and item["forecast_steps"] == 20
        and item["train_steps"] == 1
        for item in report["candidate_configs"]
    )

    horizon_coverage = plan_fno_training_candidates(
        evidence_paths=[],
        output_path=tmp_path / "horizon_coverage_plan.json",
        max_candidates=3,
        modes=[4, 8],
        widths=[16, 32],
        depths=[1, 2],
        loss_types=["mse", "early_weighted_mse", "spectral_mse"],
        forecast_steps_values=[47, 95, 190],
        train_steps_values=[1, 2],
        force=False,
    )

    assert {
        item["forecast_steps"] for item in horizon_coverage["candidate_configs"]
    } == {47, 95, 190}

    normalization_coverage = plan_fno_training_candidates(
        evidence_paths=[],
        output_path=tmp_path / "normalization_plan.json",
        max_candidates=2,
        modes=[4],
        widths=[16],
        depths=[1],
        loss_types=["mse"],
        forecast_steps_values=[12],
        train_steps_values=[1],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert {
        item["normalization"] for item in normalization_coverage["candidate_configs"]
    } == {"none", "sample_standard"}

    protocol_v2_coverage = plan_fno_training_candidates(
        evidence_paths=[],
        output_path=tmp_path / "protocol_v2_coverage_plan.json",
        max_candidates=4,
        modes=[4],
        widths=[16],
        depths=[1],
        loss_types=["early_weighted_mse", "horizon_weighted_mse"],
        forecast_steps_values=[47, 95, 190],
        train_steps_values=[1],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    protocol_v2_configs = protocol_v2_coverage["candidate_configs"]
    assert 190 in {item["forecast_steps"] for item in protocol_v2_configs}
    assert "horizon_weighted_mse" in {
        item["loss_type"] for item in protocol_v2_configs
    }
    assert {
        item["normalization"] for item in protocol_v2_configs
    } == {"none", "sample_standard"}

    result = CliRunner().invoke(
        app,
        [
            "plan-fno-training-candidates",
            "--evidence-path",
            str(evidence_path),
            "--output-path",
            str(tmp_path / "plan_cli.json"),
            "--max-candidates",
            "3",
            "--mode",
            "4",
            "--mode",
            "8",
            "--width",
            "16",
            "--depth",
            "1",
            "--loss-type",
            "mse",
            "--loss-type",
            "early_weighted_mse",
            "--forecast-steps-value",
            "12",
            "--train-steps-value",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "FNO_TRAINING_CANDIDATES_PLANNED"
    assert payload["metrics"]["candidate_count"] >= 1


def test_fno_training_candidate_planner_expands_capacity_from_selected_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    evidence_path = tmp_path / "iter003_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "config": {
                        "modes": 8,
                        "width": 32,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 190,
                        "train_steps": 100,
                    },
                    "middle_guardrail": {"evaluated": True, "rel_mse": 0.4002},
                    "late_guardrail": {"evaluated": True, "rel_mse": 0.4498},
                },
                "candidate_records": [
                    {
                        "config": {
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 190,
                            "train_steps": 100,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "refine_training_horizons": [
                        "early_window",
                        "middle_guardrail",
                        "late_guardrail",
                    ],
                    "fno_training_configs": [
                        {
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 190,
                            "train_steps": 100,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[evidence_path],
        output_path=tmp_path / "expanded_plan.json",
        max_candidates=4,
        modes=[8],
        widths=[32],
        depths=[1],
        loss_types=["early_weighted_mse", "horizon_weighted_mse"],
        forecast_steps_values=[190],
        train_steps_values=[100],
        normalizations=["sample_standard"],
        force=False,
    )

    configs = report["candidate_configs"]
    assert configs
    assert not any(
        item["modes"] == 8
        and item["width"] == 32
        and item["depth"] == 1
        and item["loss_type"] == "early_weighted_mse"
        and item["normalization"] == "sample_standard"
        and item["forecast_steps"] == 190
        and item["train_steps"] == 100
        for item in configs
    )
    assert any(
        item["modes"] == 12
        or item["width"] == 48
        or item["depth"] == 2
        or item["train_steps"] == 150
        for item in configs
    )


def test_fno_training_candidate_planner_respects_scout_bounds_from_contract(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    evidence_path = tmp_path / "heavy_previous_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "heavy_candidate",
                    "config": {
                        "model_family": "fno_ar1",
                        "modes": 20,
                        "width": 80,
                        "depth": 5,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 95,
                        "train_steps": 500,
                    },
                    "fno_selection_score": 91.0,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_ar1",
                            "modes": 20,
                            "width": 80,
                            "depth": 5,
                            "loss_type": "horizon_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 95,
                            "train_steps": 500,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_ar1", "fno_lite"],
                    "prefer_training_objectives": [
                        "horizon_weighted_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "middle_guardrail",
                    ],
                    "modes": [16, 20],
                    "widths": [64, 80],
                    "depths": [4, 5],
                    "forecast_steps": [95],
                    "train_steps": [450, 500],
                    "normalizations": ["sample_standard"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[evidence_path],
        output_path=tmp_path / "scout_plan.json",
        max_candidates=4,
        modes=[4, 8],
        widths=[16, 32],
        depths=[1],
        loss_types=["early_weighted_mse", "horizon_weighted_mse"],
        forecast_steps_values=[12, 20, 47, 95],
        train_steps_values=[1, 2, 20],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["early_middle_source_mode"] is True
    assert report["candidate_configs"]
    for config in report["candidate_configs"]:
        assert config["modes"] in {4, 8}
        assert config["width"] in {16, 32}
        assert config["depth"] == 1
        assert config["forecast_steps"] in {12, 20, 47, 95}
        assert config["train_steps"] in {1, 2, 20}
        assert config["normalization"] in {"none", "sample_standard"}


def test_fno_training_candidate_planner_covers_capacity_axes_under_small_budget(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    evidence_path = tmp_path / "iter006_like_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_003",
                    "config": {
                        "modes": 8,
                        "width": 32,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 190,
                        "train_steps": 200,
                    },
                },
                "candidate_records": [
                    {
                        "config": {
                            "modes": 8,
                            "width": 32,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 190,
                            "train_steps": 200,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "refine_training_horizons": [
                        "middle_guardrail",
                        "late_guardrail",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[evidence_path],
        output_path=tmp_path / "capacity_plan.json",
        max_candidates=4,
        modes=[4, 8, 12],
        widths=[16, 32, 48],
        depths=[1, 2],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[47, 95, 190],
        train_steps_values=[20, 50, 150, 200],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    configs = report["candidate_configs"]
    assert len(configs) == 4
    assert any(
        item["modes"] == 12
        and item["width"] == 48
        and item["depth"] == 2
        and item["train_steps"] == 250
        and item["forecast_steps"] == 190
        for item in configs
    )


def test_fno_training_candidate_planner_covers_capacity_axis_in_early_middle_mode(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    evidence_path = tmp_path / "positive_graft_like_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "graft_evidence": {
                    "best_task_prediction_score_delta": 0.1,
                    "late_non_regression_passed": True,
                },
                "selected_candidate": {
                    "candidate_id": "small_positive",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 4,
                        "width": 16,
                        "depth": 1,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 51,
                    },
                    "fno_selection_score": 58.0,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "horizon_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 51,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite", "fno_ar1"],
                    "prefer_training_objectives": [
                        "horizon_weighted_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                    "modes": [4, 8, 12, 16],
                    "widths": [16, 32, 48, 64],
                    "depths": [1, 2, 3, 4],
                    "forecast_steps": [12, 20],
                    "train_steps": [1, 2, 20, 50, 100, 150],
                    "normalizations": ["sample_standard"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[evidence_path],
        output_path=tmp_path / "early_middle_capacity_plan.json",
        max_candidates=6,
        modes=[4, 8, 12, 16],
        widths=[16, 32, 48, 64],
        depths=[1, 2, 3, 4],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[12, 20],
        train_steps_values=[1, 2, 20, 50, 100, 150],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["early_middle_source_mode"] is True
    configs = report["candidate_configs"]
    assert len(configs) == 6
    assert any(
        item["modes"] == 16
        and item["width"] == 64
        and item["depth"] == 4
        and item["train_steps"] == 150
        for item in configs
    )


def test_fno_training_candidate_planner_prioritizes_best_graft_anchor_over_weaker_spectral_update(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    best_graft_path = tmp_path / "iter034_best_graft.json"
    best_graft_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "graft_evidence": {
                    "best_task_prediction_score_delta": 0.1403,
                    "late_non_regression_passed": True,
                    "recommended_training_objectives": [
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                },
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 150,
                    },
                    "fno_selection_score": 60.02,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "horizon_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 150,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": [
                        "horizon_weighted_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    weaker_spectral_path = tmp_path / "iter036_weaker_spectral.json"
    weaker_spectral_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "graft_evidence": {
                    "best_task_prediction_score_delta": 0.0904,
                    "late_non_regression_passed": True,
                    "recommended_training_objectives": [
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                },
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "spectral_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 100,
                    },
                    "fno_selection_score": 59.61,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "spectral_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 100,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": [
                        "spectral_mse",
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[weaker_spectral_path, best_graft_path],
        output_path=tmp_path / "best_graft_anchor_plan.json",
        max_candidates=6,
        modes=[4, 8, 12, 16],
        widths=[16, 32, 48, 64],
        depths=[1, 2, 3, 4],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[12, 20],
        train_steps_values=[1, 2, 20, 50, 100, 150],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["planning_preferences"]["loss_types"][:2] == [
        "early_weighted_mse",
        "horizon_weighted_mse",
    ]
    assert report["planning_preferences"]["best_selected_config"]["loss_type"] == (
        "horizon_weighted_mse"
    )
    assert report["candidate_configs"][0]["loss_type"] in {
        "early_weighted_mse",
        "horizon_weighted_mse",
    }
    assert report["candidate_configs"][0]["loss_type"] != "spectral_mse"
    assert report["candidate_configs"][0]["modes"] == 16
    assert report["candidate_configs"][0]["width"] == 64
    assert report["candidate_configs"][0]["depth"] == 4
    assert report["candidate_configs"][0]["train_steps"] == 150
    assert any(
        item["loss_type"] == "early_weighted_mse"
        for item in report["candidate_configs"][:3]
    )


def test_fno_training_candidate_planner_adds_conv1d_probe_for_plateau_regression(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    best_path = tmp_path / "best_plateau.json"
    best_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 150,
                    },
                    "fno_selection_score": 60.1,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "horizon_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 150,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": [
                        "fno_lite",
                        "fno_ar1",
                        "conv1d_residual",
                    ],
                    "prefer_training_objectives": [
                        "horizon_weighted_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "new_early_dynamics_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    regressed_path = tmp_path / "regressed_plateau.json"
    regressed_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "spectral_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 100,
                    },
                    "fno_selection_score": 59.6,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "spectral_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 100,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": [
                        "fno_lite",
                        "fno_ar1",
                        "conv1d_residual",
                    ],
                    "prefer_training_objectives": [
                        "spectral_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "new_early_dynamics_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[best_path, regressed_path],
        output_path=tmp_path / "conv1d_probe_plan.json",
        max_candidates=6,
        modes=[4, 8, 12, 16],
        widths=[16, 32, 48, 64],
        depths=[1, 2, 3, 4],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[12, 20],
        train_steps_values=[1, 2, 20, 50, 100, 150],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["planning_preferences"]["regression_mode"] is True
    assert report["planning_preferences"]["requires_new_dynamics_source"] is True
    assert report["candidate_configs"][0]["model_family"] in {
        "fno_ar1",
        "conv1d_residual",
    }
    assert report["candidate_configs"][1]["model_family"] in {
        "fno_ar1",
        "conv1d_residual",
    }
    assert any(
        item["model_family"] == "conv1d_residual"
        and "local_residual_convolution_source" in item["rationale"]
        for item in report["candidate_configs"]
    )
    assert any(
        item["model_family"] == "fno_ar1"
        and "autoregressive_rollout_source" in item["rationale"]
        for item in report["candidate_configs"]
    )


def test_fno_training_candidate_planner_preserves_middle_forecasts_after_short_graft_plateau(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    best_path = tmp_path / "best_short_graft.json"
    best_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "graft_evidence": {
                    "best_task_prediction_score_delta": 0.24,
                    "late_non_regression_passed": True,
                    "recommended_refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                },
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 500,
                    },
                    "fno_selection_score": 60.1,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "horizon_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 500,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": [
                        "fno_lite",
                        "fno_ar1",
                        "conv1d_residual",
                    ],
                    "prefer_training_objectives": [
                        "horizon_weighted_mse",
                        "early_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    regressed_path = tmp_path / "regressed_short_graft.json"
    regressed_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 16,
                        "width": 64,
                        "depth": 4,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 150,
                    },
                    "fno_selection_score": 59.0,
                },
                "candidate_records": [
                    {
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 16,
                            "width": 64,
                            "depth": 4,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 12,
                            "train_steps": 150,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "training_model_families": [
                        "fno_lite",
                        "fno_ar1",
                        "conv1d_residual",
                    ],
                    "prefer_training_objectives": [
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "early_window",
                        "early_middle_graft_source",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[best_path, regressed_path],
        output_path=tmp_path / "middle_forecast_plan.json",
        max_candidates=6,
        modes=[4, 8, 12, 16],
        widths=[16, 32, 48, 64],
        depths=[1, 2, 3, 4],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[12, 20, 47, 95],
        train_steps_values=[20, 50, 100, 150, 500],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["planning_preferences"]["regression_mode"] is True
    assert report["planning_preferences"]["requires_middle_guardrail_source"] is True
    forecast_steps = {item["forecast_steps"] for item in report["candidate_configs"]}
    assert forecast_steps & {47, 95}
    assert any(
        item["model_family"] in {"fno_ar1", "conv1d_residual"}
        and item["forecast_steps"] >= 47
        for item in report["candidate_configs"]
    )


def test_fno_training_candidate_planner_caps_capacity_after_guardrail_regression(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
    )

    best_path = tmp_path / "iter008_best.json"
    best_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "config": {
                        "modes": 12,
                        "width": 48,
                        "depth": 2,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 190,
                        "train_steps": 300,
                    },
                    "fno_selection_score": 95.67,
                },
                "candidate_records": [
                    {
                        "config": {
                            "modes": 12,
                            "width": 48,
                            "depth": 2,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 190,
                            "train_steps": 300,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "refine_training_horizons": ["middle_guardrail", "late_guardrail"],
                },
            }
        ),
        encoding="utf-8",
    )
    regressed_path = tmp_path / "iter009_regressed.json"
    regressed_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "config": {
                        "modes": 16,
                        "width": 64,
                        "depth": 3,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 190,
                        "train_steps": 350,
                    },
                    "fno_selection_score": 95.59,
                },
                "candidate_records": [
                    {
                        "config": {
                            "modes": 16,
                            "width": 64,
                            "depth": 3,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                            "forecast_steps": 190,
                            "train_steps": 350,
                        }
                    }
                ],
                "next_generation_constraints": {
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "refine_training_horizons": ["middle_guardrail", "late_guardrail"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = plan_fno_training_candidates(
        evidence_paths=[best_path, regressed_path],
        output_path=tmp_path / "regression_plan.json",
        max_candidates=4,
        modes=[4, 8, 12, 16],
        widths=[16, 32, 48, 64],
        depths=[1, 2, 3],
        loss_types=["early_weighted_mse", "horizon_weighted_mse", "spectral_mse"],
        forecast_steps_values=[47, 95, 190],
        train_steps_values=[20, 50, 300, 350],
        normalizations=["none", "sample_standard"],
        force=False,
    )

    assert report["planning_preferences"]["regression_mode"] is True
    assert report["planning_preferences"]["best_selected_score"] == 95.67
    assert all(item["modes"] <= 12 for item in report["candidate_configs"])
    assert all(item["width"] <= 48 for item in report["candidate_configs"])
    assert all(item["depth"] <= 2 for item in report["candidate_configs"])
    assert all(item["train_steps"] <= 300 for item in report["candidate_configs"])
    assert any(
        item["modes"] == 12
        and item["width"] == 48
        and item["depth"] == 2
        and item["train_steps"] == 300
        and item["loss_type"] == "horizon_weighted_mse"
        and "regression_repair_anchor_capacity" in item["rationale"]
        for item in report["candidate_configs"]
    )
    assert any(
        item["modes"] == 12
        and item["width"] == 48
        and item["depth"] == 2
        and item["train_steps"] == 300
        and item["loss_type"] == "spectral_mse"
        and "regression_repair_anchor_capacity" in item["rationale"]
        for item in report["candidate_configs"]
    )


def test_fno_training_optimization_loop_runs_candidates_and_records_evidence(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
        record_fno_training_optimization_evidence,
        run_fno_training_optimization_loop,
    )

    source_path = tmp_path / "burgers_fno_optimization.hdf5"
    data = np.zeros((10, 55, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.04 * sample + 0.02 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    plan = plan_fno_training_candidates(
        evidence_paths=[],
        output_path=tmp_path / "plan.json",
        max_candidates=2,
        modes=[4],
        widths=[12, 16],
        depths=[1],
        loss_types=["mse", "early_weighted_mse"],
        forecast_steps_values=[12],
        train_steps_values=[1],
        force=False,
    )
    report = run_fno_training_optimization_loop(
        candidate_plan_path=tmp_path / "plan.json",
        source_path=source_path,
        output_dir=tmp_path / "optimization",
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        time_reduction=1,
        space_reduction=2,
        batch_size=3,
        learning_rate=1e-3,
        seed=23,
        force=False,
    )

    assert report["state_hint"] == "FNO_TRAINING_OPTIMIZATION_LOOP_RAN"
    assert len(report["candidate_records"]) == len(plan["candidate_configs"])
    assert report["selected_candidate"]["model_family"] == "fno_lite"
    assert report["training_directive"]["action"] in {
        "continue_current_fno_family",
        "change_loss",
        "change_horizon",
        "change_architecture",
        "fallback_to_conv1d",
        "expand_new_model_family",
    }
    first_record = report["candidate_records"][0]
    assert first_record["config"]["loss_type"] in {"mse", "early_weighted_mse"}
    assert isinstance(first_record["score"], float)
    assert first_record["score"] == first_record["fno_selection_score"]
    assert isinstance(report["selected_candidate"]["score"], float)
    assert report["selected_candidate"]["score"] == report["selected_candidate"][
        "fno_selection_score"
    ]
    assert first_record["loss_curve"]["val"]
    assert Path(first_record["artifacts"]["checkpoint_path"]).exists()
    assert Path(report["artifacts"]["optimization_report_path"]).exists()

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=tmp_path
        / "optimization"
        / "fno_training_optimization_report.json",
        output_path=tmp_path / "fno_optimization_evidence.json",
        force=False,
    )
    assert evidence["state_hint"] == "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED"
    assert evidence["source_type"] == "fno_training_optimization"
    assert evidence["selected_candidate"]["candidate_id"]
    assert evidence["next_generation_constraints"]["fno_training_configs"]

    record_result = CliRunner().invoke(
        app,
        [
            "record-fno-training-optimization-evidence",
            "--optimization-report-path",
            str(tmp_path / "optimization" / "fno_training_optimization_report.json"),
            "--output-path",
            str(tmp_path / "fno_optimization_evidence_cli.json"),
        ],
    )
    assert record_result.exit_code == 0, record_result.output
    record_payload = json.loads(record_result.output)
    assert (
        record_payload["state_hint"]
        == "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED"
    )
    assert Path(
        record_payload["artifacts"]["fno_training_optimization_evidence_path"]
    ).exists()

    result = CliRunner().invoke(
        app,
        [
            "run-fno-training-optimization-loop",
            "--candidate-plan-path",
            str(tmp_path / "plan.json"),
            "--source-path",
            str(source_path),
            "--output-dir",
            str(tmp_path / "optimization_cli"),
            "--dataset-key",
            "tensor",
            "--train-sample-limit",
            "6",
            "--val-sample-limit",
            "3",
            "--context-steps",
            "10",
            "--time-reduction",
            "1",
            "--space-reduction",
            "2",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.001",
            "--seed",
            "23",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "FNO_TRAINING_OPTIMIZATION_LOOP_RAN"
    assert Path(payload["artifacts"]["optimization_report_path"]).exists()


def test_fno_training_evidence_records_graft_report_for_next_constraints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
    )

    optimization_report_path = tmp_path / "optimization_report.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"placeholder")
    candidate_record = {
        "candidate_id": "fno_train_candidate_001",
        "model_family": "fno_lite",
        "operator_type": "spectral_neural_operator_1d",
        "parameter_count": 12,
        "runtime_seconds": 1.0,
        "train_steps": 2,
        "loss_curve": {"train": [1.0], "val": [0.5]},
        "final_train_loss": 1.0,
        "final_val_loss": 0.5,
        "early_rel_mse": 0.2,
        "early_score": 1.8,
        "middle_guardrail": {"evaluated": True, "score": 10.0, "rel_mse": 0.2},
        "late_guardrail": {"evaluated": False},
        "decision": "continue_training",
        "reason_codes": ["candidate"],
        "artifacts": {"checkpoint_path": str(checkpoint_path)},
        "config": {
            "modes": 8,
            "width": 16,
            "depth": 1,
            "loss_type": "early_weighted_mse",
            "normalization": "none",
            "forecast_steps": 47,
            "train_steps": 2,
        },
    }
    optimization_report_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
                "candidate_records": [candidate_record],
                "selected_candidate": candidate_record,
                "training_directive": {
                    "action": "continue_current_fno_family",
                    "reason_codes": ["test"],
                },
                "artifacts": {
                    "optimization_report_path": str(optimization_report_path),
                },
            }
        )
    )
    graft_report_path = tmp_path / "graft_report.json"
    graft_report_path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_FNO_GRAFT_EVALUATOR_RAN",
                "baseline_score": {"task_prediction_score": 34.0},
                "source_score": {"task_prediction_score": 12.0},
                "best_candidate": {
                    "candidate_id": "fno_graft_w10_57_a0.25",
                    "task_prediction_score": 34.4,
                    "task_prediction_score_delta": 0.4,
                    "late_non_regression_passed": True,
                    "late_score_delta": 0.0,
                    "window_start": 10,
                    "window_stop": 57,
                    "blend_weight": 0.25,
                },
                "candidate_records": [
                    {
                        "candidate_id": "fno_graft_w10_57_a0.25",
                        "task_prediction_score_delta": 0.4,
                    },
                    {
                        "candidate_id": "fno_graft_w57_105_a0.25",
                        "task_prediction_score_delta": -1.0,
                    },
                ],
            }
        )
    )

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=optimization_report_path,
        output_path=tmp_path / "evidence.json",
        graft_report_path=graft_report_path,
    )

    assert evidence["graft_report_path"] == str(graft_report_path)
    assert evidence["graft_evidence"]["best_task_prediction_score_delta"] == 0.4
    assert evidence["graft_evidence"]["positive_candidate_count"] == 1
    constraints = evidence["next_generation_constraints"]
    assert constraints["graft_objective"]["required"] is True
    assert constraints["graft_objective"]["last_best_window"] == [10, 57]
    assert "early_middle_graft_source" in constraints["refine_training_horizons"]


def test_fno_evidence_opens_new_dynamics_family_on_weak_positive_graft_plateau(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
    )

    optimization_report_path = tmp_path / "optimization_report.json"
    candidate_record = {
        "candidate_id": "weak_positive_fno",
        "fno_selection_score": 60.0,
        "final_val_loss": 0.2,
        "early_rel_mse": 0.1,
        "early_score": 10.0,
        "middle_guardrail": {"evaluated": False},
        "late_guardrail": {"evaluated": False},
        "decision": "continue_training",
        "reason_codes": ["continue_training"],
        "config": {
            "model_family": "fno_lite",
            "modes": 16,
            "width": 64,
            "depth": 4,
            "loss_type": "horizon_weighted_mse",
            "normalization": "sample_standard",
            "forecast_steps": 12,
            "train_steps": 400,
        },
    }
    optimization_report_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
                "candidate_records": [candidate_record],
                "selected_candidate": candidate_record,
                "training_directive": {
                    "action": "continue_current_fno_family",
                    "reason_codes": ["early_signal_learned"],
                },
            }
        ),
        encoding="utf-8",
    )
    graft_report_path = tmp_path / "graft_report.json"
    graft_report_path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_FNO_GRAFT_EVALUATOR_RAN",
                "baseline_score": {"task_prediction_score": 34.0},
                "source_score": {"task_prediction_score": 12.0},
                "best_candidate": {
                    "candidate_id": "weak_graft",
                    "task_prediction_score": 34.14,
                    "task_prediction_score_delta": 0.14,
                    "late_non_regression_passed": True,
                    "late_score_delta": 0.0,
                    "window_start": 10,
                    "window_stop": 20,
                    "blend_weight": 0.75,
                },
                "candidate_records": [
                    {
                        "candidate_id": "weak_graft",
                        "task_prediction_score_delta": 0.14,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=optimization_report_path,
        output_path=tmp_path / "evidence.json",
        graft_report_path=graft_report_path,
        force=False,
    )

    constraints = evidence["next_generation_constraints"]
    assert "fno_ar1" in constraints["training_model_families"]
    assert "conv1d_residual" in constraints["training_model_families"]
    assert "new_early_dynamics_source" in constraints["refine_training_horizons"]
    assert "rollout_consistency" in constraints["prefer_training_objectives"]
    assert (
        constraints["graft_objective"]["plateau_requires_new_dynamics_source"]
        is True
    )


def test_fno_training_accepts_horizon_weighted_mse(tmp_path: Path) -> None:
    from pde_agent.core.task1_short_window_training import (
        plan_fno_training_candidates,
        run_fno_training_optimization_loop,
    )

    source_path = tmp_path / "burgers_horizon_loss.hdf5"
    data = np.zeros((10, 80, 32), dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float32)
    for sample in range(data.shape[0]):
        for time_index in range(data.shape[1]):
            data[sample, time_index] = np.sin(x + 0.03 * sample + 0.02 * time_index)
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    plan = plan_fno_training_candidates(
        evidence_paths=[],
        output_path=tmp_path / "plan.json",
        max_candidates=1,
        modes=[4],
        widths=[12],
        depths=[1],
        loss_types=["horizon_weighted_mse"],
        forecast_steps_values=[20],
        train_steps_values=[1],
        force=False,
    )
    assert plan["candidate_configs"][0]["loss_type"] == "horizon_weighted_mse"

    report = run_fno_training_optimization_loop(
        candidate_plan_path=tmp_path / "plan.json",
        source_path=source_path,
        output_dir=tmp_path / "optimization",
        dataset_key="tensor",
        train_sample_limit=6,
        val_sample_limit=3,
        context_steps=10,
        time_reduction=1,
        space_reduction=2,
        batch_size=3,
        learning_rate=1e-3,
        seed=29,
        force=False,
    )

    record = report["candidate_records"][0]
    assert record["config"]["loss_type"] == "horizon_weighted_mse"
    assert record["config"]["model_family"] == "fno_lite"
    assert record["loss_curve"]["train"]
    assert np.isfinite(record["final_val_loss"])


def test_fno_optimization_constraints_retain_pool_middle_and_late_guardrails(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
    )

    report_path = tmp_path / "optimization_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
                "selected_candidate": {
                    "candidate_id": "early_only",
                    "early_rel_mse": 1.2,
                    "early_score": 0.1,
                    "final_val_loss": 0.2,
                    "runtime_seconds": 0.01,
                    "config": {
                        "modes": 4,
                        "width": 16,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 47,
                        "train_steps": 1,
                    },
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                },
                "candidate_records": [
                    {
                        "candidate_id": "early_only",
                        "config": {
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "forecast_steps": 47,
                            "train_steps": 1,
                        },
                        "middle_guardrail": {"evaluated": False},
                        "late_guardrail": {"evaluated": False},
                    },
                    {
                        "candidate_id": "middle_probe",
                        "config": {
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "forecast_steps": 95,
                            "train_steps": 1,
                        },
                        "middle_guardrail": {"evaluated": True},
                        "late_guardrail": {"evaluated": False},
                    },
                    {
                        "candidate_id": "late_probe",
                        "config": {
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "forecast_steps": 190,
                            "train_steps": 1,
                        },
                        "middle_guardrail": {"evaluated": True},
                        "late_guardrail": {"evaluated": True},
                    },
                ],
                "training_directive": {
                    "action": "change_horizon",
                    "selected_candidate_id": "early_only",
                    "target_family": "fno_lite",
                    "reason_codes": ["best_early_score", "change_horizon"],
                    "next_command": "plan-fno-training-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=report_path,
        output_path=tmp_path / "evidence.json",
        force=False,
    )

    horizons = evidence["next_generation_constraints"]["refine_training_horizons"]
    assert "middle_guardrail" in horizons
    assert "late_guardrail" in horizons


def test_fno_selection_prefers_multi_objective_guardrail_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
    )

    report_path = tmp_path / "optimization_report.json"
    shared_config = {
        "modes": 4,
        "width": 16,
        "depth": 1,
        "loss_type": "early_weighted_mse",
        "normalization": "none",
        "train_steps": 1,
    }
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
                "selected_candidate": {
                    "candidate_id": "early_only",
                    "config": {**shared_config, "forecast_steps": 47},
                    "early_score": 90.0,
                    "early_rel_mse": 0.01,
                    "final_val_loss": 0.1,
                    "runtime_seconds": 0.01,
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                    "selection_reason_codes": ["best_early_score"],
                },
                "candidate_records": [
                    {
                        "candidate_id": "early_only",
                        "config": {**shared_config, "forecast_steps": 47},
                        "early_score": 90.0,
                        "early_rel_mse": 0.01,
                        "final_val_loss": 0.1,
                        "runtime_seconds": 0.01,
                        "middle_guardrail": {"evaluated": False},
                        "late_guardrail": {"evaluated": False},
                    },
                    {
                        "candidate_id": "balanced_horizon",
                        "config": {**shared_config, "forecast_steps": 190},
                        "early_score": 85.0,
                        "early_rel_mse": 0.02,
                        "final_val_loss": 0.2,
                        "runtime_seconds": 0.02,
                        "middle_guardrail": {"evaluated": True, "score": 80.0},
                        "late_guardrail": {"evaluated": True, "score": 70.0},
                    },
                ],
                "training_directive": {
                    "action": "change_horizon",
                    "selected_candidate_id": "early_only",
                    "target_family": "fno_lite",
                    "reason_codes": ["best_early_score"],
                    "next_command": "plan-fno-training-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=report_path,
        output_path=tmp_path / "evidence.json",
        force=False,
    )

    selected = evidence["selected_candidate"]
    assert selected["candidate_id"] == "balanced_horizon"
    assert selected["fno_selection_score"] > 80.0
    assert selected["selection_components"]["middle"]["evaluated"] is True
    assert selected["selection_components"]["late"]["evaluated"] is True
    assert "multi_objective_selection" in selected["selection_reason_codes"]


def test_fno_selection_prefers_early_middle_candidate_over_late_only_guardrail(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        record_fno_training_optimization_evidence,
    )

    report_path = tmp_path / "optimization_report.json"
    shared_config = {
        "modes": 8,
        "width": 16,
        "depth": 1,
        "normalization": "none",
        "train_steps": 20,
    }
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_TRAINING_OPTIMIZATION_LOOP_RAN",
                "selected_candidate": {
                    "candidate_id": "late_guardrail_candidate",
                    "config": {
                        **shared_config,
                        "forecast_steps": 190,
                        "loss_type": "spectral_mse",
                    },
                    "early_score": 0.000001,
                    "early_rel_mse": 0.95,
                    "final_val_loss": 0.2,
                    "runtime_seconds": 0.02,
                    "middle_guardrail": {
                        "evaluated": True,
                        "score": 0.00000001,
                        "rel_mse": 2.28,
                    },
                    "late_guardrail": {
                        "evaluated": True,
                        "score": 4.0,
                        "rel_mse": 2.38,
                    },
                },
                "candidate_records": [
                    {
                        "candidate_id": "early_middle_candidate",
                        "config": {
                            **shared_config,
                            "forecast_steps": 95,
                            "loss_type": "early_weighted_mse",
                            "normalization": "sample_standard",
                        },
                        "early_score": 0.00034,
                        "early_rel_mse": 0.63,
                        "final_val_loss": 0.3,
                        "runtime_seconds": 0.03,
                        "middle_guardrail": {
                            "evaluated": True,
                            "score": 0.00000045,
                            "rel_mse": 1.92,
                        },
                        "late_guardrail": {"evaluated": False},
                    },
                    {
                        "candidate_id": "late_guardrail_candidate",
                        "config": {
                            **shared_config,
                            "forecast_steps": 190,
                            "loss_type": "spectral_mse",
                        },
                        "early_score": 0.000001,
                        "early_rel_mse": 0.95,
                        "final_val_loss": 0.2,
                        "runtime_seconds": 0.02,
                        "middle_guardrail": {
                            "evaluated": True,
                            "score": 0.00000001,
                            "rel_mse": 2.28,
                        },
                        "late_guardrail": {
                            "evaluated": True,
                            "score": 4.0,
                            "rel_mse": 2.38,
                        },
                    },
                ],
                "training_directive": {
                    "action": "change_horizon",
                    "selected_candidate_id": "late_guardrail_candidate",
                    "target_family": "fno_lite",
                    "reason_codes": ["late_guardrail_available"],
                    "next_command": "plan-fno-training-candidates",
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = record_fno_training_optimization_evidence(
        optimization_report_path=report_path,
        output_path=tmp_path / "evidence.json",
        force=False,
    )

    selected = evidence["selected_candidate"]
    assert selected["candidate_id"] == "early_middle_candidate"
    assert "early_middle_objective_selected" in selected["selection_reason_codes"]
    assert "late_guardrail_available" not in selected["selection_reason_codes"]


def test_hypothesis_designer_consumes_fno_training_optimization_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": "early plateau"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "fno_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "training_directive": {
                    "action": "change_loss",
                    "target_family": "fno_lite",
                    "reason_codes": ["early_rel_mse_high"],
                },
                "selected_candidate": {
                    "candidate_id": "fno_candidate_001",
                    "model_family": "fno_lite",
                    "config": {
                        "modes": 4,
                        "width": 16,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 12,
                        "train_steps": 1,
                    },
                },
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": ["spectral_mse"],
                    "refine_training_horizons": ["early_window"],
                    "avoid_training_model_families": [],
                    "fno_training_configs": [
                        {
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[evidence_path],
        output_path=None,
    )

    assert report["diagnosis"]["training_evidence_count"] == 1
    assert any(
        item["source_type"] == "fno_training_optimization"
        for item in report["diagnosis"]["evidence_summaries"]
    )
    by_family = {
        item["mechanism_family"]: item for item in report["mechanism_hypotheses"]
    }
    assert by_family["operator_structure"]["next_generation_action"] == "prefer"
    assert (
        "fno_training_optimization"
        in by_family["operator_structure"]["source_evidence_types"]
    )


def test_fno_training_harness_contract_exposes_optional_actions_and_budget(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_harness_contract,
    )

    output_path = tmp_path / "contract.json"
    report = build_fno_training_harness_contract(
        output_path=output_path,
        workspace_dir=tmp_path / "run",
        source_path=tmp_path / "burgers.hdf5",
        previous_evidence_paths=[tmp_path / "previous_evidence.json"],
        dataset_key="tensor",
        train_sample_limit=16,
        val_sample_limit=8,
        batch_size=4,
        max_candidates=3,
        force=False,
    )

    assert output_path.exists()
    assert report["state_hint"] == "FNO_TRAINING_HARNESS_CONTRACT_BUILT"
    assert report["data_contract"]["source_path"] == str(tmp_path / "burgers.hdf5")
    assert report["budget_contract"]["train_sample_limit"] == 16
    assert report["budget_contract"]["val_sample_limit"] == 8
    assert report["budget_contract"]["batch_size"] == 4
    actions = {item["action_id"]: item for item in report["available_actions"]}
    assert list(actions) == [
        "plan_fno_training_candidates",
        "run_fno_training_optimization",
        "record_fno_training_optimization_evidence",
        "build_hypothesis_algorithm_design",
        "validate_hypothesis_algorithm_design",
        "stop_and_report",
    ]
    assert "--evidence-path" in actions["build_hypothesis_algorithm_design"]["command"]
    assert "--train-sample-limit 16" in actions["run_fno_training_optimization"][
        "command"
    ]
    assert report["recommended_policy"]["not_a_fixed_pipeline"] is True
    assert report["next_command"].startswith("Claude SDK agent should")
    assert actions["stop_and_report"]["state_hint"] == (
        "CLAUDE_AGENT_SDK_REPORTED_DECISION"
    )


def test_fno_training_harness_contract_filters_forecasts_by_real_source_horizon(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_harness_contract,
    )

    source_path = tmp_path / "burgers.hdf5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 201, 16), dtype=np.float32))

    report = build_fno_training_harness_contract(
        output_path=tmp_path / "contract.json",
        workspace_dir=tmp_path / "run",
        source_path=source_path,
        forecast_steps_values=[12, 20, 47, 95],
        context_steps=10,
        time_reduction=5,
        force=False,
    )

    assert report["candidate_space"]["forecast_steps_values"] == [12, 20]
    guard = report["candidate_space_guards"]["source_horizon"]
    assert guard["raw_time_steps"] == 201
    assert guard["reduced_time_steps"] == 41
    assert guard["max_forecast_steps"] == 31
    assert "--forecast-steps-value 47" not in report["available_actions"][0]["command"]
    assert "--forecast-steps-value 95" not in report["available_actions"][0]["command"]


def test_fno_training_harness_contract_repairs_sample_budget_for_small_source(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_harness_contract,
    )

    source_path = tmp_path / "small_burgers.hdf5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((100, 200, 256), dtype=np.float32))

    report = build_fno_training_harness_contract(
        output_path=tmp_path / "contract.json",
        workspace_dir=tmp_path / "run",
        source_path=source_path,
        train_sample_limit=192,
        val_sample_limit=32,
        batch_size=128,
        force=False,
    )

    repair = report["budget_contract"]["training_contract_repair"]
    assert repair["applied"] is True
    assert repair["reason"] == "train_val_sample_limits_exceed_source_sample_count"
    assert repair["source_sample_count"] == 100
    assert report["budget_contract"]["train_sample_limit"] == 68
    assert report["budget_contract"]["val_sample_limit"] == 32
    assert report["budget_contract"]["batch_size"] == 68
    optimize_action = {
        item["action_id"]: item for item in report["available_actions"]
    }["run_fno_training_optimization"]
    assert "--train-sample-limit 68" in optimize_action["command"]
    assert "--val-sample-limit 32" in optimize_action["command"]
    assert "--batch-size 68" in optimize_action["command"]


def test_fno_training_iteration_contract_uses_previous_evidence_for_next_agent_run(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    previous_evidence_path = tmp_path / "previous_evidence.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_003",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 4,
                        "width": 16,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "normalization": "none",
                        "forecast_steps": 190,
                        "train_steps": 1,
                    },
                    "selection_components": {
                        "early": 0.5,
                        "middle": 0.4,
                        "late": 0.2,
                    },
                    "fno_selection_score": 0.43,
                },
                "candidate_records": [
                    {
                        "candidate_id": "fno_train_candidate_003",
                        "config": {
                            "model_family": "fno_lite",
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "normalization": "none",
                            "forecast_steps": 190,
                            "train_steps": 1,
                        },
                    }
                ],
                "next_generation_constraints": {
                    "prefer_training_objectives": [
                        "early_weighted_mse",
                        "horizon_weighted_mse",
                    ],
                    "refine_training_horizons": [
                        "shorter_early_window",
                        "middle_guardrail",
                        "late_guardrail",
                    ],
                    "fno_training_configs": [
                        {
                            "modes": 4,
                            "width": 16,
                            "depth": 1,
                            "loss_type": "early_weighted_mse",
                            "normalization": "none",
                            "forecast_steps": 190,
                            "train_steps": 1,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=previous_evidence_path,
        output_dir=tmp_path / "agent_iter",
        workspace_dir=tmp_path,
        source_path=tmp_path / "burgers.hdf5",
        train_sample_limit=32,
        val_sample_limit=16,
        max_candidates=4,
        force=False,
    )

    assert report["state_hint"] == "FNO_TRAINING_ITERATION_CONTRACT_BUILT"
    assert report["previous_evidence_path"] == str(previous_evidence_path)
    assert report["agent_contract"]["autonomy_policy"]["agent_selects_candidates"] is True
    assert report["agent_contract"]["autonomy_policy"]["supervisor_must_not_pin_parameters"] is True
    assert report["agent_contract"]["required_agent_behaviors"] == [
        "read_previous_evidence",
        "build_or_read_harness_contract",
        "plan_candidates_from_evidence",
        "run_bounded_optimization",
        "record_evidence",
        "build_and_validate_hypothesis_design",
        "summarize_next_constraints",
    ]
    assert report["agent_contract"]["previous_selected_candidate"]["candidate_id"] == (
        "fno_train_candidate_003"
    )
    harness = json.loads(Path(report["artifacts"]["harness_contract_path"]).read_text())
    active_contract_text = json.dumps(report, sort_keys=True).lower()
    active_harness_text = json.dumps(harness, sort_keys=True).lower()
    active_prompt_text = Path(report["artifacts"]["prompt_path"]).read_text(
        encoding="utf-8"
    ).lower()
    assert "hermes" not in active_contract_text
    assert "hermes" not in active_harness_text
    assert "hermes" not in active_prompt_text
    assert "heartbeat" not in active_contract_text
    assert "heartbeat" not in active_harness_text
    assert "heartbeat" not in active_prompt_text
    assert harness["previous_evidence_paths"] == [str(previous_evidence_path)]
    assert harness["candidate_space"]["modes"] == [4, 8]
    assert harness["candidate_space"]["widths"] == [16, 32]
    assert harness["candidate_space"]["depths"] == [1, 2]
    assert harness["candidate_space"]["forecast_steps_values"] == [12, 20, 47, 95, 190]
    assert harness["candidate_space"]["train_steps_values"] == [1, 2, 20, 51]
    assert harness["candidate_space"]["normalizations"] == ["none", "sample_standard"]
    assert harness["budget_contract"]["max_candidates"] == 4
    assert "--train-steps-value 20" in harness["available_actions"][0]["command"]
    prompt = Path(report["artifacts"]["prompt_path"]).read_text(encoding="utf-8")
    assert "Do not hand-pick hyperparameters outside the contract" in prompt
    assert "Do not Read an entire evidence file first" in prompt
    assert "Use Grep for selected_candidate" in prompt
    assert "fno_training_optimization_evidence.json" in prompt
    assert "run-claude-agent-sdk-runtime" == report["next_command"]


def test_fno_training_iteration_contract_expands_candidate_space_from_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    previous_evidence_path = tmp_path / "previous_evidence_expanded.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 8,
                        "width": 32,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 190,
                        "train_steps": 150,
                    },
                    "selection_components": {
                        "early": 0.3,
                        "middle": 0.2,
                        "late": 0.1,
                    },
                    "fno_selection_score": 93.7,
                    "next_generation_constraints": {
                        "prefer_training_objectives": [
                            "early_weighted_mse",
                            "horizon_weighted_mse",
                        ],
                        "refine_training_horizons": [
                            "middle_guardrail",
                            "late_guardrail",
                        ],
                        "modes": [8, 12],
                        "widths": [32, 48],
                        "depths": [2, 3],
                        "train_steps": [150, 200],
                        "forecast_steps": [190],
                        "normalizations": ["sample_standard"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=previous_evidence_path,
        output_dir=tmp_path / "agent_iter_expanded",
        workspace_dir=tmp_path,
        source_path=tmp_path / "burgers.hdf5",
        train_sample_limit=32,
        val_sample_limit=16,
        max_candidates=4,
        force=False,
    )

    harness = json.loads(Path(report["artifacts"]["harness_contract_path"]).read_text())
    assert harness["candidate_space"]["modes"] == [4, 8, 12]
    assert harness["candidate_space"]["widths"] == [16, 32, 48]
    assert harness["candidate_space"]["depths"] == [1, 2, 3]
    assert harness["candidate_space"]["train_steps_values"] == [20, 50, 150, 200]
    assert harness["candidate_space"]["forecast_steps_values"] == [47, 95, 190]
    assert harness["candidate_space"]["normalizations"] == ["none", "sample_standard"]


def test_fno_training_iteration_contract_with_graft_limits_to_early_middle_sources(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    previous_evidence_path = tmp_path / "previous_evidence_graft.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "selected_candidate": {
                    "candidate_id": "full_horizon_prior",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 12,
                        "width": 48,
                        "depth": 2,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "none",
                        "forecast_steps": 190,
                        "train_steps": 300,
                    },
                    "next_generation_constraints": {
                        "forecast_steps": [190],
                        "train_steps": [300],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=previous_evidence_path,
        output_dir=tmp_path / "agent_iter_graft",
        workspace_dir=tmp_path,
        source_path=tmp_path / "burgers.hdf5",
        baseline_prediction_path=tmp_path / "baseline.hdf5",
        task1_test_path=tmp_path / "task1_test.hdf5",
        task1_target_path=tmp_path / "task1_val.hdf5",
        train_sample_limit=32,
        val_sample_limit=16,
        max_candidates=4,
        force=False,
    )

    harness = json.loads(Path(report["artifacts"]["harness_contract_path"]).read_text())
    assert harness["candidate_space"]["modes"] == [4, 8]
    assert harness["candidate_space"]["widths"] == [16, 32]
    assert harness["candidate_space"]["depths"] == [1, 2]
    assert harness["candidate_space"]["train_steps_values"] == [1, 2, 20]
    assert harness["candidate_space"]["forecast_steps_values"] == [47, 95]
    plan_command = harness["available_actions"][0]["command"]
    assert "--forecast-steps-value 47" in plan_command
    assert "--forecast-steps-value 95" in plan_command
    assert "--forecast-steps-value 190" not in plan_command
    assert "--train-steps-value 300" not in plan_command
    assert "--train-steps-value 50" not in plan_command
    assert report["agent_contract"]["fast_feedback_guard_active"] is True
    assert "run_task1_fno_graft_from_optimization_report" in {
        action["action_id"] for action in harness["available_actions"]
    }


def test_fno_training_iteration_contract_repairs_graft_test_path_sample_mismatch(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    previous_evidence_path = tmp_path / "previous_evidence_graft.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "selected_candidate": {
                    "candidate_id": "full_horizon_prior",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 12,
                        "width": 48,
                        "depth": 2,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "none",
                        "forecast_steps": 190,
                        "train_steps": 300,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    source_path = tmp_path / "burgers.hdf5"
    baseline_path = tmp_path / "baseline.hdf5"
    test_path = tmp_path / "task1_test.hdf5"
    target_path = tmp_path / "task1_val.hdf5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 1001, 16), dtype=np.float32))
    with h5py.File(baseline_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((100, 200, 256), dtype=np.float32))
    with h5py.File(test_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1000, 10, 256), dtype=np.float32))
    with h5py.File(target_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((100, 200, 256), dtype=np.float32))

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=previous_evidence_path,
        output_dir=tmp_path / "agent_iter_graft_repair",
        workspace_dir=tmp_path,
        source_path=source_path,
        baseline_prediction_path=baseline_path,
        task1_test_path=test_path,
        task1_target_path=target_path,
        train_sample_limit=32,
        val_sample_limit=16,
        max_candidates=4,
        force=False,
    )

    harness = json.loads(Path(report["artifacts"]["harness_contract_path"]).read_text())
    repair = harness["data_contract"]["graft_contract_repair"]
    assert repair["applied"] is True
    assert repair["reason"] == "baseline_sample_count_matches_target_not_test"
    assert repair["requested_test_path"] == str(test_path)
    assert repair["resolved_test_path"] == str(target_path)
    graft_action = next(
        action
        for action in harness["available_actions"]
        if action["action_id"] == "run_task1_fno_graft_from_optimization_report"
    )
    assert f"--test-path {target_path}" in graft_action["command"]
    assert f"--test-path {test_path}" not in graft_action["command"]
    assert graft_action["contract_repair"]["applied"] is True


def test_fno_training_iteration_contract_expands_after_positive_late_safe_graft(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        build_fno_training_iteration_contract_from_evidence,
    )

    previous_evidence_path = tmp_path / "previous_positive_graft_evidence.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "graft_evidence": {
                    "best_task_prediction_score_delta": 0.1048,
                    "late_non_regression_passed": True,
                    "positive_candidate_count": 4,
                },
                "selected_candidate": {
                    "candidate_id": "positive_scout",
                    "config": {
                        "model_family": "fno_lite",
                        "modes": 4,
                        "width": 16,
                        "depth": 1,
                        "loss_type": "horizon_weighted_mse",
                        "normalization": "sample_standard",
                        "forecast_steps": 12,
                        "train_steps": 1,
                    },
                    "next_generation_constraints": {
                        "modes": [4, 8, 12, 16, 20],
                        "widths": [16, 32, 48, 64, 80],
                        "depths": [1, 2, 3, 4, 5],
                        "forecast_steps": [12, 20],
                        "train_steps": [1, 2, 20, 50, 100, 150, 300],
                        "normalizations": ["sample_standard"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    source_path = tmp_path / "burgers.hdf5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((2, 201, 16), dtype=np.float32))

    report = build_fno_training_iteration_contract_from_evidence(
        previous_evidence_path=previous_evidence_path,
        output_dir=tmp_path / "agent_iter_positive_graft",
        workspace_dir=tmp_path,
        source_path=source_path,
        baseline_prediction_path=tmp_path / "baseline.hdf5",
        task1_test_path=tmp_path / "task1_test.hdf5",
        task1_target_path=tmp_path / "task1_val.hdf5",
        train_sample_limit=64,
        val_sample_limit=24,
        max_candidates=4,
        force=False,
    )

    harness = json.loads(Path(report["artifacts"]["harness_contract_path"]).read_text())
    assert harness["budget_contract"]["max_candidates"] == 6
    assert harness["candidate_space"]["modes"] == [4, 8, 12, 16]
    assert harness["candidate_space"]["widths"] == [16, 32, 48, 64]
    assert harness["candidate_space"]["depths"] == [1, 2, 3, 4]
    assert harness["candidate_space"]["train_steps_values"] == [
        1,
        2,
        20,
        50,
        51,
        100,
        150,
    ]
    assert harness["candidate_space"]["forecast_steps_values"] == [12, 20]
    plan_command = harness["available_actions"][0]["command"]
    assert "--max-candidates 6" in plan_command
    assert "--mode 16" in plan_command
    assert "--width 64" in plan_command
    assert "--train-steps-value 150" in plan_command
    assert "--mode 20" not in plan_command
    assert "--train-steps-value 300" not in plan_command


def test_fno_training_iteration_contract_command_returns_runtime_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        build_fno_training_iteration_contract_command,
    )

    previous_evidence_path = tmp_path / "previous_evidence.json"
    previous_evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "candidate_records": [],
                "next_generation_constraints": {},
            }
        ),
        encoding="utf-8",
    )

    payload = build_fno_training_iteration_contract_command(
        previous_evidence_path=str(previous_evidence_path),
        output_dir=str(tmp_path / "agent_iter"),
        workspace_dir=str(tmp_path),
        source_path=str(tmp_path / "burgers.hdf5"),
        dataset_key="tensor",
        train_sample_limit=32,
        val_sample_limit=16,
        context_steps=10,
        time_reduction=5,
        space_reduction=4,
        batch_size=4,
        learning_rate=1e-3,
        seed=0,
        max_candidates=4,
        force=False,
    )

    assert payload["state_hint"] == "FNO_TRAINING_ITERATION_CONTRACT_BUILT"
    assert payload["command"] == "build-fno-training-iteration-contract"
    assert payload["next_hint"] == "run-claude-agent-sdk-runtime"
    assert Path(payload["artifacts"]["prompt_path"]).exists()
    assert Path(payload["artifacts"]["harness_contract_path"]).exists()


def test_materialize_training_windows_for_hypothesis_sandbox_writes_hdf5_bundle(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        materialize_training_windows_for_hypothesis_sandbox,
    )

    windows_path = tmp_path / "val_windows.npz"
    context = np.ones((3, 10, 16), dtype=np.float32)
    target = np.full((3, 20, 16), 2.0, dtype=np.float32)
    full_reference = np.concatenate([context, target], axis=1)
    np.savez_compressed(
        windows_path,
        context=context,
        target=target,
        full_reference=full_reference,
    )

    report = materialize_training_windows_for_hypothesis_sandbox(
        windows_path=windows_path,
        output_dir=tmp_path / "sandbox_bundle",
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_SANDBOX_WINDOW_BUNDLE_MATERIALIZED"
    assert report["source_windows_path"] == str(windows_path)
    assert report["shapes"]["context"] == [3, 10, 16]
    assert report["shapes"]["future_target"] == [3, 20, 16]
    assert report["shapes"]["target"] == [3, 30, 16]
    assert report["shapes"]["reference"] == [3, 30, 16]
    for key in ["context_path", "target_path", "reference_path", "manifest_path"]:
        assert Path(report["artifacts"][key]).exists()
    with h5py.File(report["artifacts"]["context_path"], "r") as handle:
        np.testing.assert_allclose(handle["tensor"][...], context)
    with h5py.File(report["artifacts"]["target_path"], "r") as handle:
        np.testing.assert_allclose(handle["tensor"][...], full_reference)


def test_materialize_training_windows_command_returns_sandbox_paths(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        materialize_training_windows_for_hypothesis_sandbox_command,
    )

    windows_path = tmp_path / "val_windows.npz"
    np.savez_compressed(
        windows_path,
        context=np.zeros((2, 10, 8), dtype=np.float32),
        target=np.zeros((2, 15, 8), dtype=np.float32),
    )

    payload = materialize_training_windows_for_hypothesis_sandbox_command(
        windows_path=str(windows_path),
        output_dir=str(tmp_path / "bundle"),
        force=False,
    )

    assert payload["state_hint"] == "HYPOTHESIS_SANDBOX_WINDOW_BUNDLE_MATERIALIZED"
    assert payload["command"] == "materialize-training-windows-for-hypothesis-sandbox"
    assert Path(payload["artifacts"]["context_path"]).exists()
    assert Path(payload["artifacts"]["target_path"]).exists()
    assert payload["next_hint"] == "run-hypothesis-algorithm-sandbox"


def test_convert_hypothesis_algorithm_evidence_for_candidate_promotion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_short_window_training import (
        convert_hypothesis_algorithm_evidence_for_candidate_promotion,
    )
    from pde_agent.core.literature_algorithm_lab import (
        evaluate_algorithm_candidate_promotion,
    )

    evidence_path = tmp_path / "hypothesis_algorithm_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
                "algorithm_card": {
                    "blueprint_id": "bp_algorithm_designer_001_residual_correction",
                    "candidate_family": "residual_correction",
                    "hypothesis_id": "h_residual_correction_001",
                    "evidence_status": "promote_to_strategy",
                },
                "evidence": {
                    "rules_passed": True,
                    "score_delta": 0.25,
                    "task1_total_score_delta": 0.25,
                    "inference_time_delta": 0.01,
                    "physics_residual_delta": None,
                    "sandbox_report_path": "sandbox.json",
                    "judge_report_path": "judge.json",
                },
                "promotion_rubric": {
                    "decision": "promote_to_strategy",
                    "reason_codes": ["task1_total_score_improved"],
                },
                "next_generation_constraints": {
                    "prefer_mechanism_families": ["residual_correction"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = convert_hypothesis_algorithm_evidence_for_candidate_promotion(
        hypothesis_evidence_path=evidence_path,
        output_dir=tmp_path / "promotion_bridge",
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_EVIDENCE_PROMOTION_BRIDGE_BUILT"
    assert report["algorithm_candidate_id"] == (
        "hypothesis_bp_algorithm_designer_001_residual_correction"
    )
    candidates = json.loads(Path(report["artifacts"]["algorithm_candidates_path"]).read_text())
    sandbox_evidence = json.loads(
        Path(report["artifacts"]["sandbox_evidence_path"]).read_text()
    )
    assert candidates["state_hint"] == "ALGORITHM_CANDIDATES_COMPILED"
    assert candidates["algorithm_candidates"][0]["algorithm_candidate_id"] == report[
        "algorithm_candidate_id"
    ]
    assert candidates["algorithm_candidates"][0]["mechanism"] == (
        "Apply a sandbox-validated residual_correction hypothesis from Hypothesis Lab."
    )
    assert candidates["algorithm_candidates"][0]["target_problem"] == "PDE_Burgers"
    assert candidates["algorithm_candidates"][0]["target_task"] == "task1"
    assert candidates["algorithm_candidates"][0]["promotion_criteria"][
        "requires_ablation"
    ] is False
    assert sandbox_evidence["candidate_id"] == report["algorithm_candidate_id"]
    assert sandbox_evidence["score_delta_vs_current_best"] == 0.25
    assert sandbox_evidence["reproducibility"]["deterministic"] is True

    promotion = evaluate_algorithm_candidate_promotion(
        algorithm_candidates_path=report["artifacts"]["algorithm_candidates_path"],
        algorithm_candidate_id=report["algorithm_candidate_id"],
        sandbox_evidence_paths=[report["artifacts"]["sandbox_evidence_path"]],
        output_path=tmp_path / "candidate_promotion.json",
        current_best_score=0.0,
        force=False,
    )
    assert promotion["decision"] == "promote"
    assert promotion["promote_to_strategy_growth"] is True

    from pde_agent.core.literature_expansion import (
        record_promoted_algorithm_strategy_proposal,
    )

    proposal = record_promoted_algorithm_strategy_proposal(
        candidate_promotion_path=tmp_path / "candidate_promotion.json",
        output_path=tmp_path / "strategy_proposal.json",
        force=False,
    )
    assert proposal["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert proposal["source_type"] == "agent_hypothesis"


def test_convert_hypothesis_algorithm_evidence_command_returns_bridge_paths(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        convert_hypothesis_algorithm_evidence_for_candidate_promotion_command,
    )

    evidence_path = tmp_path / "hypothesis_algorithm_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
                "algorithm_card": {
                    "blueprint_id": "bp_test",
                    "candidate_family": "residual_correction",
                    "hypothesis_id": "h_test",
                    "evidence_status": "promote_to_strategy",
                },
                "evidence": {
                    "rules_passed": True,
                    "score_delta": 0.1,
                    "inference_time_delta": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = convert_hypothesis_algorithm_evidence_for_candidate_promotion_command(
        hypothesis_evidence_path=str(evidence_path),
        output_dir=str(tmp_path / "bridge"),
        force=False,
    )

    assert payload["state_hint"] == "HYPOTHESIS_EVIDENCE_PROMOTION_BRIDGE_BUILT"
    assert payload["command"] == (
        "convert-hypothesis-algorithm-evidence-for-candidate-promotion"
    )
    assert Path(payload["artifacts"]["algorithm_candidates_path"]).exists()
    assert Path(payload["artifacts"]["sandbox_evidence_path"]).exists()
    assert payload["next_hint"] == "evaluate-algorithm-candidate-promotion"
