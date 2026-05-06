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


def test_task1_short_window_training_command_and_cli_are_hermes_callable(
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
