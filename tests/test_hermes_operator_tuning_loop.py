import json
from pathlib import Path
from typing import Any


def test_hermes_operator_tuning_loop_selects_fastest_feasible_unet_config(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_operator_tuning_loop import (
        run_hermes_operator_tuning_loop,
    )

    calls: list[tuple[int | None, int | None]] = []

    def fake_profiler(**kwargs: Any) -> dict[str, Any]:
        batch = kwargs["inference_batch_size"]
        threads = kwargs["torch_num_threads"]
        calls.append((batch, threads))
        projected = {
            (4, 1): 129.0,
            (4, 2): 121.0,
            (8, 1): 99.0,
            (8, 2): 93.0,
        }[(batch, threads)]
        profile_path = Path(kwargs["output_dir"]) / "unet_runtime_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "UNET_RUNTIME_PROFILED",
            "artifacts": {"runtime_profile_path": str(profile_path)},
            "recommendation": {
                "next_command": (
                    "predict-unet-competition"
                    if projected <= kwargs["timeout_seconds"]
                    else "record-unet-timeout-evidence"
                ),
                "strategy_id": "unet_full_submission_runtime_v0",
            },
            "shards": [
                {
                    "status": "completed",
                    "sample_count": 16,
                    "inference_elapsed_seconds": projected / 1000.0 * 16.0,
                    "projected_full_inference_seconds": projected,
                    "timeout_risk": projected > kwargs["timeout_seconds"],
                    "phase_timings": {"rollout_seconds": projected / 1000.0 * 15.8},
                }
            ],
        }

    report = run_hermes_operator_tuning_loop(
        output_dir=tmp_path / "operator_tuning",
        operator_family="UNet",
        task_id=1,
        test_path="task1_test.hdf5",
        checkpoint_path="burgers_Unet.tar",
        shard_sizes=[16],
        inference_batch_sizes=[4, 8],
        torch_num_threads_values=[1, 2],
        timeout_seconds=120.0,
        full_sample_count=1000,
        profiler=fake_profiler,
        force=False,
    )

    assert calls == [(4, 1), (4, 2), (8, 1), (8, 2)]
    assert report["state_hint"] == "HERMES_OPERATOR_TUNING_LOOP_RAN"
    assert report["decision"]["action"] == "accept_execution_config"
    assert report["selected_config"] == {
        "inference_batch_size": 8,
        "torch_num_threads": 2,
    }
    assert report["best_projected_full_inference_seconds"] == 93.0
    assert report["permissions"]["diagnostic_profile"] is True
    assert report["permissions"]["execution_path_tuning"] is True
    assert report["permissions"]["model_logic_tuning"] is False
    result_path = Path(report["artifacts"]["local_strategy_result_path"])
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["state_hint"] == "LOCAL_EXPERIMENT_RESULT"
    assert result["result_status"] == "completed"
    assert result["source"]["selected_config"]["torch_num_threads"] == 2


def test_hermes_operator_tuning_loop_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_operator_tuning_loop as command_module
    from typer.testing import CliRunner

    def fake_loop(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "report.json"
        result_path = tmp_path / "result.json"
        report_path.write_text("{}", encoding="utf-8")
        result_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_OPERATOR_TUNING_LOOP_RAN",
            "operator_family": "UNet",
            "loop_report_path": str(report_path),
            "artifacts": {"local_strategy_result_path": str(result_path)},
            "evaluated_config_count": 1,
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 2,
            },
            "best_projected_full_inference_seconds": 98.0,
            "decision": {"action": "accept_execution_config"},
            "next_command": "predict-unet-competition",
        }

    monkeypatch.setattr(
        command_module,
        "run_hermes_operator_tuning_loop",
        fake_loop,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-operator-tuning-loop",
            "--output-dir",
            str(tmp_path / "loop"),
            "--operator-family",
            "UNet",
            "--task-id",
            "1",
            "--test-path",
            "task1_test.hdf5",
            "--checkpoint-path",
            "burgers_Unet.tar",
            "--shard-size",
            "16",
            "--inference-batch-size",
            "8",
            "--torch-num-threads",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_OPERATOR_TUNING_LOOP_RAN"
    assert payload["metrics"]["selected_inference_batch_size"] == 8
    assert payload["next_hint"] == "predict-unet-competition"
