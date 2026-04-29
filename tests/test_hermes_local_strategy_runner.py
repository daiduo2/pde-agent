import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _official_fno_state_dict(
    *,
    torch: object,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(37)
    state: dict[str, object] = {
        "fc0.weight": 0.001 * torch.randn(width, history_steps + 1, generator=generator),
        "fc0.bias": torch.zeros(width),
        "fc1.weight": 0.001 * torch.randn(fc_hidden, width, generator=generator),
        "fc1.bias": torch.zeros(fc_hidden),
        "fc2.weight": 0.001 * torch.randn(1, fc_hidden, generator=generator),
        "fc2.bias": torch.zeros(1),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = 0.001 * torch.randn(
            width,
            width,
            modes,
            dtype=torch.complex64,
            generator=generator,
        )
        state[f"w{index}.weight"] = 0.001 * torch.randn(
            width,
            width,
            1,
            generator=generator,
        )
        state[f"w{index}.bias"] = torch.zeros(width)
    return state


def _write_hdf5(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_run_hermes_local_strategy_loop_accepts_after_second_attempt(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_local_strategy_runner import (
        run_hermes_local_strategy_loop,
    )

    def executor(claim: dict[str, object], workspace_dir: Path) -> Path:
        experiment = claim["selected_experiment"]
        assert isinstance(experiment, dict)
        result_path = workspace_dir / "result.json"
        if experiment["strategy_id"] == "fno_spectral_viscosity_filter_v0":
            payload = {
                "state_hint": "LOCAL_EXPERIMENT_RESULT",
                "score": 70.0,
                "baseline_score": 72.0,
                "inference_time_seconds": 50.0,
            }
        else:
            payload = {
                "state_hint": "LOCAL_EXPERIMENT_RESULT",
                "score": 75.0,
                "baseline_score": 72.0,
                "inference_time_seconds": 55.0,
            }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return result_path

    report = run_hermes_local_strategy_loop(
        output_dir=tmp_path / "loop",
        applicable_pde="1D Burgers",
        model_family="FNO",
        failure_modes=["long_horizon_drift", "spectral_dissipation_mismatch"],
        max_experiments=2,
        min_score_delta=1.0,
        max_inference_seconds=120.0,
        executor=executor,
        force=False,
    )

    assert report["state_hint"] == "HERMES_LOCAL_STRATEGY_LOOP_RAN"
    assert report["decision"]["decision"] == "accept_best_strategy"
    assert report["final_route"]["route"]["action"] == "record_strategy_proposal"
    assert report["iteration_count"] == 2
    assert [item["strategy_id"] for item in report["iterations"]] == [
        "fno_spectral_viscosity_filter_v0",
        "fno_rollout_consistency_finetune_v0",
    ]
    assert report["leaderboard"]["best_result"]["score_delta"] == 3.0
    assert Path(report["loop_report_path"]).exists()


def test_run_hermes_local_strategy_loop_cli_dry_run(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-local-strategy-loop",
            "--output-dir",
            str(tmp_path / "loop"),
            "--applicable-pde",
            "1D Burgers",
            "--model-family",
            "FNO",
            "--failure-mode",
            "long_horizon_drift",
            "--max-experiments",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_LOCAL_STRATEGY_LOOP_RAN"
    assert payload["metrics"]["iteration_count"] == 1
    assert payload["metrics"]["decision"] == "accept_best_strategy"
    assert payload["next_hint"] == "record-strategy-proposal"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="FNO competition executor requires torch")
def test_run_hermes_local_strategy_loop_cli_with_fno_competition_executor(
    tmp_path: Path,
) -> None:
    import torch
    from pde_agent.cli import app
    from typer.testing import CliRunner

    test_path = _write_hdf5(
        tmp_path / "task1_test.hdf5",
        np.ones((1, 10, 256), dtype=np.float32),
    )
    target_path = _write_hdf5(
        tmp_path / "target.hdf5",
        np.ones((1, 200, 256), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "fno.pt"
    torch.save({"model_state_dict": _official_fno_state_dict(torch=torch)}, checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-local-strategy-loop",
            "--output-dir",
            str(tmp_path / "loop"),
            "--applicable-pde",
            "1D Burgers",
            "--model-family",
            "FNO",
            "--failure-mode",
            "long_horizon_drift",
            "--max-experiments",
            "1",
            "--executor",
            "fno-competition",
            "--task-id",
            "1",
            "--test-path",
            str(test_path),
            "--target-path",
            str(target_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--submission-id",
            "team_hermes",
            "--baseline-score",
            "0.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_LOCAL_STRATEGY_LOOP_RAN"
    assert payload["metrics"]["iteration_count"] == 1
    assert payload["metrics"]["executor"] == "fno-competition"
    assert payload["next_hint"] == "record-strategy-proposal"
