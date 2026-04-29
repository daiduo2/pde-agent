import json
from pathlib import Path
from typing import Any


def _write_competition_hdf5(path: Path, *, sample_count: int) -> None:
    h5py = __import__("h5py")
    np = __import__("numpy")
    tensor = np.zeros((sample_count, 10, 256), dtype=np.float32)
    for sample_index in range(sample_count):
        tensor[sample_index, :, :] = sample_index
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
        handle.create_dataset("x_coordinate", data=np.linspace(0.0, 1.0, 256))
        handle.attrs["fixture"] = "unet-runtime-profile"


def test_profile_unet_runtime_writes_shard_scaling_report(tmp_path: Path) -> None:
    from pde_agent.core.unet_runtime_profile import profile_unet_runtime

    test_path = tmp_path / "task1_test.hdf5"
    _write_competition_hdf5(test_path, sample_count=5)
    calls: list[int] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        h5py = __import__("h5py")
        with h5py.File(kwargs["test_path"], "r") as handle:
            sample_count = int(handle["tensor"].shape[0])
        calls.append(sample_count)
        return {
            "inputs": {"sample_count": sample_count},
            "artifacts": {
                "candidate_execution_report_path": str(
                    Path(kwargs["output_dir"]) / "candidate_execution_report.json"
                )
            },
            "inference": {
                "elapsed_seconds": sample_count * 0.2,
                "phase_timings": {"rollout_seconds": sample_count * 0.15},
            },
        }

    report = profile_unet_runtime(
        task_id=1,
        test_path=test_path,
        checkpoint_path="data/pdebench/checkpoints/burgers_Unet.tar",
        output_dir=tmp_path / "profile",
        shard_sizes=[1, 4],
        timeout_seconds=120.0,
        full_sample_count=1000,
        force=False,
        runner=fake_runner,
    )

    assert report["state_hint"] == "UNET_RUNTIME_PROFILED"
    assert calls == [1, 4]
    assert [record["sample_count"] for record in report["shards"]] == [1, 4]
    assert report["shards"][0]["projected_full_inference_seconds"] == 200.0
    assert report["shards"][0]["phase_timings"]["rollout_seconds"] == 0.15
    assert report["shards"][1]["timeout_risk"] is True
    assert report["recommendation"]["next_command"] == "record-unet-timeout-evidence"
    assert Path(report["artifacts"]["runtime_profile_path"]).exists()
    h5py = __import__("h5py")
    with h5py.File(report["shards"][1]["shard_test_path"], "r") as handle:
        assert handle["tensor"].shape == (4, 10, 256)
        assert handle["x_coordinate"].shape == (256,)


def test_profile_unet_runtime_recommends_submission_when_projection_fits_budget(
    tmp_path: Path,
) -> None:
    from pde_agent.core.unet_runtime_profile import profile_unet_runtime

    test_path = tmp_path / "task1_test.hdf5"
    _write_competition_hdf5(test_path, sample_count=4)

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        h5py = __import__("h5py")
        with h5py.File(kwargs["test_path"], "r") as handle:
            sample_count = int(handle["tensor"].shape[0])
        return {
            "inputs": {"sample_count": sample_count},
            "artifacts": {},
            "inference": {"elapsed_seconds": sample_count * 0.01},
        }

    report = profile_unet_runtime(
        task_id=1,
        test_path=test_path,
        checkpoint_path="checkpoint.pt",
        output_dir=tmp_path / "profile",
        shard_sizes=[2],
        timeout_seconds=120.0,
        full_sample_count=1000,
        runner=fake_runner,
    )

    assert report["recommendation"]["next_command"] == "predict-unet-competition"
    assert report["recommendation"]["strategy_id"] == "unet_full_submission_runtime_v0"
    assert report["shards"][0]["projected_full_inference_seconds"] == 10.0


def test_profile_unet_runtime_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.profile_unet_runtime as command_module
    from typer.testing import CliRunner

    def fake_profile_unet_runtime(**_: Any) -> dict[str, Any]:
        profile_path = tmp_path / "profile.json"
        profile_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "UNET_RUNTIME_PROFILED",
            "task_id": 1,
            "timeout_seconds": 120.0,
            "inference_batch_size": None,
            "torch_num_threads": None,
            "artifacts": {"runtime_profile_path": str(profile_path)},
            "shards": [
                {
                    "status": "completed",
                    "sample_count": 1,
                    "inference_elapsed_seconds": 0.1,
                    "projected_full_inference_seconds": 100.0,
                    "timeout_risk": False,
                }
            ],
            "recommendation": {
                "next_command": "predict-unet-competition",
                "strategy_id": "unet_full_submission_runtime_v0",
            },
        }

    monkeypatch.setattr(
        command_module,
        "profile_unet_runtime",
        fake_profile_unet_runtime,
    )

    result = CliRunner().invoke(
        app,
        [
            "profile-unet-runtime",
            "--task-id",
            "1",
            "--test-path",
            "task1_test.hdf5",
            "--checkpoint-path",
            "burgers_Unet.tar",
            "--output-dir",
            str(tmp_path / "profile"),
            "--shard-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "UNET_RUNTIME_PROFILED"
    assert payload["metrics"]["max_shard_sample_count"] == 1
    assert payload["next_hint"] == "predict-unet-competition"
