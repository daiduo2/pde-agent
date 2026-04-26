import json
from pathlib import Path

import pytest

from pde_agent.core.fno_iteration_runner import (
    FNOIterationRunnerError,
    run_fno_iteration,
)

torch = pytest.importorskip("torch", reason="FNO iteration runner tests require torch")
np = pytest.importorskip("numpy")


def _official_fno_state_dict(
    *,
    history_steps: int = 3,
    width: int = 4,
    modes: int = 2,
    fc_hidden: int = 5,
) -> dict[str, object]:
    state: dict[str, object] = {
        "fc0.weight": torch.zeros(width, history_steps + 1),
        "fc0.bias": torch.zeros(width),
        "fc1.weight": torch.zeros(fc_hidden, width),
        "fc1.bias": torch.zeros(fc_hidden),
        "fc2.weight": torch.zeros(1, fc_hidden),
        "fc2.bias": torch.zeros(1),
    }
    for index in range(4):
        state[f"conv{index}.weights1"] = torch.zeros(
            width,
            width,
            modes,
            dtype=torch.complex64,
        )
        state[f"w{index}.weight"] = torch.zeros(
            width,
            width,
            1,
        )
        state[f"w{index}.bias"] = torch.zeros(width)
    return state


def _write_checkpoint(path: Path) -> None:
    torch.save({"model_state_dict": _official_fno_state_dict()}, path)


def _write_windows(path: Path) -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    context = np.zeros((2, 4, 8), dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        for time_index in range(3):
            target[sample, time_index, :] = (time_index + 1) * 0.1 * np.sin(
                x + sample * 0.1
            )
    np.savez(path, context=context, target=target)


def test_run_fno_iteration_writes_report_for_all_candidates(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "iteration"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    report = run_fno_iteration(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        max_train_steps=1,
        learning_rate=1e-4,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "FNO_ITERATION_COMPLETED"
    assert report["candidate_count"] == 2
    assert report["winner"]["experiment_id"]
    assert report["artifacts"]["iteration_report_path"] == str(
        output_dir / "fno_iteration_report.json"
    )
    assert Path(report["artifacts"]["comparison_report_path"]).exists()
    assert all(candidate["evaluation_report_path"] for candidate in report["candidates"])
    written = json.loads((output_dir / "fno_iteration_report.json").read_text())
    assert written == report


def test_run_fno_iteration_rejects_non_empty_output_without_force(
    tmp_path: Path,
) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "iteration"
    output_dir.mkdir()
    existing = output_dir / "keep.txt"
    existing.write_text("previous", encoding="utf-8")
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    with pytest.raises(
        FNOIterationRunnerError,
        match="output_dir already contains files",
    ):
        run_fno_iteration(
            windows_path=windows_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            max_train_steps=1,
            learning_rate=1e-4,
            nu=0.001,
            dt=1.0,
            dx=1.0,
            force=False,
        )

    assert existing.read_text(encoding="utf-8") == "previous"


def test_run_fno_iteration_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "iteration"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    result = CliRunner().invoke(
        app,
        [
            "run-fno-iteration",
            "--windows-path",
            str(windows_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--max-train-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-fno-iteration"
    assert payload["state_hint"] == "FNO_ITERATION_COMPLETED"
    assert payload["artifacts"]["iteration_report_path"] == str(
        output_dir / "fno_iteration_report.json"
    )
    assert payload["metrics"]["candidate_count"] == 2
    assert payload["metrics"]["winner_experiment_id"]
    assert payload["next_hint"] == "run-fno-iteration"
