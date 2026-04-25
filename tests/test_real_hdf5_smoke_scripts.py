from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _download_module():
    repo_root = Path(__file__).resolve().parents[1]
    return _load_module(
        "download_pdebench_file",
        repo_root / "scripts" / "download_pdebench_file.py",
    )


def _prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    return _load_module(
        "prepare_real_hdf5_smoke",
        repo_root / "scripts" / "prepare_real_hdf5_smoke.py",
    )


def test_compute_md5_for_local_file(tmp_path: Path) -> None:
    module = _download_module()
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"burgers")

    assert module.compute_md5(file_path) == hashlib.md5(b"burgers").hexdigest()


def test_refuse_overwrite_without_force(tmp_path: Path) -> None:
    module = _download_module()
    output = tmp_path / "existing.hdf5"
    output.write_bytes(b"old")

    with pytest.raises(FileExistsError):
        module.ensure_can_write(output, force=False)

    module.ensure_can_write(output, force=True)


def test_preflight_detects_md5_mismatch(tmp_path: Path) -> None:
    module = _download_module()
    output = tmp_path / "file.hdf5"
    output.write_bytes(b"wrong")

    result = module.preflight_file(
        output,
        expected_md5="not-the-real-md5",
        min_size_bytes=1,
        inspect_hdf5=False,
    )

    assert result["exists"] is True
    assert result["md5_ok"] is False
    assert result["size_ok"] is True


def test_main_returns_nonzero_for_md5_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _download_module()
    output = tmp_path / "file.hdf5"
    output.write_bytes(b"wrong")

    status = module.main(
        [
            "--preflight-only",
            "--output",
            str(output),
            "--md5",
            "not-the-real-md5",
            "--expected-size-gb",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Preflight failed" in captured.err


def test_main_returns_nonzero_for_size_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _download_module()
    output = tmp_path / "file.hdf5"
    output.write_bytes(b"tiny")

    status = module.main(
        [
            "--preflight-only",
            "--output",
            str(output),
            "--md5",
            "",
            "--expected-size-gb",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Preflight failed" in captured.err


def test_main_returns_nonzero_for_missing_preflight_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _download_module()
    output = tmp_path / "missing.hdf5"

    status = module.main(
        [
            "--preflight-only",
            "--output",
            str(output),
            "--md5",
            "abc",
            "--expected-size-gb",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Preflight failed" in captured.err


def test_preflight_reports_tiny_hdf5_keys(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    module = _download_module()
    output = tmp_path / "tiny.hdf5"
    with h5py.File(output, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1, 4, 8), dtype=np.float32))

    result = module.preflight_file(
        output,
        expected_md5=None,
        min_size_bytes=1,
        inspect_hdf5=True,
    )

    assert result["exists"] is True
    assert result["size_ok"] is True
    assert result["hdf5"]["datasets"][0]["path"] == "tensor"
    assert result["hdf5"]["datasets"][0]["shape"] == [1, 4, 8]


def test_prepare_real_hdf5_smoke_generates_inspect_runtime_files(
    tmp_path: Path,
) -> None:
    module = _prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "real_hdf5_smoke"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.inspect.template.yaml").write_text(
        "\n".join(
            [
                "run_id: {{RUN_ID}}",
                "repo_root: {{REPO_ROOT}}",
                "fixture_root: {{FIXTURE_ROOT}}",
                "run_root: {{RUN_ROOT}}",
                "config_hash: {{CONFIG_HASH}}",
                "config: {{BASE_CONFIG_RUNTIME}}",
                "commands:",
                "  - {{COMMAND_PREFIX}} init-run",
                "  - {{COMMAND_PREFIX}} resolve-config",
                "  - {{COMMAND_PREFIX}} inspect-data",
                "success_transition: DATA_INSPECTED",
            ]
        )
    )
    hdf5_path = repo_root / "tiny.hdf5"
    hdf5_path.write_bytes(b"placeholder hdf5")

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key="tensor",
        sample_limit=1,
        mode="inspect",
    )

    config = json.loads(outputs["base_config_runtime"].read_text())
    run_state = outputs["run_state"].read_text()
    assert config["data"]["mode"] == "pdebench_hdf5"
    assert config["data"]["source"] == str(hdf5_path)
    assert config["data"]["source_path"] == str(hdf5_path)
    assert config["data"]["dataset_key"] == "tensor"
    assert "init-run" in run_state
    assert "resolve-config" in run_state
    assert "inspect-data" in run_state
    assert ".venv/bin/pde-agent" in run_state
    assert "{{" not in run_state


def test_prepare_real_hdf5_smoke_full_mode_contains_full_chain(
    tmp_path: Path,
) -> None:
    module = _prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "real_hdf5_smoke"
    fixture_root.mkdir(parents=True)
    command_names = [
        "init-run",
        "resolve-config",
        "inspect-data",
        "eval-rollout",
        "analyze-spectrum",
        "check-physics",
        "interpret-error",
        "select-hypothesis",
        "record-timing",
        "collect-artifacts",
    ]
    (fixture_root / "run_state.full.template.yaml").write_text(
        "\n".join(
            [
                "run_id: {{RUN_ID}}",
                "repo_root: {{REPO_ROOT}}",
                "fixture_root: {{FIXTURE_ROOT}}",
                "run_root: {{RUN_ROOT}}",
                "config_hash: {{CONFIG_HASH}}",
                "config: {{BASE_CONFIG_RUNTIME}}",
                "commands:",
                *[f"  - {{{{COMMAND_PREFIX}}}} {name}" for name in command_names],
                "success_transition: HYPOTHESIS_SELECTED",
            ]
        )
    )
    hdf5_path = repo_root / "tiny.hdf5"
    hdf5_path.write_bytes(b"placeholder hdf5")

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key="tensor",
        sample_limit=1,
        mode="full",
    )

    config = json.loads(outputs["base_config_runtime"].read_text())
    run_state = outputs["run_state"].read_text()
    assert config["data"]["mode"] == "pdebench_hdf5"
    assert config["data"]["source"] == str(hdf5_path)
    assert config["data"]["source_path"] == str(hdf5_path)
    assert config["data"]["dataset_key"] == "tensor"
    for command_name in command_names:
        assert command_name in run_state
    assert ".venv/bin/pde-agent" in run_state
    assert "{{" not in run_state


def test_real_hdf5_inspect_runtime_command_chain(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    repo_root = Path(__file__).resolve().parents[1]
    module = _prepare_module()
    hdf5_path = tmp_path / "tiny_real.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "tensor",
            data=np.zeros((1, 1000, 1024), dtype=np.float32),
        )

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key="tensor",
        sample_limit=1,
        mode="inspect",
    )

    commands = []
    for line in outputs["run_state"].read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))

    last = ""
    for command in commands:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        last = result.stdout

    assert '"state_hint": "DATA_INSPECTED"' in last


def test_prepare_real_hdf5_smoke_rejects_bad_mode(tmp_path: Path) -> None:
    module = _prepare_module()
    hdf5_path = tmp_path / "tiny.hdf5"
    hdf5_path.write_bytes(b"placeholder hdf5")

    with pytest.raises(SystemExit, match="mode must be"):
        module.prepare_runtime(
            repo_root=tmp_path,
            hdf5_path=hdf5_path,
            dataset_key="tensor",
            sample_limit=1,
            mode="bad",
        )


def test_prepare_real_hdf5_smoke_rejects_missing_hdf5(tmp_path: Path) -> None:
    module = _prepare_module()

    with pytest.raises(SystemExit, match="HDF5 file does not exist"):
        module.prepare_runtime(
            repo_root=tmp_path,
            hdf5_path=tmp_path / "missing.hdf5",
            dataset_key="tensor",
            sample_limit=1,
            mode="inspect",
        )
