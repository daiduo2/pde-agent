from __future__ import annotations

import hashlib
import importlib.util
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
