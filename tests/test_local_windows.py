import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pde_agent.cli import app
from pde_agent.core.local_windows import LocalWindowError, build_local_windows


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)


def test_build_local_windows_slices_reduced_hdf5(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    raw = np.arange(3 * 12 * 8, dtype=np.float32).reshape(3, 12, 8)
    _write_hdf5(source, raw)
    output = tmp_path / "windows.npz"

    summary = build_local_windows(
        source_path=source,
        output_path=output,
        dataset_key="tensor",
        start_index=1,
        sample_limit=2,
        context_steps=2,
        forecast_steps=3,
        time_reduction=2,
        space_reduction=2,
        force=False,
    )

    reduced = raw[1:3, ::2, ::2]
    with np.load(output, allow_pickle=False) as archive:
        assert {
            "context",
            "target",
            "full_reference",
            "metadata_json",
        }.issubset(archive.files)
        assert np.array_equal(archive["context"], reduced[:, :2, :].astype(np.float32))
        assert np.array_equal(archive["target"], reduced[:, 2:5, :].astype(np.float32))
        assert np.array_equal(
            archive["full_reference"], reduced[:, :5, :].astype(np.float32)
        )
        assert archive["metadata_json"].shape == ()
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["source_path"] == str(source)
    assert metadata["dataset_key"] == "tensor"
    assert metadata["raw_shape"] == [3, 12, 8]
    assert metadata["reduced_shape"] == [2, 6, 4]
    assert metadata["start_index"] == 1
    assert metadata["sample_limit"] == 2
    assert metadata["context_steps"] == 2
    assert metadata["forecast_steps"] == 3
    assert summary["context_shape"] == [2, 2, 4]
    assert summary["target_shape"] == [2, 3, 4]
    assert summary["full_reference_shape"] == [2, 5, 4]


def test_build_local_windows_refuses_overwrite_without_force(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 8, 4), dtype=np.float32))
    output = tmp_path / "windows.npz"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(LocalWindowError, match="already exists"):
        build_local_windows(
            source_path=source,
            output_path=output,
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_sample_limit_overrun(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((2, 8, 4), dtype=np.float32))

    with pytest.raises(LocalWindowError, match="exceeds sample count"):
        build_local_windows(
            source_path=source,
            output_path=tmp_path / "windows.npz",
            dataset_key="tensor",
            start_index=1,
            sample_limit=2,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_short_reduced_time(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 9, 4), dtype=np.float32))
    output = tmp_path / "windows.npz"

    with pytest.raises(LocalWindowError, match="reduced time dimension"):
        build_local_windows(
            source_path=source,
            output_path=output,
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=3,
            forecast_steps=3,
            time_reduction=2,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_missing_dataset_key(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 8, 4), dtype=np.float32), key="other")

    with pytest.raises(LocalWindowError, match="missing dataset key"):
        build_local_windows(
            source_path=source,
            output_path=tmp_path / "windows.npz",
            dataset_key="tensor",
            start_index=0,
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_non_integer_window_args(
    tmp_path: Path,
) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((1, 8, 4), dtype=np.float32))

    with pytest.raises(LocalWindowError, match="must be an integer"):
        build_local_windows(
            source_path=source,
            output_path=tmp_path / "windows.npz",
            dataset_key="tensor",
            start_index=0.5,  # type: ignore[arg-type]
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_rejects_nested_dataset_key(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "burgers.hdf5"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("group")
        group.create_dataset("tensor", data=np.ones((1, 8, 4), dtype=np.float32))

    with pytest.raises(LocalWindowError, match="root dataset"):
        build_local_windows(
            source_path=source,
            output_path=tmp_path / "windows.npz",
            dataset_key="group/tensor",
            start_index=0,
            sample_limit=1,
            context_steps=2,
            forecast_steps=2,
            time_reduction=1,
            space_reduction=1,
            force=False,
        )


def test_build_local_windows_cli_writes_json_payload(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((2, 10, 8), dtype=np.float32))
    output = tmp_path / "derived" / "windows.npz"

    result = CliRunner().invoke(
        app,
        [
            "build-local-windows",
            "--source-path",
            str(source),
            "--output-path",
            str(output),
            "--dataset-key",
            "tensor",
            "--start-index",
            "0",
            "--sample-limit",
            "2",
            "--context-steps",
            "2",
            "--forecast-steps",
            "3",
            "--time-reduction",
            "2",
            "--space-reduction",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-local-windows"
    assert payload["state_hint"] == "LOCAL_WINDOWS_BUILT"
    assert payload["artifacts"]["windows_npz_path"] == str(output)
    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["context_steps"] == 2
    assert payload["metrics"]["forecast_steps"] == 3
    assert payload["metrics"]["space_points"] == 4
    assert output.exists()
