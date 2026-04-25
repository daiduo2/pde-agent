import json
from pathlib import Path

import numpy as np
import pytest

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
        assert np.array_equal(archive["context"], reduced[:, :2, :].astype(np.float32))
        assert np.array_equal(archive["target"], reduced[:, 2:5, :].astype(np.float32))
        assert np.array_equal(
            archive["full_reference"], reduced[:, :5, :].astype(np.float32)
        )
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
