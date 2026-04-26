import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.local_window_suite import (
    LocalWindowSuiteError,
    build_local_window_suite,
)


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)


def test_build_local_window_suite_writes_multiple_windows_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "burgers.hdf5"
    raw = np.arange(3 * 21 * 8, dtype=np.float32).reshape(3, 21, 8)
    _write_hdf5(source, raw)
    output_dir = tmp_path / "suite"

    manifest = build_local_window_suite(
        source_path=source,
        output_dir=output_dir,
        dataset_key="tensor",
        start_index=1,
        sample_limit=2,
        context_steps=4,
        forecast_steps=[3, 5],
        time_reduction=2,
        space_reduction=2,
        force=False,
    )

    manifest_path = output_dir / "local_window_suite_manifest.json"
    assert manifest["state_hint"] == "LOCAL_WINDOW_SUITE_BUILT"
    assert manifest["manifest_path"] == str(manifest_path)
    assert manifest["source"]["raw_shape"] == [3, 21, 8]
    assert manifest["source"]["reduced_shape"] == [2, 11, 4]
    assert manifest["source"]["max_forecast_steps"] == 7
    assert manifest["requested_forecast_steps"] == [3, 5]
    assert manifest["profile_count"] == 2
    assert manifest["next_command"] == "run-fno-iteration-dryrun"
    assert [profile["profile_id"] for profile in manifest["profiles"]] == [
        "f003",
        "f005",
    ]
    assert [profile["target_shape"] for profile in manifest["profiles"]] == [
        [2, 3, 4],
        [2, 5, 4],
    ]
    assert (output_dir / "windows_f003.npz").exists()
    assert (output_dir / "windows_f005.npz").exists()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written == manifest

    with np.load(output_dir / "windows_f005.npz", allow_pickle=False) as archive:
        reduced = raw[1:3, ::2, ::2]
        assert np.array_equal(archive["context"], reduced[:, :4, :])
        assert np.array_equal(archive["target"], reduced[:, 4:9, :])


def test_build_local_window_suite_rejects_impossible_horizon_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((2, 11, 8), dtype=np.float32))
    output_dir = tmp_path / "suite"

    with pytest.raises(LocalWindowSuiteError, match="exceeds max_forecast_steps"):
        build_local_window_suite(
            source_path=source,
            output_dir=output_dir,
            dataset_key="tensor",
            start_index=0,
            sample_limit=2,
            context_steps=4,
            forecast_steps=[2, 3],
            time_reduction=2,
            space_reduction=2,
            force=False,
        )

    assert not output_dir.exists()


def test_build_local_window_suite_rejects_duplicate_horizons(tmp_path: Path) -> None:
    source = tmp_path / "burgers.hdf5"
    _write_hdf5(source, np.ones((2, 21, 8), dtype=np.float32))

    with pytest.raises(LocalWindowSuiteError, match="duplicate forecast_steps"):
        build_local_window_suite(
            source_path=source,
            output_dir=tmp_path / "suite",
            dataset_key="tensor",
            start_index=0,
            sample_limit=2,
            context_steps=4,
            forecast_steps=[3, 3],
            time_reduction=2,
            space_reduction=2,
            force=False,
        )
