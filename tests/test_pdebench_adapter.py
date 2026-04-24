import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.core.data_config import DataConfig, load_data_config
from pde_agent.core.pdebench_adapter import (
    DataInspectionError,
    inspect_data_source,
    load_array_bundle,
)


def test_load_data_config_defaults_to_synthetic(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved.json"
    config_path.write_text(json.dumps({"data": "not-a-dict"}), encoding="utf-8")

    config = load_data_config(config_path)

    assert config == DataConfig(
        mode="synthetic",
        problem_id="PDE_Burgers",
        source_path=None,
        dataset_key=None,
        context_steps=10,
        forecast_steps=190,
        time_reduction=5,
        space_reduction=4,
        sample_limit=None,
    )


def test_npz_fixture_loads_canonical_bundle(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.npz"
    context = np.ones((2, 3, 5), dtype=np.float32)
    target = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
    prediction = target + 1
    full_reference = np.concatenate([context, target], axis=1)
    np.savez(
        source_path,
        context=context,
        prediction=prediction,
        target=target,
        full_reference=full_reference,
        metadata_json=json.dumps({"nu": 0.01, "dt": 0.2, "dx": 0.3}),
    )

    bundle = load_array_bundle(
        DataConfig(
            mode="npz_fixture",
            problem_id="PDE_Burgers",
            source_path=source_path,
            dataset_key=None,
            context_steps=3,
            forecast_steps=4,
            time_reduction=1,
            space_reduction=1,
            sample_limit=None,
        )
    )

    assert np.array_equal(bundle.context, context)
    assert np.array_equal(bundle.prediction, prediction)
    assert np.array_equal(bundle.target, target)
    assert bundle.full_reference is not None
    assert np.array_equal(bundle.full_reference, full_reference)
    assert bundle.metadata["problem_id"] == "PDE_Burgers"
    assert bundle.metadata["source_mode"] == "npz_fixture"
    assert bundle.metadata["nu"] == 0.01
    assert bundle.metadata["dt"] == 0.2
    assert bundle.metadata["dx"] == 0.3
    assert bundle.metadata["sample_count"] == 2
    assert bundle.metadata["context_time"] == 3
    assert bundle.metadata["forecast_time"] == 4
    assert bundle.metadata["space_points"] == 5


def test_npz_fixture_missing_source_path_fails() -> None:
    config = DataConfig(
        mode="npz_fixture",
        problem_id="PDE_Burgers",
        source_path=None,
        dataset_key=None,
        context_steps=3,
        forecast_steps=4,
        time_reduction=1,
        space_reduction=1,
        sample_limit=None,
    )

    with pytest.raises(DataInspectionError, match="source_path is required"):
        load_array_bundle(config)


def test_hdf5_inspection_reports_shape(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source_path = tmp_path / "burgers.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("tensor", data=np.ones((3, 20, 16), dtype=np.float32))

    info = inspect_data_source(
        DataConfig(
            mode="pdebench_hdf5",
            problem_id="PDE_Burgers",
            source_path=source_path,
            dataset_key=None,
            context_steps=2,
            forecast_steps=3,
            time_reduction=2,
            space_reduction=4,
            sample_limit=2,
        )
    )

    assert info["dataset_key"] == "tensor"
    assert info["shape"] == (3, 20, 16)
    assert info["dtype"] == "float32"
    assert info["reduced_shape"] == (2, 10, 4)
    assert info["contract_shape"] == {
        "context": (2, 2, 4),
        "target": (2, 3, 4),
    }


def test_hdf5_loads_reduced_bundle(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source_path = tmp_path / "burgers.h5"
    data = np.arange(2 * 12 * 8, dtype=np.float32).reshape(2, 12, 8)
    with h5py.File(source_path, "w") as handle:
        dataset = handle.create_dataset("tensor", data=data)
        dataset.attrs["nu"] = 0.01
        handle.attrs["dt"] = 0.2
        handle.attrs["dx"] = 0.3

    bundle = load_array_bundle(
        DataConfig(
            mode="pdebench_hdf5",
            problem_id="PDE_Burgers",
            source_path=source_path,
            dataset_key="tensor",
            context_steps=2,
            forecast_steps=3,
            time_reduction=2,
            space_reduction=2,
            sample_limit=1,
        )
    )

    reduced = data[:1, ::2, ::2]
    assert np.array_equal(bundle.context, reduced[:, :2, :])
    assert np.array_equal(bundle.target, reduced[:, 2:5, :])
    assert np.array_equal(bundle.prediction, bundle.target)
    assert bundle.full_reference is not None
    assert np.array_equal(bundle.full_reference, reduced)
    assert bundle.metadata["dataset_key"] == "tensor"
    assert bundle.metadata["nu"] == 0.01
    assert bundle.metadata["dt"] == 0.2
    assert bundle.metadata["dx"] == 0.3
    assert bundle.metadata["time_reduction"] == 2
    assert bundle.metadata["space_reduction"] == 2


def test_hdf5_inspection_rejects_ambiguous_dataset_keys(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source_path = tmp_path / "ambiguous.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("first", data=np.ones((1, 6, 4), dtype=np.float32))
        handle.create_dataset("group/second", data=np.ones((1, 6, 4), dtype=np.float32))

    config = DataConfig(
        mode="pdebench_hdf5",
        problem_id="PDE_Burgers",
        source_path=source_path,
        dataset_key=None,
        context_steps=2,
        forecast_steps=3,
        time_reduction=1,
        space_reduction=1,
        sample_limit=None,
    )

    with pytest.raises(DataInspectionError, match="ambiguous HDF5 dataset keys"):
        inspect_data_source(config)
