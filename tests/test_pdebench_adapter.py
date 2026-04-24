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


def test_load_data_config_resolves_relative_source_path_from_config_dir(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "resolved.json"
    config_path.write_text(
        json.dumps({"data": {"mode": "npz_fixture", "source_path": "../data.npz"}}),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert config.source_path == tmp_path / "data.npz"


def test_synthetic_inspection_uses_stable_schema() -> None:
    info = inspect_data_source(
        DataConfig(
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
    )

    assert info == {
        "source_mode": "synthetic",
        "source_path": None,
        "dataset_key": None,
        "available_keys": [],
        "raw_shape": None,
        "raw_dtype": None,
        "reduced_shape": None,
        "contract_shape": {
            "context": (None, 10, None),
            "prediction": (None, 190, None),
            "target": (None, 190, None),
        },
        "sample_count": None,
        "context_time": 10,
        "forecast_time": 190,
        "space_points": None,
        "is_compatible": True,
        "warnings": ["synthetic mode does not inspect external array data"],
    }


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


def test_npz_inspection_uses_stable_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.npz"
    np.savez(
        source_path,
        context=np.ones((2, 3, 5), dtype=np.float32),
        prediction=np.ones((2, 4, 5), dtype=np.float64),
        target=np.ones((2, 4, 5), dtype=np.float64),
    )

    info = inspect_data_source(
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

    assert info == {
        "source_mode": "npz_fixture",
        "source_path": str(source_path),
        "dataset_key": None,
        "available_keys": ("context", "prediction", "target"),
        "raw_shape": None,
        "raw_dtype": "float64",
        "reduced_shape": None,
        "contract_shape": {
            "context": (2, 3, 5),
            "prediction": (2, 4, 5),
            "target": (2, 4, 5),
        },
        "sample_count": 2,
        "context_time": 3,
        "forecast_time": 4,
        "space_points": 5,
        "is_compatible": True,
        "warnings": [],
    }


def test_npz_inspection_rejects_invalid_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid.npz"
    np.savez(
        source_path,
        context=np.ones((2, 3), dtype=np.float32),
        prediction=np.ones((2, 4, 5), dtype=np.float64),
        target=np.ones((2, 4, 5), dtype=np.float64),
    )

    config = DataConfig(
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

    with pytest.raises(DataInspectionError, match="context must"):
        inspect_data_source(config)


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
    assert info["source_mode"] == "pdebench_hdf5"
    assert info["source_path"] == str(source_path)
    assert info["available_keys"] == ("tensor",)
    assert info["raw_shape"] == (3, 20, 16)
    assert info["raw_dtype"] == "float32"
    assert info["reduced_shape"] == (2, 10, 4)
    assert info["contract_shape"] == {
        "context": (2, 2, 4),
        "prediction": (2, 3, 4),
        "target": (2, 3, 4),
    }
    assert info["sample_count"] == 2
    assert info["context_time"] == 2
    assert info["forecast_time"] == 3
    assert info["space_points"] == 4
    assert info["is_compatible"] is True
    assert info["warnings"] == []


def test_hdf5_loads_reduced_bundle(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    source_path = tmp_path / "burgers.h5"
    data = np.arange(3 * 12 * 8, dtype=np.float32).reshape(3, 12, 8)
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
    assert bundle.full_reference is not None
    assert bundle.full_reference.shape[0] == 1
    assert np.array_equal(bundle.context, reduced[:, :2, :])
    assert np.array_equal(bundle.target, reduced[:, 2:5, :])
    assert np.array_equal(bundle.prediction, bundle.target)
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
