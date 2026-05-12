from __future__ import annotations

from pathlib import Path

import numpy as np


class BurgersHDF5IOError(RuntimeError):
    pass


def load_hdf5_tensor(path: str | Path, *, dataset_key: str = "tensor") -> np.ndarray:
    h5py = __import__("h5py")
    path = Path(path)
    if not path.is_file():
        raise BurgersHDF5IOError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise BurgersHDF5IOError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(handle[dataset_key][:], dtype=np.float32)
    if data.ndim != 3:
        raise BurgersHDF5IOError(f"expected rank-3 HDF5 tensor, got {data.shape}")
    if data.shape[1] == 1000:
        data = data[:, ::5, :]
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    return data

