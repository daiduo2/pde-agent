# Local PDEBench Data

Place local PDEBench/Burgers files under:

```text
data/pdebench/burgers/
```

Expected competition files may include:

```text
1D_Burgers_Sols_Nu0.001.hdf5
1D_Burgers_Sols_Nu0.001_pred.hdf5
1D_Burgers_Sols_Nu0.001_val.hdf5
1D_Burgers_Sols_Nu0.001_submit.hdf5
```

Large data files, checkpoints, generated NPZ files, and generated HDF5 files are ignored by git. The harness reads these files only when a run config explicitly points to them.
