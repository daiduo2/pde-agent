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

## PDEBench Burgers Smoke Target

The real HDF5 smoke test expects the Burgers file at:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

Official source: PDEBench Datasets, DaRUS, doi:10.18419/DARUS-2986.

Current expected MD5 from DaRUS V8:

```text
44cb784d5a07aa2b1c864cabdcf625f9
```

This file is approximately 7.7 GB and must not be committed.
