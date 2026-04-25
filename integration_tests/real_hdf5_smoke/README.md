# Real HDF5 Smoke

This fixture validates a local real PDEBench/Burgers HDF5 file without committing data or running training.

Default target:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

Prepare inspect-mode runtime files:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --sample-limit 1
```

If the file is elsewhere:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --hdf5-path /path/to/1D_Burgers_Sols_Nu0.001.hdf5 --sample-limit 1
```

If `inspect-data` reports ambiguous dataset keys, rerun with:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --dataset-key <key> --sample-limit 1
```

Full mode is opt-in:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --mode full --sample-limit 1
```

Full mode is read-only plumbing validation. Current HDF5 full mode uses target-copy prediction and must not be reported as model performance.
