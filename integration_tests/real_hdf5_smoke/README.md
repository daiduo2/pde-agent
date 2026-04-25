# Real HDF5 Smoke

This fixture validates a local real PDEBench/Burgers HDF5 file without committing data or running training.

Default target:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

Preflight an existing file at the default target:

```bash
uv run --frozen python scripts/download_pdebench_file.py --preflight-only --inspect-hdf5
```

To download with a verified stable direct URL:

```bash
uv run --frozen python scripts/download_pdebench_file.py --url '<verified-direct-url>' --resume --progress
```

If no stable direct URL is configured, download the single Burgers file manually from DaRUS, place it at the default target path, then run the preflight command above.

Prepare inspect-mode runtime files:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --sample-limit 1
```

The official PDEBench training file has raw shape `[10000, 201, 1024]`.
With `time_reduction=5`, it only supports 41 reduced time points. For that
short-file smoke, use:

```bash
uv run --frozen python scripts/prepare_real_hdf5_smoke.py --dataset-key tensor --sample-limit 1 --forecast-steps 31
```

The default `--forecast-steps 190` is reserved for the competition long-time
file with enough reduced time points.

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
