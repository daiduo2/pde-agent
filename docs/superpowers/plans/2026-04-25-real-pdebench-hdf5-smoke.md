# Real PDEBench HDF5 Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled real PDEBench/Burgers HDF5 local smoke path with explicit single-file download/preflight and portable runtime state generation.

**Architecture:** Keep data acquisition separate from harness execution. A download/preflight script manages one official Burgers HDF5 file under ignored `data/pdebench/burgers/`; a smoke preparation script renders portable templates into runtime config/run_state; existing `inspect-data` and diagnostic commands remain the source of truth for schema and metric validation.

**Tech Stack:** Python 3.12+, stdlib argparse/hashlib/urllib, optional h5py, pytest, existing pde-agent CLI.

---

## File Structure

- Create `scripts/download_pdebench_file.py`: explicit single-file downloader and local preflight helpers.
- Create `scripts/prepare_real_hdf5_smoke.py`: generates runtime config and run_state for inspect/full modes.
- Create `integration_tests/real_hdf5_smoke/README.md`: documents download, preflight, inspect mode, full mode, and target-copy caveat.
- Create `integration_tests/real_hdf5_smoke/base_config.template.json`: portable template for real HDF5 smoke.
- Create `integration_tests/real_hdf5_smoke/override.json`: runtime override metadata.
- Create `integration_tests/real_hdf5_smoke/task_spec.json`: bounded task spec.
- Create `integration_tests/real_hdf5_smoke/bootstrap_prompt.md`: Hermes bootstrap instructions.
- Create `integration_tests/real_hdf5_smoke/prompt_template.md`: Hermes prompt template.
- Create `integration_tests/real_hdf5_smoke/run_state.inspect.template.yaml`: inspect-only generated command template.
- Create `integration_tests/real_hdf5_smoke/run_state.full.template.yaml`: full read-only diagnostic command template.
- Modify `.gitignore`: ignore generated real smoke runtime files and `.part` downloads.
- Create `tests/test_real_hdf5_smoke_scripts.py`: unit tests for downloader/preflight and smoke preparation.

---

### Task 1: Download And Preflight Helpers

**Files:**
- Create: `scripts/download_pdebench_file.py`
- Test: `tests/test_real_hdf5_smoke_scripts.py`

- [ ] **Step 1: Write failing tests for checksum, overwrite protection, and HDF5 preflight**

Create `tests/test_real_hdf5_smoke_scripts.py` with the initial tests:

```python
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _download_module():
    repo_root = Path(__file__).resolve().parents[1]
    return _load_module(
        "download_pdebench_file",
        repo_root / "scripts" / "download_pdebench_file.py",
    )


def test_compute_md5_for_local_file(tmp_path: Path) -> None:
    module = _download_module()
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"burgers")

    assert module.compute_md5(file_path) == hashlib.md5(b"burgers").hexdigest()


def test_refuse_overwrite_without_force(tmp_path: Path) -> None:
    module = _download_module()
    output = tmp_path / "existing.hdf5"
    output.write_bytes(b"old")

    with pytest.raises(SystemExit, match="already exists"):
        module.ensure_can_write(output, force=False)


def test_preflight_detects_md5_mismatch(tmp_path: Path) -> None:
    module = _download_module()
    file_path = tmp_path / "sample.hdf5"
    file_path.write_bytes(b"not-real-hdf5")

    with pytest.raises(SystemExit, match="MD5 mismatch"):
        module.preflight_file(file_path, expected_md5="0" * 32, inspect_hdf5=False)


def test_preflight_reports_tiny_hdf5_keys(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    module = _download_module()
    file_path = tmp_path / "tiny.hdf5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1, 2, 3), dtype=np.float32))

    report = module.preflight_file(file_path, expected_md5=None, inspect_hdf5=True)

    assert report["path"] == str(file_path)
    assert report["exists"] is True
    assert report["hdf5"]["datasets"][0]["key"] == "tensor"
    assert report["hdf5"]["datasets"][0]["shape"] == [1, 2, 3]
    assert report["hdf5"]["datasets"][0]["dtype"] == "float32"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
```

Expected:

```text
FileNotFoundError or ModuleNotFoundError for scripts/download_pdebench_file.py
```

- [ ] **Step 3: Implement download/preflight helper**

Create `scripts/download_pdebench_file.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = ""
DEFAULT_OUTPUT = Path("data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5")
DEFAULT_MD5 = "44cb784d5a07aa2b1c864cabdcf625f9"
DEFAULT_EXPECTED_SIZE_GB = 7.7
CHUNK_SIZE = 1024 * 1024 * 8


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"output already exists, use --force to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output: Path, force: bool) -> dict[str, object]:
    if not url:
        raise SystemExit(
            "No default DaRUS direct-download URL is configured yet. "
            "Use --url after verifying the current DaRUS file endpoint, or download "
            "manually and run --preflight-only."
        )
    ensure_can_write(output, force=force)
    part_path = output.with_suffix(output.suffix + ".part")
    if part_path.exists():
        part_path.unlink()
    started = time.monotonic()
    with urllib.request.urlopen(url) as response, part_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=CHUNK_SIZE)
    part_path.replace(output)
    elapsed = time.monotonic() - started
    return {
        "url": url,
        "output": str(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
    }


def preflight_file(
    path: Path,
    *,
    expected_md5: str | None,
    inspect_hdf5: bool,
) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"file does not exist: {path}")
    actual_md5 = compute_md5(path)
    if expected_md5 is not None and actual_md5 != expected_md5:
        raise SystemExit(
            f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
        )
    report: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "md5": actual_md5,
    }
    if expected_md5 is not None:
        report["expected_md5"] = expected_md5
    if inspect_hdf5:
        report["hdf5"] = inspect_hdf5_metadata(path)
    return report


def inspect_hdf5_metadata(path: Path) -> dict[str, object]:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required for --inspect-hdf5") from exc

    datasets: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def collect(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                datasets.append(
                    {
                        "key": name,
                        "shape": [int(axis) for axis in value.shape],
                        "dtype": str(value.dtype),
                    }
                )

        handle.visititems(collect)
    return {"datasets": datasets}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or preflight one PDEBench Burgers HDF5 file."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-md5", default=DEFAULT_MD5)
    parser.add_argument("--expected-size-gb", type=float, default=DEFAULT_EXPECTED_SIZE_GB)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--inspect-hdf5", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    output = args.output if args.output.is_absolute() else root / args.output

    if not args.preflight_only:
        download_file(args.url, output, force=args.force)

    report = preflight_file(
        output,
        expected_md5=args.expected_md5,
        inspect_hdf5=args.inspect_hdf5,
    )
    report["expected_size_gb"] = args.expected_size_gb
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
```

Expected:

```text
4 passed
```

If `h5py` is unavailable:

```text
3 passed, 1 skipped
```

- [ ] **Step 5: Commit**

```bash
git add scripts/download_pdebench_file.py tests/test_real_hdf5_smoke_scripts.py
git commit -m "feat: add pdebench file preflight"
```

---

### Task 2: Real HDF5 Smoke Templates And Preparation Script

**Files:**
- Create: `scripts/prepare_real_hdf5_smoke.py`
- Create: `integration_tests/real_hdf5_smoke/README.md`
- Create: `integration_tests/real_hdf5_smoke/base_config.template.json`
- Create: `integration_tests/real_hdf5_smoke/override.json`
- Create: `integration_tests/real_hdf5_smoke/task_spec.json`
- Create: `integration_tests/real_hdf5_smoke/bootstrap_prompt.md`
- Create: `integration_tests/real_hdf5_smoke/prompt_template.md`
- Create: `integration_tests/real_hdf5_smoke/run_state.inspect.template.yaml`
- Create: `integration_tests/real_hdf5_smoke/run_state.full.template.yaml`
- Modify: `tests/test_real_hdf5_smoke_scripts.py`

- [ ] **Step 1: Extend tests for smoke preparation**

Append these tests to `tests/test_real_hdf5_smoke_scripts.py`:

```python
def _prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    return _load_module(
        "prepare_real_hdf5_smoke",
        repo_root / "scripts" / "prepare_real_hdf5_smoke.py",
    )


def test_prepare_real_hdf5_smoke_generates_inspect_runtime_files(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    module = _prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "real_hdf5_smoke"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.inspect.template.yaml").write_text(
        "command_prefix: {{COMMAND_PREFIX}}\n"
        "base: {{BASE_CONFIG_RUNTIME}}\n"
        "cmd: {{COMMAND_PREFIX}} inspect-data --run-root {{RUN_ROOT}}\n",
        encoding="utf-8",
    )
    hdf5_path = tmp_path / "tiny.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1, 200, 256), dtype=np.float32))

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key="tensor",
        sample_limit=1,
        mode="inspect",
    )

    runtime_config = json.loads(outputs["base_config_runtime"].read_text())
    run_state = outputs["run_state"].read_text()
    assert runtime_config["data"]["mode"] == "pdebench_hdf5"
    assert runtime_config["data"]["source_path"] == str(hdf5_path)
    assert runtime_config["data"]["dataset_key"] == "tensor"
    assert "inspect-data" in run_state
    assert "{{" not in run_state


def test_prepare_real_hdf5_smoke_full_mode_contains_full_chain(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    module = _prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "real_hdf5_smoke"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.full.template.yaml").write_text(
        "cmd1: {{COMMAND_PREFIX}} inspect-data --run-root {{RUN_ROOT}}\n"
        "cmd2: {{COMMAND_PREFIX}} collect-artifacts --run-root {{RUN_ROOT}}\n",
        encoding="utf-8",
    )
    hdf5_path = tmp_path / "tiny.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("tensor", data=np.zeros((1, 200, 256), dtype=np.float32))

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key=None,
        sample_limit=1,
        mode="full",
    )

    run_state = outputs["run_state"].read_text()
    assert "inspect-data" in run_state
    assert "collect-artifacts" in run_state
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
```

Expected:

```text
FileNotFoundError for scripts/prepare_real_hdf5_smoke.py
```

- [ ] **Step 3: Implement `prepare_real_hdf5_smoke.py`**

Create `scripts/prepare_real_hdf5_smoke.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ID = "real_hdf5_smoke_001"
CONFIG_HASH = "real-hdf5-smoke-001"
DEFAULT_HDF5_PATH = Path("data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5")


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "real_hdf5_smoke"


def build_base_config(
    hdf5_path: Path,
    *,
    dataset_key: str | None,
    sample_limit: int,
) -> dict[str, object]:
    return {
        "model": {
            "name": "none",
            "mode": "read_only_diagnostic",
        },
        "data": {
            "mode": "pdebench_hdf5",
            "problem_id": "PDE_Burgers",
            "source_path": str(hdf5_path),
            "dataset_key": dataset_key,
            "context_steps": 10,
            "forecast_steps": 190,
            "time_reduction": 5,
            "space_reduction": 4,
            "sample_limit": sample_limit,
        },
    }


def render_template(repo_root: Path, mode: str) -> str:
    root = fixture_root(repo_root)
    template_name = f"run_state.{mode}.template.yaml"
    template = (root / template_name).read_text(encoding="utf-8")
    command_prefix = repo_root / ".venv" / "bin" / "pde-agent"
    run_root = root / "runs" / RUN_ID
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{RUN_ROOT}}": str(run_root),
        "{{COMMAND_PREFIX}}": str(command_prefix),
        "{{BASE_CONFIG_RUNTIME}}": str(root / "base_config.runtime.json"),
        "{{CONFIG_HASH}}": CONFIG_HASH,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(
    *,
    repo_root: Path,
    hdf5_path: Path,
    dataset_key: str | None,
    sample_limit: int,
    mode: str,
) -> dict[str, Path]:
    if mode not in {"inspect", "full"}:
        raise SystemExit("mode must be 'inspect' or 'full'")
    if not hdf5_path.exists():
        raise SystemExit(
            f"HDF5 file does not exist: {hdf5_path}. "
            "Place the file under data/pdebench/burgers/, pass --hdf5-path, "
            "or run scripts/download_pdebench_file.py explicitly."
        )
    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    runtime_config = root / "base_config.runtime.json"
    run_state = root / "run_state.yaml"
    runtime_config.write_text(
        json.dumps(
            build_base_config(
                hdf5_path.resolve(),
                dataset_key=dataset_key,
                sample_limit=sample_limit,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_state.write_text(render_template(repo_root, mode), encoding="utf-8")
    return {"base_config_runtime": runtime_config, "run_state": run_state}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real HDF5 smoke runtime files.")
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--dataset-key")
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--mode", choices=["inspect", "full"], default="inspect")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    hdf5_path = args.hdf5_path if args.hdf5_path.is_absolute() else root / args.hdf5_path
    outputs = prepare_runtime(
        repo_root=root,
        hdf5_path=hdf5_path,
        dataset_key=args.dataset_key,
        sample_limit=args.sample_limit,
        mode=args.mode,
    )
    print(f"generated {outputs['base_config_runtime']}")
    print(f"generated {outputs['run_state']}")
    print(f"hdf5 {hdf5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Add real smoke fixture templates**

Create `integration_tests/real_hdf5_smoke/base_config.template.json`:

```json
{
  "model": {
    "name": "none",
    "mode": "read_only_diagnostic"
  },
  "data": {
    "mode": "pdebench_hdf5",
    "problem_id": "PDE_Burgers",
    "source_path": "GENERATED_BY_scripts/prepare_real_hdf5_smoke.py",
    "dataset_key": null,
    "context_steps": 10,
    "forecast_steps": 190,
    "time_reduction": 5,
    "space_reduction": 4,
    "sample_limit": 1
  }
}
```

Create `integration_tests/real_hdf5_smoke/override.json`:

```json
{
  "run": {
    "phase": "real_hdf5_smoke"
  }
}
```

Create `integration_tests/real_hdf5_smoke/task_spec.json`:

```json
{
  "problem_id": "PDE_Burgers",
  "task": "run read-only real HDF5 smoke diagnostics",
  "forbidden": [
    "training",
    "code_patch",
    "submission_build",
    "implicit_download"
  ]
}
```

Create `integration_tests/real_hdf5_smoke/bootstrap_prompt.md`:

```markdown
# Real HDF5 Smoke Protocol

Read the required files in the current directory. Then execute only the commands listed in `run_state.yaml`, in order.

Do not download data. Do not inspect command help. Do not search outside the current directory. Do not patch code. Do not run training. Do not guess missing paths.
```

Create `integration_tests/real_hdf5_smoke/prompt_template.md`:

```markdown
You are running a bounded PDE Agent real HDF5 smoke test.

Required behavior:

- Read `run_state.yaml`.
- Read `bootstrap_prompt.md`.
- Read `task_spec.json`.
- Read `base_config.runtime.json`.
- Read `override.json`.
- Execute the exact allowed commands from `run_state.yaml`.
- Stop after the final allowed command succeeds.

Forbidden behavior:

- Do not download data.
- Do not patch code.
- Do not run training.
- Do not inspect command help.
- Do not search outside the current directory.
- Do not guess missing paths.

Important caveat:

Full mode is read-only plumbing validation. Current HDF5 full mode uses target-copy prediction and must not be reported as model performance.
```

- [ ] **Step 5: Add run_state templates**

Create `integration_tests/real_hdf5_smoke/run_state.inspect.template.yaml`:

```yaml
run_id: {{RUN_ID}}
run_root: {{RUN_ROOT}}
working_directory: {{FIXTURE_ROOT}}
command_prefix: {{COMMAND_PREFIX}}

current_state: TASK_PARSED

allowed_actions:
  - read_run_state
  - read_bootstrap_prompt
  - read_task_spec
  - read_base_config_runtime
  - read_override
  - call_init_run
  - call_resolve_config
  - call_inspect_data

forbidden_actions:
  - patch_code
  - run_training
  - call_build_submission
  - implicit_download
  - guess_missing_paths
  - skip_required_reads
  - search_outside_current_directory
  - inspect_command_help
  - probe_command_surface

required_reads:
  - {{FIXTURE_ROOT}}/run_state.yaml
  - {{FIXTURE_ROOT}}/bootstrap_prompt.md
  - {{FIXTURE_ROOT}}/task_spec.json
  - {{BASE_CONFIG_RUNTIME}}
  - {{FIXTURE_ROOT}}/override.json

required_artifacts:
  - {{RUN_ROOT}}/metrics/data_inspection.json

allowed_commands:
  - name: init_run
    command: {{COMMAND_PREFIX}} init-run --run-id {{RUN_ID}} --runs-root {{FIXTURE_ROOT}}/runs
  - name: resolve_config
    command: {{COMMAND_PREFIX}} resolve-config --run-id {{RUN_ID}} --base-config-file {{BASE_CONFIG_RUNTIME}} --override-file {{FIXTURE_ROOT}}/override.json --task-spec-file {{FIXTURE_ROOT}}/task_spec.json --run-root {{RUN_ROOT}}
  - name: inspect_data
    command: {{COMMAND_PREFIX}} inspect-data --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}

success_transition:
  on_complete: DATA_INSPECTED
```

Create `integration_tests/real_hdf5_smoke/run_state.full.template.yaml` with the same header and required reads, but allowed actions/commands should include:

```yaml
  - call_eval_rollout
  - call_analyze_spectrum
  - call_check_physics
  - call_interpret_error
  - call_select_hypothesis
  - call_record_timing
  - call_collect_artifacts
```

and allowed commands should append:

```yaml
  - name: eval_rollout
    command: {{COMMAND_PREFIX}} eval-rollout --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: analyze_spectrum
    command: {{COMMAND_PREFIX}} analyze-spectrum --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: check_physics
    command: {{COMMAND_PREFIX}} check-physics --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: interpret_error
    command: {{COMMAND_PREFIX}} interpret-error --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: select_hypothesis
    command: {{COMMAND_PREFIX}} select-hypothesis --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: record_timing
    command: {{COMMAND_PREFIX}} record-timing --run-id {{RUN_ID}} --run-root {{RUN_ROOT}}
  - name: collect_artifacts
    command: {{COMMAND_PREFIX}} collect-artifacts --run-id {{RUN_ID}} --run-root {{RUN_ROOT}} --config-hash {{CONFIG_HASH}}
```

Full mode required artifact should be:

```yaml
required_artifacts:
  - {{RUN_ROOT}}/artifacts/manifest.json

success_transition:
  on_complete: HYPOTHESIS_SELECTED
```

- [ ] **Step 6: Add README**

Create `integration_tests/real_hdf5_smoke/README.md`:

````markdown
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
````

- [ ] **Step 7: Run tests**

Run:

```bash
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
```

Expected:

```text
all tests in tests/test_real_hdf5_smoke_scripts.py passed
```

- [ ] **Step 8: Commit**

```bash
git add scripts/prepare_real_hdf5_smoke.py integration_tests/real_hdf5_smoke tests/test_real_hdf5_smoke_scripts.py
git commit -m "feat: add real hdf5 smoke preparation"
```

---

### Task 3: Git Hygiene And End-To-End Tiny HDF5 Verification

**Files:**
- Modify: `.gitignore`
- Modify: `tests/test_real_hdf5_smoke_scripts.py`

- [ ] **Step 1: Add ignore rules**

Append to `.gitignore`:

```gitignore
integration_tests/real_hdf5_smoke/base_config.runtime.json
integration_tests/real_hdf5_smoke/run_state.yaml
integration_tests/real_hdf5_smoke/runs/
data/pdebench/burgers/*.part
```

- [ ] **Step 2: Add end-to-end tiny HDF5 inspect-mode test**

Append to `tests/test_real_hdf5_smoke_scripts.py`:

```python
import subprocess


def test_real_hdf5_inspect_runtime_command_chain(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    repo_root = Path(__file__).resolve().parents[1]
    module = _prepare_module()
    hdf5_path = tmp_path / "tiny_real.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset(
            "tensor",
            data=np.zeros((1, 1000, 1024), dtype=np.float32),
        )

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        hdf5_path=hdf5_path,
        dataset_key="tensor",
        sample_limit=1,
        mode="inspect",
    )

    commands = []
    for line in outputs["run_state"].read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))

    last = ""
    for command in commands:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        last = result.stdout

    assert '"state_hint": "DATA_INSPECTED"' in last
```

- [ ] **Step 3: Run targeted tests**

Run:

```bash
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
```

Expected:

```text
all tests in tests/test_real_hdf5_smoke_scripts.py passed
```

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run --frozen pytest tests -q
git diff --check
git status --short --ignored integration_tests/real_hdf5_smoke data/pdebench/burgers
```

Expected:

```text
all tests passed
git diff --check has no output
generated real_hdf5_smoke runtime files are ignored if present
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/test_real_hdf5_smoke_scripts.py
git commit -m "test: verify real hdf5 smoke chain"
```

---

### Task 4: Documentation Review And Manual Download Readiness

**Files:**
- Modify: `integration_tests/real_hdf5_smoke/README.md`
- Modify: `data/README.md`
- Test: no new test file

- [ ] **Step 1: Update data README with official target**

Add this section to `data/README.md`:

````markdown
## PDEBench Burgers Smoke Target

The first real-data smoke target is:

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
```

Official source:

```text
PDEBench Datasets, DaRUS, doi:10.18419/DARUS-2986
```

Current expected MD5 from DaRUS V8:

```text
44cb784d5a07aa2b1c864cabdcf625f9
```

The file is large, approximately 7.7 GB. Do not commit it.
````

- [ ] **Step 2: Update real smoke README with download/preflight commands**

Ensure `integration_tests/real_hdf5_smoke/README.md` includes:

````markdown
Preflight an existing file:

```bash
uv run --frozen python scripts/download_pdebench_file.py --preflight-only --inspect-hdf5
```

If a stable DaRUS direct URL has been verified, download explicitly:

```bash
uv run --frozen python scripts/download_pdebench_file.py --url '<verified-direct-url>'
```

If no stable direct URL is configured, download the single Burgers file manually from DaRUS and place it at the default target path, then run preflight.
````

- [ ] **Step 3: Verify docs and tests**

Run:

```bash
uv run --frozen pytest tests -q
git diff --check
rg -n "1D_Burgers_Sols_Nu0.001.hdf5|44cb784d5a07aa2b1c864cabdcf625f9|target-copy" data/README.md integration_tests/real_hdf5_smoke/README.md
```

Expected:

```text
tests pass
git diff --check has no output
rg finds the target file, MD5, and target-copy caveat
```

- [ ] **Step 4: Commit**

```bash
git add data/README.md integration_tests/real_hdf5_smoke/README.md
git commit -m "docs: document real hdf5 smoke workflow"
```

---

### Final Verification

After all tasks:

```bash
uv run --frozen pytest tests -q
uv run --frozen pytest tests/test_real_hdf5_smoke_scripts.py -v
git diff --check HEAD~4..HEAD
git status --short
```

Expected:

```text
all tests pass
no whitespace errors
working tree clean except ignored generated runtime files if they were produced
```
