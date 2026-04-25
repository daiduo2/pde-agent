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
    source = str(hdf5_path)
    return {
        "model": {
            "name": "none",
            "mode": "read_only_diagnostic",
        },
        "data": {
            "mode": "pdebench_hdf5",
            "problem_id": "PDE_Burgers",
            "source": source,
            "source_path": source,
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
    template = (root / f"run_state.{mode}.template.yaml").read_text(encoding="utf-8")
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
    base_config_runtime = root / "base_config.runtime.json"
    run_state = root / "run_state.yaml"
    base_config_runtime.write_text(
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
    return {"base_config_runtime": base_config_runtime, "run_state": run_state}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real HDF5 smoke runtime files.")
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--dataset-key")
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--mode", choices=["inspect", "full"], default="inspect")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    hdf5_path = (
        args.hdf5_path
        if args.hdf5_path.is_absolute()
        else repo_root / args.hdf5_path
    )
    outputs = prepare_runtime(
        repo_root=repo_root,
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
