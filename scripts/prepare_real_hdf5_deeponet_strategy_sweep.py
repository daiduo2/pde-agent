from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_ID = "real_hdf5_deeponet_strategy_sweep_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_real_hdf5_deeponet_strategy_sweep"


def render_template(
    *,
    repo_root: Path,
    source_path: Path,
    output_dir: Path,
    dataset_key: str,
    sample_limit: int,
    forecast_steps: int,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{SOURCE_PATH}}": str(source_path),
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{DATASET_KEY}}": dataset_key,
        "{{SAMPLE_LIMIT}}": str(sample_limit),
        "{{FORECAST_STEPS}}": str(forecast_steps),
        "{{FORECAST_STEPS_PADDED}}": f"{forecast_steps:03d}",
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(
    *,
    repo_root: Path,
    source_path: Path,
    output_dir: Path,
    dataset_key: str,
    sample_limit: int,
    forecast_steps: int,
) -> dict[str, Path]:
    if not source_path.exists():
        raise SystemExit(f"source HDF5 file does not exist: {source_path}")
    if not dataset_key:
        raise SystemExit("dataset_key must be non-empty")
    if sample_limit < 2:
        raise SystemExit("sample_limit must be at least 2 for start-index canary")
    if forecast_steps <= 0:
        raise SystemExit("forecast_steps must be positive")

    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir.resolve()
    baseline_dir = resolved_output_dir / "baseline"
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            source_path=source_path.resolve(),
            output_dir=resolved_output_dir,
            dataset_key=dataset_key,
            sample_limit=sample_limit,
            forecast_steps=forecast_steps,
        ),
        encoding="utf-8",
    )
    return {
        "run_state": run_state,
        "baseline_windows_path": (
            baseline_dir / f"windows_f{forecast_steps:03d}.npz"
        ),
        "baseline_score_path": baseline_dir / "persistence_score.json",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes real-HDF5 DeepONet strategy-sweep runtime files."
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "integration_tests/hermes_real_hdf5_deeponet_strategy_sweep/runs/"
            "real_hdf5_deeponet_strategy_sweep_001"
        ),
    )
    parser.add_argument("--dataset-key", default="tensor")
    parser.add_argument("--sample-limit", type=int, default=2)
    parser.add_argument("--forecast-steps", type=int, default=31)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    source_path = (
        args.source_path if args.source_path.is_absolute() else repo_root / args.source_path
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    )
    outputs = prepare_runtime(
        repo_root=repo_root,
        source_path=source_path,
        output_dir=output_dir,
        dataset_key=args.dataset_key,
        sample_limit=args.sample_limit,
        forecast_steps=args.forecast_steps,
    )
    print(f"generated {outputs['run_state']}")
    print(f"source {source_path}")
    print(f"baseline_windows_path {outputs['baseline_windows_path']}")
    print(f"baseline_score_path {outputs['baseline_score_path']}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
