from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_ID = "real_hdf5_baseline_smoke_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_real_hdf5_baseline_smoke"


def render_template(
    *,
    repo_root: Path,
    source_path: Path,
    output_dir: Path,
    dataset_key: str,
    start_index: int,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
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
        "{{START_INDEX}}": str(start_index),
        "{{SAMPLE_LIMIT}}": str(sample_limit),
        "{{CONTEXT_STEPS}}": str(context_steps),
        "{{FORECAST_STEPS}}": str(forecast_steps),
        "{{TIME_REDUCTION}}": str(time_reduction),
        "{{SPACE_REDUCTION}}": str(space_reduction),
        "{{NU}}": f"{nu:g}",
        "{{DT}}": f"{dt:g}",
        "{{DX}}": f"{dx:g}",
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
    start_index: int,
    sample_limit: int,
    context_steps: int,
    forecast_steps: int,
    time_reduction: int,
    space_reduction: int,
    nu: float,
    dt: float,
    dx: float,
) -> dict[str, Path]:
    if not source_path.exists():
        raise SystemExit(f"source HDF5 file does not exist: {source_path}")
    if not dataset_key:
        raise SystemExit("dataset_key must be non-empty")
    if start_index < 0:
        raise SystemExit("start_index must be non-negative")
    if sample_limit <= 0:
        raise SystemExit("sample_limit must be positive")
    if context_steps <= 0:
        raise SystemExit("context_steps must be positive")
    if forecast_steps <= 0:
        raise SystemExit("forecast_steps must be positive")
    if time_reduction <= 0:
        raise SystemExit("time_reduction must be positive")
    if space_reduction <= 0:
        raise SystemExit("space_reduction must be positive")
    if nu < 0:
        raise SystemExit("nu must be non-negative")
    if dt <= 0:
        raise SystemExit("dt must be positive")
    if dx <= 0:
        raise SystemExit("dx must be positive")

    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            source_path=source_path.resolve(),
            output_dir=output_dir.resolve(),
            dataset_key=dataset_key,
            start_index=start_index,
            sample_limit=sample_limit,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            time_reduction=time_reduction,
            space_reduction=space_reduction,
            nu=nu,
            dt=dt,
            dx=dx,
        ),
        encoding="utf-8",
    )
    return {"run_state": run_state}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes real-HDF5 baseline smoke runtime files."
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
            "integration_tests/hermes_real_hdf5_baseline_smoke/runs/"
            "real_hdf5_baseline_smoke_001"
        ),
    )
    parser.add_argument("--dataset-key", default="tensor")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=2)
    parser.add_argument("--context-steps", type=int, default=10)
    parser.add_argument("--forecast-steps", type=int, default=31)
    parser.add_argument("--time-reduction", type=int, default=5)
    parser.add_argument("--space-reduction", type=int, default=4)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dx", type=float, default=1.0)
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
        start_index=args.start_index,
        sample_limit=args.sample_limit,
        context_steps=args.context_steps,
        forecast_steps=args.forecast_steps,
        time_reduction=args.time_reduction,
        space_reduction=args.space_reduction,
        nu=args.nu,
        dt=args.dt,
        dx=args.dx,
    )
    print(f"generated {outputs['run_state']}")
    print(f"source {source_path}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
