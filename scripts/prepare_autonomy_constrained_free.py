from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_ID = "autonomy_constrained_free_001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "hermes_autonomy_constrained_free"


def render_template(
    *,
    repo_root: Path,
    windows_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    max_train_steps: int,
    learning_rate: float,
) -> str:
    root = fixture_root(repo_root)
    template = (root / "run_state.template.yaml").read_text(encoding="utf-8")
    replacements = {
        "{{REPO_ROOT}}": str(repo_root),
        "{{FIXTURE_ROOT}}": str(root),
        "{{RUN_ID}}": RUN_ID,
        "{{WINDOWS_PATH}}": str(windows_path),
        "{{CHECKPOINT_PATH}}": str(checkpoint_path),
        "{{OUTPUT_DIR}}": str(output_dir),
        "{{MAX_TRAIN_STEPS}}": str(max_train_steps),
        "{{LEARNING_RATE}}": f"{learning_rate:g}",
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def prepare_runtime(
    *,
    repo_root: Path,
    windows_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    max_train_steps: int,
    learning_rate: float,
) -> dict[str, Path]:
    if not windows_path.exists():
        raise SystemExit(f"windows file does not exist: {windows_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint file does not exist: {checkpoint_path}")
    if max_train_steps <= 0:
        raise SystemExit("max_train_steps must be positive")
    if learning_rate <= 0:
        raise SystemExit("learning_rate must be positive")

    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    run_state = root / "run_state.yaml"
    run_state.write_text(
        render_template(
            repo_root=repo_root,
            windows_path=windows_path.resolve(),
            checkpoint_path=checkpoint_path.resolve(),
            output_dir=output_dir.resolve(),
            max_train_steps=max_train_steps,
            learning_rate=learning_rate,
        ),
        encoding="utf-8",
    )
    return {"run_state": run_state}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hermes constrained-free autonomy runtime files."
    )
    parser.add_argument(
        "--windows-path",
        type=Path,
        default=Path("/tmp/pde-agent-window-suite-task1-public/windows_f031.npz"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("data/pdebench/checkpoints/burgers_FNO.tar"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pde-agent-hermes-autonomy-constrained-free-20260427-001"),
    )
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path.is_absolute()
        else repo_root / args.checkpoint_path
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    )
    outputs = prepare_runtime(
        repo_root=repo_root,
        windows_path=args.windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
    )
    print(f"generated {outputs['run_state']}")
    print(f"windows {args.windows_path}")
    print(f"checkpoint {checkpoint_path}")
    print(f"output_dir {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
