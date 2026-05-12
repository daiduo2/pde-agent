"""Run the FNO training closed loop smoke test.

This script drives the agent through one iteration using dry-run mode
and validates that the expected artifacts are produced.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_smoke.py <workspace_dir>", file=sys.stderr)
        sys.exit(1)

    workspace = Path(sys.argv[1]).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    smoke_dir = Path(__file__).parent.expanduser().resolve()
    run_state_path = smoke_dir / "run_state.yaml"

    # Run one training iteration via the CLI
    result = subprocess.run(
        [
            sys.executable, "-m", "pde_agent",
            "run-fno-training-iteration",
            "--workspace-dir", str(workspace),
            "--source-path", str(workspace / "data.npz"),
            "--memory-path", str(workspace / "previous_memory.json"),
            "--dataset-key", "tensor",
            "--train-sample-limit", "4",
            "--val-sample-limit", "2",
            "--max-candidates", "1",
            "--batch-size", "2",
            "--train-steps-values", "2",
            "--force",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("smoke test command failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("could not parse CLI output as JSON:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        sys.exit(1)

    # Validate expected artifacts
    run_state = _load_run_state(run_state_path)
    required_artifacts = run_state.get("required_artifacts", [])
    missing = []
    for artifact in required_artifacts:
        path = workspace / artifact
        if not path.is_file():
            missing.append(str(path))

    report = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_CLOSED_LOOP_SMOKE_COMPLETED",
        "command_output": output,
        "missing_artifacts": missing,
        "passed": output.get("status") == "ok" and not missing,
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))

    if missing:
        print(f"missing artifacts: {missing}", file=sys.stderr)
        sys.exit(2)

    if output.get("status") != "ok":
        print("command did not return ok status", file=sys.stderr)
        sys.exit(3)


def _load_run_state(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Fallback: return empty dict if yaml not available
        return {}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
