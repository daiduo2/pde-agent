from __future__ import annotations

import json
from pathlib import Path

import numpy as np


RUN_ID = "array_fixture_smoke_001"
CONFIG_HASH = "array-fixture-smoke-001"


def fixture_root(repo_root: Path) -> Path:
    return repo_root / "integration_tests" / "array_fixture_smoke"


def build_base_config(repo_root: Path) -> dict[str, object]:
    root = fixture_root(repo_root)
    return {
        "model": {
            "name": "none",
            "mode": "read_only_diagnostic",
        },
        "data": {
            "mode": "npz_fixture",
            "problem_id": "PDE_Burgers",
            "source_path": str(root / "fixture.npz"),
            "dataset_key": None,
            "context_steps": 2,
            "forecast_steps": 5,
            "time_reduction": 5,
            "space_reduction": 4,
            "sample_limit": None,
        },
    }


def render_run_state(repo_root: Path) -> str:
    root = fixture_root(repo_root)
    command_prefix = repo_root / ".venv" / "bin" / "pde-agent"
    base_config_runtime = root / "base_config.runtime.json"
    template_path = root / "run_state.template.yaml"
    if not template_path.exists():
        template_path = fixture_root(Path(__file__).resolve().parents[2]) / (
            "run_state.template.yaml"
        )
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{REPO_ROOT}}", str(repo_root))
        .replace("{{FIXTURE_ROOT}}", str(root))
        .replace("{{COMMAND_PREFIX}}", str(command_prefix))
        .replace("{{BASE_CONFIG_RUNTIME}}", str(base_config_runtime))
    )


def write_fixture_npz(path: Path) -> None:
    target = np.ones((2, 5, 16), dtype=np.float32)
    prediction = target * 0.85
    np.savez(
        path,
        context=np.zeros((2, 2, 16), dtype=np.float32),
        prediction=prediction,
        target=target,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root = fixture_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)

    fixture_path = root / "fixture.npz"
    runtime_config_path = root / "base_config.runtime.json"
    run_state_path = root / "run_state.yaml"

    write_fixture_npz(fixture_path)
    runtime_config_path.write_text(
        json.dumps(build_base_config(repo_root), indent=2) + "\n",
        encoding="utf-8",
    )
    run_state_path.write_text(render_run_state(repo_root), encoding="utf-8")

    print(f"generated {fixture_path}")
    print(f"generated {runtime_config_path}")
    print(f"generated {run_state_path}")


if __name__ == "__main__":
    main()
