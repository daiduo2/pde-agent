"""Prepare fixture for FNO training closed loop smoke test.

Sets up a workspace with mock data so the agent can run one iteration
without requiring real HDF5 files or PyTorch.
"""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: prepare_fixture.py <workspace_dir>", file=sys.stderr)
        sys.exit(1)

    workspace = Path(sys.argv[1]).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # Create mock NPZ data (shape [N, T, X])
    np.savez(
        workspace / "data.npz",
        tensor=np.random.randn(4, 20, 64).astype(np.float32),
        context=np.random.randn(4, 10, 64).astype(np.float32),
        target=np.random.randn(4, 12, 64).astype(np.float32),
    )

    # Create a mock source HDF5 file if h5py is available
    try:
        import h5py

        with h5py.File(workspace / "data.h5", "w") as f:
            f.create_dataset("tensor", data=np.random.randn(4, 20, 64).astype(np.float32))
    except ImportError:
        pass

    # Write a mock previous memory to test inter-iteration propagation
    memory = {
        "schema_version": 1,
        "state_hint": "FNO_TRAINING_MEMORY_BUILT",
        "iteration_index": -1,
        "source_optimization_report_path": None,
        "best_so_far": {},
        "trajectory_summary": [],
        "avoid_repeating": [],
        "family_outcomes": {},
        "positive_signals": [],
        "search_guidance": [],
        "convergence_history": {},
        "next_hints": ["start first iteration"],
    }
    (workspace / "previous_memory.json").write_text(
        json.dumps(memory, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"fixture prepared: {workspace}")


if __name__ == "__main__":
    main()
