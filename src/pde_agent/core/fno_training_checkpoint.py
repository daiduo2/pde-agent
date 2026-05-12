"""Checkpoint save/resume for FNO training closed loops."""

import json
from pathlib import Path
from typing import Any


STATE_HINT = "FNO_TRAINING_CHECKPOINTED"


class FnoTrainingCheckpointError(RuntimeError):
    pass


def checkpoint_training_loop(
    *,
    workspace_dir: str | Path,
    iteration_index: int,
    memory: dict[str, Any],
    convergence: dict[str, Any],
    best_config: dict[str, Any] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    force: bool = False,
) -> Path:
    root = Path(workspace_dir)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fno_training_checkpoint_{iteration_index:04d}.json"
    if checkpoint_path.exists() and not force:
        raise FnoTrainingCheckpointError(f"checkpoint already exists: {checkpoint_path}")
    payload = {
        "checkpoint_version": 1,
        "state_hint": STATE_HINT,
        "iteration_index": iteration_index,
        "memory": memory,
        "convergence": convergence,
        "best_config": best_config,
        "trajectory": trajectory or [],
        "timestamp": _iso_now(),
    }
    checkpoint_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def resume_training_loop(
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FnoTrainingCheckpointError(f"checkpoint not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FnoTrainingCheckpointError(f"invalid checkpoint JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FnoTrainingCheckpointError("checkpoint must be a JSON object")
    if value.get("checkpoint_version") != 1:
        raise FnoTrainingCheckpointError("unsupported checkpoint version")
    return {
        "iteration_index": value.get("iteration_index", 0),
        "memory": value.get("memory", {}),
        "convergence": value.get("convergence", {}),
        "best_config": value.get("best_config"),
        "trajectory": value.get("trajectory", []),
        "timestamp": value.get("timestamp"),
        "checkpoint_path": str(path),
    }


def list_training_checkpoints(
    *,
    workspace_dir: str | Path,
) -> list[Path]:
    root = Path(workspace_dir) / "checkpoints"
    if not root.is_dir():
        return []
    checkpoints = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix == ".json" and path.stem.startswith("fno_training_checkpoint_")
        ],
        key=lambda p: p.name,
    )
    return checkpoints


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
