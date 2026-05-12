"""Literature algorithm lab helpers functions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
def _find_by_id(value: Any, selected_id: str, id_key: str) -> dict[str, Any]:
    for item in _object_list(value, "items"):
        if item.get(id_key) == selected_id:
            return item
    raise LiteratureAlgorithmLabError(f"{id_key} not found: {selected_id}")

def _operator_relevant_context_units(
    context_units: list[dict[str, Any]],
    *,
    keywords: list[str],
) -> list[dict[str, Any]]:
    selected = []
    for unit in context_units:
        title = str(unit.get("section_title") or "").lower()
        text = str(unit.get("text") or "").lower()
        haystack = f"{title}\n{text}"
        if any(keyword in haystack for keyword in keywords):
            selected.append(unit)
    return selected or context_units[: min(5, len(context_units))]

def _executor_spec(cli_command: str) -> dict[str, Any]:
    if cli_command == "run-checkpoint-family-sandbox":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        }
    if cli_command == "execute-fno-finetune-candidate":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--candidate-plan-path",
                "--windows-path",
                "--checkpoint-path",
                "--max-train-steps",
            ],
        }
    return {
        "cli_command": cli_command,
        "required_options": [
            "--blueprint-path",
            "--test-path",
            "--target-path",
            "--output-dir",
        ],
    }

def _default_parameter_grid(candidate_family: str) -> dict[str, Any]:
    if candidate_family == "official_checkpoint_family":
        return {
            "calibration_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "fno_finetune_steps": [0, 1],
        }
    if candidate_family == "lightweight_residual_corrector":
        return {"residual_shrinkage": [0.25, 0.5, 0.75], "horizon_decay": [40.0, 80.0]}
    if candidate_family == "spectral_rollout_stabilization":
        return {"spectral_strength": [0.25, 0.5, 1.0], "cutoff_ratio": [0.5, 0.75]}
    if candidate_family == "physics_residual_regularization":
        return {"physics_weight": [0.001, 0.01, 0.1], "spectral_weight": [0.0, 0.1]}
    if candidate_family == "scheduled_rollout_training":
        return {"rollout_steps": [1, 2, 4], "teacher_forcing_ratio": [0.5, 0.75]}
    if candidate_family == "operator_correction_postprocess":
        return {
            "correction_strength": [0.25, 0.5, 0.75],
            "spectral_strength": [0.0, 0.5],
        }
    return {"candidate_enabled": [True]}

def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteratureAlgorithmLabError(f"JSON artifact must be an object: {path}")
    return payload

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureAlgorithmLabError(f"output file already exists: {path}")

def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LiteratureAlgorithmLabError(f"{label} not found: {path}")
    return path

def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()

def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result
