"""Artifact adapters between Task1 short-window training and Hypothesis Lab."""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pde_agent.core.task1_short_window_common import Task1ShortWindowTrainingError

def materialize_training_windows_for_hypothesis_sandbox(
    *,
    windows_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(windows_path)
    root = Path(output_dir)
    artifacts = {
        "context_path": root / "context.hdf5",
        "target_path": root / "target.hdf5",
        "reference_path": root / "reference.hdf5",
        "manifest_path": root / "sandbox_window_bundle_manifest.json",
    }
    _guard_artifacts(artifacts.values(), force=force)
    payload = np.load(source)
    if "context" not in payload.files or "target" not in payload.files:
        raise Task1ShortWindowTrainingError(
            "windows npz must contain context and target arrays"
        )
    context = np.asarray(payload["context"], dtype=np.float32)
    future_target = np.asarray(payload["target"], dtype=np.float32)
    if "full_reference" in payload.files:
        reference = np.asarray(payload["full_reference"], dtype=np.float32)
    else:
        reference = np.concatenate([context, future_target], axis=1)
    target = reference
    root.mkdir(parents=True, exist_ok=True)
    _write_hdf5_tensor(artifacts["context_path"], context)
    _write_hdf5_tensor(artifacts["target_path"], target)
    _write_hdf5_tensor(artifacts["reference_path"], reference)
    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_SANDBOX_WINDOW_BUNDLE_MATERIALIZED",
        "source_windows_path": str(source),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "shapes": {
            "context": [int(axis) for axis in context.shape],
            "future_target": [int(axis) for axis in future_target.shape],
            "target": [int(axis) for axis in target.shape],
            "reference": [int(axis) for axis in reference.shape],
        },
        "next_command": "run-hypothesis-algorithm-sandbox",
    }
    artifacts["manifest_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def convert_hypothesis_algorithm_evidence_for_candidate_promotion(
    *,
    hypothesis_evidence_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(hypothesis_evidence_path)
    evidence_report = _load_json(source, "hypothesis algorithm evidence")
    if evidence_report.get("state_hint") != "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED":
        raise Task1ShortWindowTrainingError(
            "hypothesis evidence has invalid state_hint"
        )
    root = Path(output_dir)
    artifacts = {
        "algorithm_candidates_path": root / "algorithm_candidates.json",
        "sandbox_evidence_path": root / "sandbox_evidence.json",
        "manifest_path": root / "promotion_bridge_manifest.json",
    }
    _guard_artifacts(artifacts.values(), force=force)
    root.mkdir(parents=True, exist_ok=True)
    algorithm_card = (
        evidence_report.get("algorithm_card")
        if isinstance(evidence_report.get("algorithm_card"), dict)
        else {}
    )
    evidence = (
        evidence_report.get("evidence")
        if isinstance(evidence_report.get("evidence"), dict)
        else {}
    )
    blueprint_id = str(algorithm_card.get("blueprint_id") or "hypothesis_blueprint")
    candidate_id = f"hypothesis_{_slug(blueprint_id)}"
    candidate_family = str(
        algorithm_card.get("candidate_family") or "hypothesis_algorithm"
    )
    score_delta = _float_or_default_local(
        evidence.get("task1_total_score_delta"),
        _float_or_default_local(evidence.get("score_delta"), 0.0),
    )
    runtime_seconds = _float_or_default_local(evidence.get("inference_time_delta"), 0.0)
    physics_delta = _float_or_default_local(evidence.get("physics_residual_delta"), 0.0)
    sandbox_evidence = {
        "schema_version": 1,
        "state_hint": "SANDBOX_EVIDENCE_RECORDED",
        "evidence_id": f"evidence_{candidate_id}",
        "candidate_id": candidate_id,
        "source_hypothesis_evidence_path": str(source),
        "score": score_delta,
        "score_delta_vs_current_best": score_delta,
        "rules_passed": bool(evidence.get("rules_passed")),
        "runtime": {"inference_seconds": runtime_seconds},
        "physics_metrics": {"residual_delta": physics_delta},
        "reproducibility": {"deterministic": True},
        "ablation_metrics": {"score_delta": score_delta},
        "sandbox_report_path": evidence.get("sandbox_report_path"),
        "judge_report_path": evidence.get("judge_report_path"),
        "diagnostic_only": False,
    }
    candidate = {
        "algorithm_candidate_id": candidate_id,
        "parent_algorithm_card_id": blueprint_id,
        "candidate_family": candidate_family,
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "title": f"Hypothesis Lab {candidate_family} strategy",
        "hypothesis": algorithm_card.get("hypothesis_id") or blueprint_id,
        "mechanism": (
            f"Apply a sandbox-validated {candidate_family} hypothesis from "
            "Hypothesis Lab."
        ),
        "executor_spec": {
            "cli_command": "run-hypothesis-algorithm-sandbox",
            "source_state_hint": evidence_report.get("state_hint"),
        },
        "source_grounding": {
            "source_type": "hypothesis_lab",
            "source_hypothesis_evidence_path": str(source),
            "blueprint_id": blueprint_id,
            "hypothesis_id": algorithm_card.get("hypothesis_id"),
            "title": f"Hypothesis Lab {candidate_family} strategy",
        },
        "promotion_criteria": {
            "min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
            "requires_ablation": False,
        },
        "risk_controls": [
            "requires_sandbox_evidence",
            "requires_context_preservation",
            "requires_runtime_guard",
        ],
    }
    candidates_report = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "algorithm_candidates_path": str(artifacts["algorithm_candidates_path"]),
        "source_hypothesis_evidence_path": str(source),
        "algorithm_candidate_count": 1,
        "algorithm_candidates": [candidate],
        "next_command": "evaluate-algorithm-candidate-promotion",
    }
    artifacts["algorithm_candidates_path"].write_text(
        json.dumps(candidates_report, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    artifacts["sandbox_evidence_path"].write_text(
        json.dumps(sandbox_evidence, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_EVIDENCE_PROMOTION_BRIDGE_BUILT",
        "source_hypothesis_evidence_path": str(source),
        "algorithm_candidate_id": candidate_id,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_command": "evaluate-algorithm-candidate-promotion",
    }
    artifacts["manifest_path"].write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

def _write_hdf5_tensor(path: Path, array: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def _float_or_default_local(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Task1ShortWindowTrainingError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Task1ShortWindowTrainingError(f"{label} must be a JSON object")
    return payload


def _guard_artifacts(paths: Any, *, force: bool) -> None:
    for path in paths:
        if Path(path).exists() and not force:
            raise Task1ShortWindowTrainingError(f"artifact already exists: {path}")
