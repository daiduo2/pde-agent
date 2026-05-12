"""Local asset AlgorithmCandidate construction for literature expansion."""

import json
from pathlib import Path
from typing import Any


LOCAL_ASSET_CANDIDATES_STATE_HINT = "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT"


class LiteratureLocalAssetCandidateError(RuntimeError):
    pass


def build_local_asset_algorithm_candidates(
    *,
    local_strategy_library_path: str | Path,
    checkpoint_paths: list[str | Path] | None,
    output_path: str | Path,
    max_strategy_candidates: int,
    force: bool,
) -> dict[str, Any]:
    if max_strategy_candidates < 0:
        raise LiteratureLocalAssetCandidateError("max_strategy_candidates must be >= 0")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    library = _read_json(Path(local_strategy_library_path))
    if library.get("state_hint") != "LOCAL_STRATEGY_LIBRARY_BUILT":
        raise LiteratureLocalAssetCandidateError(
            "local strategy library has invalid state_hint"
        )
    entries = library.get("entries")
    if not isinstance(entries, list):
        raise LiteratureLocalAssetCandidateError(
            "local strategy library entries must be a list"
        )
    candidates = []
    for checkpoint_path in checkpoint_paths or []:
        checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint")
        candidates.append(_checkpoint_asset_candidate(checkpoint_file))
    ranked_entries = sorted(
        [entry for entry in entries if isinstance(entry, dict)],
        key=lambda entry: (
            int(entry.get("priority", 999)),
            str(entry.get("strategy_id", "")),
        ),
    )
    for entry in ranked_entries[:max_strategy_candidates]:
        candidates.append(_local_strategy_asset_candidate(entry))
    if not candidates:
        raise LiteratureLocalAssetCandidateError(
            "local asset candidate generation produced no candidates"
        )
    report = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": LOCAL_ASSET_CANDIDATES_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "local_strategy_library_path": str(local_strategy_library_path),
        "algorithm_candidate_count": len(candidates),
        "algorithm_candidates": candidates,
        "next_command": "run-literature-strategy-expansion",
    }
    _write_json(output_file, report)
    return report


def checkpoint_family(checkpoint_file: Path) -> str:
    name = checkpoint_file.name.lower()
    if "unet" in name:
        return "UNet"
    if "fno" in name:
        return "FNO"
    return "NeuralOperator"


def _checkpoint_asset_candidate(checkpoint_file: Path) -> dict[str, Any]:
    family = checkpoint_family(checkpoint_file)
    candidate_id = f"candidate_local_checkpoint_{_slug(checkpoint_file.stem)}"
    return {
        "algorithm_candidate_id": candidate_id,
        "parent_algorithm_card_id": f"local_checkpoint_{_slug(checkpoint_file.stem)}",
        "candidate_family": "official_checkpoint_family",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "checkpoint",
        "hypothesis": (
            f"Evaluate local {family} checkpoint as a bounded candidate family "
            "after literature candidates are exhausted."
        ),
        "mechanism": (
            "Route the local checkpoint through the audited checkpoint-family "
            "sandbox and compare calibration weights against persistence."
        ),
        "executor_spec": {
            "cli_command": "run-checkpoint-family-sandbox",
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        },
        "parameter_grid": {
            "calibration_weights": [0.0, 0.25, 0.5, 1.0],
            "fno_finetune_steps": [0],
        },
        "resource_bindings": {"checkpoint_paths": [str(checkpoint_file)]},
        "benchmark_plan": {
            "judge": "local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 1,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "checkpoint_adapter_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": [
            "preserve first 10 context steps",
            "verify checkpoint tensor convention before promotion",
        ],
        "source_grounding": {
            "source_ids": [str(checkpoint_file)],
            "algorithm_card_id": f"local_checkpoint_{_slug(checkpoint_file.stem)}",
        },
    }


def _local_strategy_asset_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    strategy_id = _text(entry.get("strategy_id"), "strategy_id")
    hooks = _string_list(entry.get("implementation_hooks"))
    cli_command = _preferred_cli_hook(hooks)
    candidate = {
        "algorithm_candidate_id": f"candidate_local_strategy_{_slug(strategy_id)}",
        "parent_algorithm_card_id": f"local_strategy_{_slug(strategy_id)}",
        "candidate_family": "local_strategy_skill",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": _primary_model_family(entry),
        "hypothesis": _text(entry.get("hypothesis"), "hypothesis"),
        "mechanism": _text(entry.get("mechanism"), "mechanism"),
        "executor_spec": {
            "cli_command": cli_command,
            "required_options": [],
            "implementation_hooks": hooks,
        },
        "parameter_grid": {"candidate_enabled": [True]},
        "benchmark_plan": {
            "judge": "local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "local_strategy_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": _string_list(entry.get("known_risks")),
        "source_grounding": {
            "source_ids": [strategy_id],
            "algorithm_card_id": f"local_strategy_{_slug(strategy_id)}",
            "source_type": entry.get("source_type"),
            "title": entry.get("title"),
        },
    }
    mutation_evidence = entry.get("mutation_evidence")
    if isinstance(mutation_evidence, dict):
        candidate["mutation_evidence"] = mutation_evidence
    return candidate


def _preferred_cli_hook(hooks: list[str]) -> str:
    preferred = [
        "run-hypothesis-lab",
        "run-operator-tuning-loop",
        "run-burgers-spectral-viscosity-benchmark",
        "execute-fno-finetune-candidate",
        "predict-fno-competition",
        "predict-unet-competition",
    ]
    for hook in preferred:
        if hook in hooks:
            return hook
    return hooks[0] if hooks else "review-local-strategy-candidate"


def _primary_model_family(entry: dict[str, Any]) -> str:
    families = _string_list(entry.get("model_families"))
    return families[0] if families else "all"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LiteratureLocalAssetCandidateError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteratureLocalAssetCandidateError("JSON artifact must be an object")
    return payload


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise LiteratureLocalAssetCandidateError(f"{label} not found: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureLocalAssetCandidateError(f"artifact already exists: {path}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiteratureLocalAssetCandidateError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        cleaned.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)
