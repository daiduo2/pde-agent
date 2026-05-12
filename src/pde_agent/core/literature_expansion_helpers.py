"""Literature expansion helper functions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

def _append_unique(memory: dict[str, Any], key: str, value: str) -> None:
    values = memory.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        memory[key] = values
    if value not in values:
        values.append(value)

def _append_unique_value(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)

def _bounded_local_strategy_deadline_seconds(candidate_generator: str | None) -> float:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 300.0
    return 120.0

def _bounded_local_strategy_max_rounds(candidate_generator: str | None) -> int:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 5
    return 1

def _bounded_local_strategy_objective(candidate_generator: str | None) -> str:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return "early_accuracy"
    return "score"

def _bounded_local_strategy_required_reference_steps(
    candidate_generator: str | None,
) -> int | None:
    if candidate_generator == "hypothesis_lab_early_subsegment_patch_search":
        return 200
    return None

def _calibration_weights_from_candidate(candidate: dict[str, Any]) -> list[float]:
    grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        else {}
    )
    values = grid.get("calibration_weights")
    weights = [
        float(item)
        for item in values
        if isinstance(item, int | float) and not isinstance(item, bool)
    ] if isinstance(values, list) else []
    return weights or [0.0, 0.25, 0.5, 1.0]

def _candidate_key(candidate: dict[str, Any]) -> str:
    grounding = candidate.get("source_grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    card_id = grounding.get("algorithm_card_id")
    executor = candidate.get("executor_spec")
    hook = executor.get("unsupported_hook") if isinstance(executor, dict) else None
    family = candidate.get("candidate_family")
    mutation = candidate.get("mutation_evidence")
    mutation_type = (
        mutation.get("mutation_type")
        if isinstance(mutation, dict)
        and isinstance(mutation.get("mutation_type"), str)
        and mutation.get("mutation_type")
        else None
    )
    return "|".join(
        str(part)
        for part in [card_id or "unknown_card", hook or family or "unknown_family"]
        + ([mutation_type] if mutation_type else [])
    )

def _candidate_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("algorithm_candidates")
    if not isinstance(candidates, list):
        raise LiteratureExpansionError("algorithm_candidates must be a list")
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise LiteratureExpansionError("algorithm candidate must be an object")
        result.append(candidate)
    return result

def _candidate_reference_sample_limit(candidate: dict[str, Any] | None) -> int | None:
    if not isinstance(candidate, dict):
        return None
    grid = candidate.get("parameter_grid")
    if not isinstance(grid, dict):
        return None
    values = grid.get("reference_sample_limits")
    if not isinstance(values, list):
        return None
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None

def _candidate_route(
    candidate: dict[str, Any],
    *,
    checkpoint_paths: list[Path],
) -> tuple[str, float]:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return "invalid", 0.0
    cli_command = executor.get("cli_command")
    if (
        cli_command == "blocked_requires_executor_design"
        and executor.get("unsupported_hook") == "fourier_neural_operator_architecture"
        and checkpoint_paths
    ):
        return "checkpoint_adapter", 100.0
    if cli_command == "blocked_requires_executor_design":
        return "executor_design", 10.0
    return "existing_executor", 80.0

def _candidate_satisfies_mutation_constraint(
    candidate: dict[str, Any],
    *,
    constraints: dict[str, Any],
) -> bool:
    mutation_required = {
        item
        for item in constraints.get("mutation_required_for_families", [])
        if isinstance(item, str) and item
    }
    mutation_required_ids = {
        item
        for item in constraints.get("mutation_required_for_algorithm_candidate_ids", [])
        if isinstance(item, str) and item
    }
    family = candidate.get("candidate_family")
    candidate_id = candidate.get("algorithm_candidate_id")
    if (
        (not isinstance(family, str) or family not in mutation_required)
        and candidate_id not in mutation_required_ids
    ):
        return False
    evidence = candidate.get("mutation_evidence")
    if not isinstance(evidence, dict):
        return False
    mutation_type = evidence.get("mutation_type")
    return isinstance(mutation_type, str) and bool(mutation_type.strip())

def _checkpoint_family_report_path(report: dict[str, Any]) -> Path:
    for value in [
        report.get("checkpoint_family_report_path"),
        report.get("artifacts", {}).get("checkpoint_family_report_path")
        if isinstance(report.get("artifacts"), dict)
        else None,
    ]:
        if isinstance(value, str) and value.strip():
            return Path(value)
    raise LiteratureExpansionError(
        "checkpoint family sandbox report missing report path"
    )

def _checkpoint_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segment_score = best.get("segment_score")
    if isinstance(segment_score, dict):
        return {
            key: value
            for key, value in segment_score.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    segments = best.get("segments")
    if isinstance(segments, dict):
        return {
            key: value.get("score")
            for key, value in segments.items()
            if isinstance(value, dict) and isinstance(value.get("score"), int | float)
        }
    return {}

def _expansion_has_viable_actions(expansion: dict[str, Any]) -> bool:
    for key in ["action_count", "executable_count", "proposal_count"]:
        value = expansion.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False

def _extract_reference_shape_steps(message: str) -> int | None:
    match = re.search(r"got\s+\(\s*\d+\s*,\s*(\d+)\s*,\s*\d+\s*\)", message)
    if match is None:
        return None
    return int(match.group(1))

def _failed_candidate_families_for_outcome(
    *,
    candidate_key: str,
    reason_codes: list[str],
) -> list[str]:
    reason_text = " ".join(
        code.lower() for code in reason_codes if isinstance(code, str)
    )
    key_text = candidate_key.lower()
    if (
        "checkpoint_adapter" in reason_text
        or "checkpoint_weight" in reason_text
        or "best_is_persistence" in reason_text
        or (
            "fourier_neural_operator_architecture" in key_text
            and "checkpoint" in reason_text
        )
    ):
        return ["official_checkpoint_family"]
    return []

def _first_bool_grid_value(value: Any) -> bool | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, bool):
            return item
    return None

def _first_float_grid_value(value: Any) -> float | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, int | float) and not isinstance(item, bool):
            return float(item)
    return None

def _first_hdf5_dataset_name(handle: Any) -> str:
    for key in handle.keys():
        node = handle[key]
        if hasattr(node, "shape"):
            return str(key)
    raise LiteratureExpansionError("HDF5 file contains no dataset")

def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default

def _fno_finetune_steps_from_candidate(candidate: dict[str, Any]) -> int:
    grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        else {}
    )
    values = grid.get("fno_finetune_steps")
    if isinstance(values, list):
        numeric = [
            int(item)
            for item in values
            if isinstance(item, int | float) and not isinstance(item, bool)
        ]
        return max(numeric) if numeric else 0
    return 0

def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureExpansionError(f"artifact already exists: {path}")

def _learning_constraint_reason(constraints: dict[str, Any]) -> str:
    reason_codes = constraints.get("reason_codes")
    if isinstance(reason_codes, list):
        for reason in reason_codes:
            if isinstance(reason, str) and reason:
                return reason
    return "learning_constraint_avoid_family"

def _local_strategy_candidate_gate_issues(candidate: dict[str, Any]) -> list[str]:
    issues = []
    value = candidate.get("candidate_gate_issues")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip() and item not in issues:
                issues.append(item.strip())
    return issues

def _local_strategy_loop_report_path(report: dict[str, Any]) -> Path:
    for value in [
        report.get("loop_report_path"),
        report.get("artifacts", {}).get("loop_report_path")
        if isinstance(report.get("artifacts"), dict)
        else None,
    ]:
        if isinstance(value, str) and value.strip():
            return Path(value)
    raise LiteratureExpansionError("local strategy loop missing report path")

def _local_strategy_mutation_axes() -> list[dict[str, Any]]:
    return [
        {
            "id_suffix": "mutation_robust_reference_ablation_v0",
            "mutation_type": "robust_reference_ablation_retry",
            "implementation_hook": "robust_reference_ablation",
            "hypothesis": (
                "The parent local strategy improved a narrow legal-reference "
                "window but failed on a wider window; a robust "
                "reference-ablation retry should prefer smoother handoffs and "
                "require wider-window evidence before promotion."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor, but mark this "
                "candidate as a mutation that must search a wider legal "
                "reference window and report whether each early sub-window "
                "source remains stable under ablation."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "requires_wider_reference_ablation": [True],
            },
            "risk_controls": [
                "mutation must not repeat the narrow-window-only promotion path"
            ],
        },
        {
            "id_suffix": "mutation_source_stability_ablation_v0",
            "mutation_type": "source_stability_ablation_retry",
            "implementation_hook": "source_stability_ablation",
            "hypothesis": (
                "The robust reference-ablation retry did not improve the legal "
                "window; the next test should isolate whether early-window "
                "source switching is unstable across reference samples."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor, but require source "
                "stability diagnostics for each early subsegment before any "
                "handoff patch is accepted."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "source_stability_guard": [True],
            },
            "risk_controls": [
                "mutation must report per-source stability before promotion"
            ],
        },
        {
            "id_suffix": "mutation_handoff_smoothing_ablation_v0",
            "mutation_type": "handoff_smoothing_ablation_retry",
            "implementation_hook": "handoff_smoothing_ablation",
            "hypothesis": (
                "If source stability is not sufficient, the remaining failure "
                "mode may be a discontinuous early-to-middle handoff; a "
                "smoothing ablation should test conservative blend weights."
            ),
            "mechanism": (
                "Reuse the bounded local strategy executor with explicit "
                "handoff blend weights so that the sandbox can measure whether "
                "the early segment improves without degrading the middle "
                "segment."
            ),
            "parameter_grid": {
                "reference_sample_limits": [128],
                "handoff_smoothing_weights": [0.25, 0.5],
            },
            "risk_controls": [
                "mutation must not trade early score for middle-segment regression"
            ],
        },
    ]

def _local_strategy_node_id_from_candidate_id(candidate_id: str) -> str:
    if candidate_id.startswith("candidate_local_strategy_"):
        return "local_strategy_" + candidate_id.removeprefix(
            "candidate_local_strategy_"
        )
    return candidate_id

def _local_strategy_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segment_score = best.get("segment_score")
    if isinstance(segment_score, dict):
        return {
            key: value
            for key, value in segment_score.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    segments = best.get("segments")
    if isinstance(segments, dict):
        return {
            key: value.get("score")
            for key, value in segments.items()
            if isinstance(value, dict) and isinstance(value.get("score"), int | float)
        }
    signals = best.get("objective_signals")
    if isinstance(signals, dict):
        early = signals.get("early_segment_score")
        if isinstance(early, int | float) and not isinstance(early, bool):
            return {"early": float(early)}
    return {}

def _next_command(actions: list[dict[str, Any]], *, skipped_count: int = 0) -> str:
    for action in actions:
        if action.get("executable") is True:
            return str(action["next_command"])
    for action in actions:
        if action.get("strategy_proposal_path"):
            return str(action["next_command"])
    if skipped_count > 0:
        return "build-local-asset-algorithm-candidates"
    return "review-literature-expansion"

def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]

def _queue_artifact_paths(lab_report: dict[str, Any]) -> list[str]:
    artifacts = lab_report.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    paths = []
    for key in ["sandbox_report_path", "best_submission_dir", "best_prediction_path"]:
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    return paths

def _queue_failure_codes(reason_codes: list[str], *, outcome: str) -> list[str]:
    lowered = " ".join(reason_codes).lower()
    if "reference_horizon_incompatible" in lowered:
        return ["data_contract_incompatible"]
    if "unsupported" in lowered or "checkpoint" in lowered:
        return ["unsupported_executor"]
    if "task1retrievalpriorerror" in lowered:
        return ["data_contract_incompatible"]
    if "runtime" in lowered or "timeout" in lowered:
        return ["runtime_exceeded"]
    if outcome == "blocked":
        return ["unsupported_executor"]
    return ["sandbox_rule_failed"]

def _queue_segment_scores(best: dict[str, Any]) -> dict[str, Any]:
    segments = best.get("segments")
    if not isinstance(segments, dict):
        return {}
    return {
        key: value.get("score")
        for key, value in segments.items()
        if isinstance(value, dict) and isinstance(value.get("score"), int | float)
    }

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LiteratureExpansionError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiteratureExpansionError("JSON artifact must be an object")
    return value

def _recovery_plan(
    *,
    actions: list[dict[str, Any]],
    skipped_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if actions or not skipped_candidates:
        return None
    return {
        "reason": "candidate_pool_exhausted",
        "next_command": "build-local-asset-algorithm-candidates",
        "recommended_inputs": {
            "local_strategy_library_path": "runs/local_strategy_library.json",
            "checkpoint_paths": [
                "data/pdebench/checkpoints/burgers_FNO.tar",
                "data/pdebench/checkpoints/burgers_Unet.tar",
            ],
            "max_strategy_candidates": 5,
        },
        "follow_up_command": "run-literature-strategy-expansion",
    }

def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LiteratureExpansionError(f"{label} not found: {path}")
    return path

def _sandbox_queue_next_command(items: list[dict[str, Any]]) -> str:
    if any(item.get("outcome") == "passed" for item in items):
        return "evaluate-expansion-sandbox-queue"
    if any(item.get("outcome") == "failed" for item in items):
        return "summarize-literature-registry"
    if any(item.get("status") == "skipped" for item in items):
        return "review-expansion-sandbox-queue"
    return "run-literature-strategy-expansion"

def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "candidate"

def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]

def _task_id_from_candidate(candidate: dict[str, Any]) -> int:
    value = candidate.get("target_task")
    if isinstance(value, int) and value in {1, 2}:
        return value
    if isinstance(value, str):
        stripped = value.lower().replace("task", "").replace("-", "").strip()
        if stripped in {"1", "2"}:
            return int(stripped)
    return 1

def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureExpansionError(f"{label} must be a non-empty string")
    return value.strip()

def _unique_paths(paths: list[str]) -> list[Path]:
    seen: set[str] = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(Path(path))
    return unique

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
