"""Promotion-to-strategy materialization for literature expansion."""

import json
import statistics
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
)
from pde_agent.core.strategy_growth import record_strategy_proposal


EVIDENCE_ACCUMULATION_PLAN_STATE_HINT = "CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED"


class LiteraturePromotionMaterializationError(RuntimeError):
    pass


def record_promoted_algorithm_strategy_proposal(
    *,
    candidate_promotion_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_file = _require_file(
        Path(candidate_promotion_path),
        "candidate promotion",
    )
    promotion = _read_candidate_promotion(promotion_file)
    if promotion.get("decision") != "promote" or not promotion.get(
        "promote_to_strategy_growth"
    ):
        raise LiteraturePromotionMaterializationError(
            "candidate promotion did not approve strategy growth"
        )
    candidates_path = _require_file(
        Path(
            _text(
                promotion.get("algorithm_candidates_path"),
                "algorithm_candidates_path",
            )
        ),
        "algorithm candidates",
    )
    candidates = _read_json(candidates_path)
    candidate_id = _text(
        promotion.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    candidate = _algorithm_candidate_by_id(candidates, candidate_id)
    output_file = Path(output_path)
    if output_file.exists():
        if not force:
            raise LiteraturePromotionMaterializationError(
                f"strategy proposal already exists: {output_file}"
            )
        output_file.unlink()
    return record_strategy_proposal(
        strategy_id=_strategy_id_from_promoted_candidate(candidate),
        title=_promoted_strategy_title(candidate),
        source_type=_strategy_source_type_from_candidate(candidate),
        problem_context=_promoted_strategy_problem_context(candidate, promotion),
        hypothesis=_text(candidate.get("hypothesis"), "candidate hypothesis"),
        mechanism=_text(candidate.get("mechanism"), "candidate mechanism"),
        implementation_plan=_promoted_strategy_implementation_plan(
            candidate=candidate,
            promotion=promotion,
            promotion_path=promotion_file,
        ),
        expected_observables=_promoted_strategy_observables(promotion),
        known_risks=_promoted_strategy_risks(candidate),
        output_path=output_file,
    )


def convert_candidate_promotion_to_strategy_evidence(
    *,
    candidate_promotion_path: str | Path,
    extra_candidate_promotion_paths: list[str | Path] | None = None,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_files = [
        _require_file(Path(candidate_promotion_path), "candidate promotion"),
        *[
            _require_file(Path(path), "candidate promotion")
            for path in (extra_candidate_promotion_paths or [])
        ],
    ]
    promotions = [_read_candidate_promotion(path) for path in promotion_files]
    primary = promotions[0]
    _validate_same_promoted_candidate(promotions)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    weighted_deltas = _promotion_weighted_deltas(promotions)
    failure_count = sum(
        1 for promotion in promotions if promotion.get("decision") != "promote"
    )
    physics_deltas = [
        _float_or_default(_promotion_metrics(promotion).get("physics_residual_delta"), 0.0)
        for promotion in promotions
    ]
    evidence = {
        "schema_version": 1,
        "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
        "evidence_summary_path": str(output_file),
        "source_candidate_promotion_path": str(promotion_files[0]),
        "source_candidate_promotion_paths": [str(path) for path in promotion_files],
        "algorithm_candidate_id": primary.get("algorithm_candidate_id"),
        "candidate_family": primary.get("candidate_family"),
        "evaluation_window_count": len(weighted_deltas),
        "mean_score_delta": statistics.fmean(weighted_deltas),
        "score_delta_std": statistics.stdev(weighted_deltas)
        if len(weighted_deltas) > 1
        else 0.0,
        "long_window_score_delta": min(weighted_deltas),
        "physics_residual_delta": max(physics_deltas) if physics_deltas else 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": failure_count / len(promotions),
        "candidate_promotion_metric_results": [
            _promotion_metrics(promotion) for promotion in promotions
        ],
        "candidate_promotion_reason_codes": [
            _string_list(promotion.get("reason_codes")) for promotion in promotions
        ],
        "sandbox_evidence_paths": _promotion_sandbox_evidence_paths(promotions),
        "next_command": "evaluate-strategy-promotion",
    }
    _write_json(output_file, evidence)
    return evidence


def plan_candidate_evidence_accumulation(
    *,
    strategy_promotion_report_path: str | Path,
    strategy_evidence_summary_path: str | Path | None = None,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    promotion_file = _require_file(
        Path(strategy_promotion_report_path),
        "strategy promotion report",
    )
    promotion = _read_json(promotion_file)
    if promotion.get("state_hint") != "STRATEGY_PROMOTION_EVALUATED":
        raise LiteraturePromotionMaterializationError(
            "strategy promotion report has invalid state_hint"
        )
    evidence_file = _require_file(
        Path(
            strategy_evidence_summary_path
            if strategy_evidence_summary_path is not None
            else _text(promotion.get("evidence_summary_path"), "evidence_summary_path")
        ),
        "strategy evidence summary",
    )
    evidence = _read_json(evidence_file)
    rubric_file = _require_file(
        Path(_text(promotion.get("experiment_rubric_path"), "experiment_rubric_path")),
        "experiment rubric",
    )
    rubric = _read_json(rubric_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    current_windows = max(
        int(_float_or_default(evidence.get("evaluation_window_count"), 0.0)),
        0,
    )
    required_windows = _rubric_min_evaluation_windows(rubric)
    missing_windows = max(required_windows - current_windows, 0)
    needs_accumulation = (
        "insufficient_evaluation_windows" in _string_list(promotion.get("reason_codes"))
        and missing_windows > 0
    )
    plan = {
        "schema_version": 1,
        "state_hint": EVIDENCE_ACCUMULATION_PLAN_STATE_HINT,
        "evidence_accumulation_plan_path": str(output_file),
        "strategy_promotion_report_path": str(promotion_file),
        "strategy_evidence_summary_path": str(evidence_file),
        "experiment_rubric_path": str(rubric_file),
        "strategy_id": promotion.get("strategy_id"),
        "algorithm_candidate_id": evidence.get("algorithm_candidate_id"),
        "decision": "accumulate_more_evidence"
        if needs_accumulation
        else "no_accumulation_required",
        "current_window_count": current_windows,
        "required_window_count": required_windows,
        "missing_window_count": missing_windows if needs_accumulation else 0,
        "source_candidate_promotion_paths": _evidence_source_candidate_promotion_paths(
            evidence
        ),
        "recommended_actions": _evidence_accumulation_actions(
            missing_windows=missing_windows if needs_accumulation else 0,
            algorithm_candidate_id=evidence.get("algorithm_candidate_id"),
            current_windows=current_windows,
        ),
        "next_command": "run-runtime-closed-loop"
        if needs_accumulation
        else promotion.get("next_command", "append-strategy-ledger"),
    }
    _write_json(output_file, plan)
    return plan


def _read_candidate_promotion(path: Path) -> dict[str, Any]:
    promotion = _read_json(path)
    if promotion.get("state_hint") != ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT:
        raise LiteraturePromotionMaterializationError(
            "candidate promotion has invalid state_hint"
        )
    return promotion


def _validate_same_promoted_candidate(promotions: list[dict[str, Any]]) -> None:
    if not promotions:
        raise LiteraturePromotionMaterializationError(
            "at least one candidate promotion is required"
        )
    candidate_id = promotions[0].get("algorithm_candidate_id")
    for promotion in promotions:
        if promotion.get("algorithm_candidate_id") != candidate_id:
            raise LiteraturePromotionMaterializationError(
                "candidate promotion artifacts must reference the same candidate"
            )


def _promotion_weighted_deltas(promotions: list[dict[str, Any]]) -> list[float]:
    deltas: list[float] = []
    for promotion in promotions:
        metrics = _promotion_metrics(promotion)
        delta = _float_or_default(metrics.get("score_delta_vs_current_best"), 0.0)
        count = max(int(_float_or_default(promotion.get("evidence_count"), 1.0)), 1)
        deltas.extend([delta] * count)
    return deltas or [0.0]


def _promotion_metrics(promotion: dict[str, Any]) -> dict[str, Any]:
    metrics = promotion.get("metric_results")
    return metrics if isinstance(metrics, dict) else {}


def _promotion_sandbox_evidence_paths(promotions: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for promotion in promotions:
        for path in _string_list(promotion.get("sandbox_evidence_paths")):
            if path not in paths:
                paths.append(path)
    return paths


def _rubric_min_evaluation_windows(rubric: dict[str, Any]) -> int:
    thresholds = rubric.get("promotion_thresholds")
    if not isinstance(thresholds, dict):
        return 1
    return max(
        int(_float_or_default(thresholds.get("min_evaluation_windows"), 1.0)),
        1,
    )


def _evidence_accumulation_actions(
    *,
    missing_windows: int,
    algorithm_candidate_id: Any,
    current_windows: int,
) -> list[dict[str, Any]]:
    if missing_windows <= 0:
        return []
    return [
        {
            "action": "run_legal_reference_retest",
            "algorithm_candidate_id": algorithm_candidate_id,
            "repeat_count": missing_windows,
            "next_command": "run-runtime-closed-loop",
            "reason": "insufficient_evaluation_windows",
            "suggested_sandbox_args": {
                "reference_sample_limit": _next_evidence_reference_sample_limit(
                    current_windows
                ),
            },
        }
    ]


def _next_evidence_reference_sample_limit(current_windows: int) -> int:
    return max(64 * (max(current_windows, 0) + 1), 64)


def _evidence_source_candidate_promotion_paths(evidence: dict[str, Any]) -> list[str]:
    paths = _string_list(evidence.get("source_candidate_promotion_paths"))
    if paths:
        return paths
    single = evidence.get("source_candidate_promotion_path")
    return [single.strip()] if isinstance(single, str) and single.strip() else []


def _algorithm_candidate_by_id(
    candidates_report: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    candidates = _object_items(candidates_report.get("algorithm_candidates"))
    adapted = candidates_report.get("adapted_candidate")
    if isinstance(adapted, dict):
        candidates.append(adapted)
    for candidate in candidates:
        if candidate.get("algorithm_candidate_id") == candidate_id:
            return candidate
    raise LiteraturePromotionMaterializationError(
        f"algorithm candidate not found: {candidate_id}"
    )


def _strategy_id_from_promoted_candidate(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    source_algorithm_id = None
    if isinstance(source, dict):
        source_algorithm_id = source.get("algorithm_card_id")
    value = (
        source_algorithm_id
        or candidate.get("parent_algorithm_card_id")
        or candidate.get("algorithm_candidate_id")
    )
    text = _text(value, "strategy source id")
    for prefix in [
        "local_strategy_",
        "algorithm_",
        "candidate_local_strategy_",
        "candidate_",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return _slug(text)


def _promoted_strategy_title(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    if isinstance(source, dict):
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return _text(candidate.get("algorithm_candidate_id"), "algorithm_candidate_id")


def _strategy_source_type_from_candidate(candidate: dict[str, Any]) -> str:
    source = candidate.get("source_grounding")
    source_type = source.get("source_type") if isinstance(source, dict) else None
    if source_type in {"local_canonical_skill", "local_library", "local_strategy"}:
        return "local_library"
    if source_type in {"arxiv", "openalex", "github", "paper", "external_retrieval"}:
        return "external_retrieval"
    if candidate.get("candidate_family") == "local_strategy_skill":
        return "local_library"
    return "agent_hypothesis"


def _promoted_strategy_problem_context(
    candidate: dict[str, Any],
    promotion: dict[str, Any],
) -> str:
    target_problem = candidate.get("target_problem") or "PDE_Burgers"
    target_task = candidate.get("target_task") or "task1"
    metrics = promotion.get("metric_results")
    score = metrics.get("score") if isinstance(metrics, dict) else None
    delta = (
        metrics.get("score_delta_vs_current_best")
        if isinstance(metrics, dict)
        else None
    )
    return (
        f"{target_problem} {target_task} candidate promoted by Hypothesis Lab "
        f"with score={score} and score_delta_vs_current_best={delta}."
    )


def _promoted_strategy_implementation_plan(
    *,
    candidate: dict[str, Any],
    promotion: dict[str, Any],
    promotion_path: Path,
) -> dict[str, Any]:
    executor = candidate.get("executor_spec")
    cli_command = executor.get("cli_command") if isinstance(executor, dict) else None
    allowed_commands = [cli_command] if isinstance(cli_command, str) and cli_command else []
    evidence_paths = [
        str(Path(path))
        for path in _string_list(promotion.get("sandbox_evidence_paths"))
    ]
    return {
        "algorithm_candidate_id": candidate.get("algorithm_candidate_id"),
        "candidate_family": candidate.get("candidate_family"),
        "candidate_promotion_path": str(promotion_path),
        "algorithm_candidates_path": promotion.get("algorithm_candidates_path"),
        "sandbox_evidence_paths": evidence_paths,
        "allowed_commands": allowed_commands,
        "implementation_hooks": _string_list(
            executor.get("implementation_hooks") if isinstance(executor, dict) else []
        ),
        "promotion_metric_results": promotion.get("metric_results", {}),
        "promotion_reason_codes": promotion.get("reason_codes", []),
    }


def _promoted_strategy_observables(promotion: dict[str, Any]) -> list[str]:
    metrics = promotion.get("metric_results")
    if not isinstance(metrics, dict):
        metrics = {}
    observables = [
        f"score={metrics.get('score')}",
        f"score_delta_vs_current_best={metrics.get('score_delta_vs_current_best')}",
        f"inference_seconds={metrics.get('inference_seconds')}",
        f"physics_residual_delta={metrics.get('physics_residual_delta')}",
        f"rules_passed={metrics.get('rules_passed')}",
    ]
    return [item for item in observables if not item.endswith("=None")]


def _promoted_strategy_risks(candidate: dict[str, Any]) -> list[str]:
    risks = _string_list(candidate.get("risk_controls"))
    if risks:
        return risks
    return ["promotion may not generalize beyond the sandbox reference split"]


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LiteraturePromotionMaterializationError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteraturePromotionMaterializationError("JSON artifact must be an object")
    return payload


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise LiteraturePromotionMaterializationError(f"{label} not found: {path}")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteraturePromotionMaterializationError(f"artifact already exists: {path}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiteraturePromotionMaterializationError(
            f"{label} must be a non-empty string"
        )
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
