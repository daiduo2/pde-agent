"""Failure-memory driven hypothesis-to-algorithm designer core."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DESIGN_STATE_HINT = "HYPOTHESIS_ALGORITHM_DESIGN_BUILT"
VALIDATION_STATE_HINT = "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"
FORBIDDEN_DATA = [
    "task1_target_as_prediction_source",
    "local_validation_target_as_training_reference_for_submission",
]
SANDBOX_PLAN_STATE_HINT = "HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED"
SUPPORTED_EXECUTOR_ROUTES = {
    "uncertainty_gating": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_numpy_postprocess",
    },
    "residual_correction": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_numpy_residual_operator",
    },
    "data_retrieval_nonlocal": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_retrieval_alignment",
    },
    "time_window_decomposition": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_window_router",
    },
    "physics_constraint": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_physics_gate",
    },
    "operator_structure": {
        "executor_route": "run-blueprint-sandbox",
        "executor_profile": "bounded_low_rank_operator",
    },
}


class HypothesisAlgorithmDesignerError(RuntimeError):
    pass


def build_hypothesis_algorithm_design(
    *,
    failure_memory_paths: list[str | Path],
    evidence_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    strategy_library_paths: list[str | Path] | None = None,
    literature_card_paths: list[str | Path] | None = None,
    output_path: str | Path | None,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    memories = [_load_json(Path(path), "failure_memory") for path in failure_memory_paths]
    evidence_cards = [
        _load_json(Path(path), "evidence") for path in evidence_paths or []
    ]
    logs = [_read_text(Path(path)) for path in log_paths or []]
    strategy_libraries = [
        _load_json(Path(path), "strategy_library") for path in strategy_library_paths or []
    ]
    literature_cards = [
        _load_json(Path(path), "literature_card") for path in literature_card_paths or []
    ]
    diagnosis = _diagnose_failure_corpus(
        memories=memories,
        evidence_cards=evidence_cards,
        logs=logs,
        strategy_libraries=strategy_libraries,
        literature_cards=literature_cards,
    )
    hypotheses = _generate_mechanism_hypotheses(diagnosis)
    blueprints = [
        _algorithm_blueprint(index=index, hypothesis=hypothesis)
        for index, hypothesis in enumerate(hypotheses, start=1)
    ]
    diversity_audit = _diversity_audit(hypotheses)
    rubric = _designer_rubric(
        hypotheses=hypotheses,
        blueprints=blueprints,
        diversity_audit=diversity_audit,
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
    )
    report = {
        "schema_version": 1,
        "state_hint": DESIGN_STATE_HINT,
        "diagnosis": diagnosis,
        "mechanism_hypotheses": hypotheses,
        "algorithm_blueprints": blueprints,
        "diversity_audit": diversity_audit,
        "designer_rubric": rubric,
        "next_command": "validate-hypothesis-algorithm-design",
    }
    _write_optional_json(output_file, report)
    return report


def validate_hypothesis_algorithm_design(
    *,
    design_path: str | Path,
    output_path: str | Path | None,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    design = _load_json(Path(design_path), "hypothesis_algorithm_design")
    if design.get("state_hint") != DESIGN_STATE_HINT:
        raise HypothesisAlgorithmDesignerError("design has invalid state_hint")
    hypotheses = [
        item for item in design.get("mechanism_hypotheses", []) if isinstance(item, dict)
    ]
    blueprints = [
        item for item in design.get("algorithm_blueprints", []) if isinstance(item, dict)
    ]
    violations = _design_violations(
        hypotheses=hypotheses,
        blueprints=blueprints,
        diversity_audit=design.get("diversity_audit"),
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
    )
    report = {
        "schema_version": 1,
        "state_hint": VALIDATION_STATE_HINT,
        "design_path": str(design_path),
        "hypothesis_count": len(hypotheses),
        "mechanism_family_count": len(_mechanism_families(hypotheses)),
        "blueprint_count": len(blueprints),
        "violations": violations,
        "decision": "reject" if violations else "accept",
        "next_command": "plan-algorithm-sandbox"
        if not violations
        else "repair-hypothesis-algorithm-design",
    }
    _write_optional_json(output_file, report)
    return report


def record_hypothesis_algorithm_evidence(
    *,
    design_path: str | Path,
    sandbox_report_path: str | Path,
    output_path: str | Path | None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    design = _load_json(Path(design_path), "hypothesis_algorithm_design")
    sandbox = _load_json(Path(sandbox_report_path), "sandbox_report")
    blueprint_id = _source_blueprint_id_from_sandbox(sandbox)
    blueprint = _find_blueprint(design, blueprint_id)
    if blueprint is None:
        raise HypothesisAlgorithmDesignerError(
            f"blueprint not found in design: {blueprint_id}"
        )
    family = str(blueprint.get("candidate_family", "unknown"))
    evidence_summary = _sandbox_evidence_summary(sandbox)
    rubric = _promotion_rubric_decision(
        evidence=evidence_summary,
        mechanism_family=family,
    )
    decision = rubric["decision"]
    failure_codes = rubric["failure_codes"]
    evidence_status = decision
    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
        "design_path": str(design_path),
        "sandbox_report_path": str(sandbox_report_path),
        "decision": decision,
        "algorithm_card": {
            "blueprint_id": blueprint_id,
            "hypothesis_id": blueprint.get("hypothesis_id"),
            "candidate_family": family,
            "evidence_status": evidence_status,
        },
        "evidence": {
            "blueprint_id": blueprint_id,
            "judge_report_path": _sandbox_artifact_value(
                sandbox,
                "judge_report_path",
            ),
            **evidence_summary,
            "sandbox_report_path": str(sandbox_report_path),
        },
        "failure_taxonomy": {
            "failure_codes": failure_codes,
            "mechanism_family": family,
            "primary_failure_axis": evidence_summary["primary_failure_axis"],
        },
        "next_generation_constraints": rubric["next_generation_constraints"],
        "promotion_rubric": rubric,
        "next_command": rubric["next_command"],
    }
    _write_optional_json(output_file, report)
    return report


def plan_hypothesis_algorithm_sandbox(
    *,
    design_path: str | Path,
    output_path: str | Path | None,
    max_ready_items: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    if max_ready_items < 1:
        raise HypothesisAlgorithmDesignerError("max_ready_items must be positive")
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    design = _load_json(Path(design_path), "hypothesis_algorithm_design")
    if design.get("state_hint") != DESIGN_STATE_HINT:
        raise HypothesisAlgorithmDesignerError("design has invalid state_hint")
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for blueprint in design.get("algorithm_blueprints", []):
        if not isinstance(blueprint, dict):
            continue
        route = _executor_route_for_blueprint(blueprint)
        if route is None:
            blocked.append(
                {
                    "blueprint_id": blueprint.get("blueprint_id"),
                    "hypothesis_id": blueprint.get("hypothesis_id"),
                    "candidate_family": blueprint.get("candidate_family"),
                    "block_reason": "unsupported_executor_route",
                    "recommended_next_command": "prepare-bounded-executor-authoring-task",
                }
            )
            continue
        if len(ready) >= max_ready_items:
            blocked.append(
                {
                    "blueprint_id": blueprint.get("blueprint_id"),
                    "hypothesis_id": blueprint.get("hypothesis_id"),
                    "candidate_family": blueprint.get("candidate_family"),
                    "block_reason": "max_ready_items_exceeded",
                    "recommended_next_command": "increase_sandbox_budget_or_reprioritize",
                }
            )
            continue
        ready.append(
            {
                "blueprint_id": blueprint.get("blueprint_id"),
                "hypothesis_id": blueprint.get("hypothesis_id"),
                "candidate_family": blueprint.get("candidate_family"),
                "executor_route": route["executor_route"],
                "executor_profile": route["executor_profile"],
                "non_arbitrary_code_guard": True,
                "required_inputs": [
                    "context_path",
                    "target_path",
                    "initial_submission_dir",
                    "output_dir",
                ],
                "source_blueprint": blueprint,
                "evaluation_plan": blueprint.get("evaluation_plan", {}),
                "internalization_command": "record-hypothesis-algorithm-evidence",
            }
        )
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_PLAN_STATE_HINT,
        "design_path": str(design_path),
        "ready_count": len(ready),
        "blocked_count": len(blocked),
        "sandbox_items": ready,
        "blocked_items": blocked,
        "next_command": "run-blueprint-sandbox" if ready else "repair-hypothesis-algorithm-design",
    }
    _write_optional_json(output_file, report)
    return report


def adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
    *,
    sandbox_item: dict[str, Any],
    output_path: str | Path | None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    family = str(sandbox_item.get("candidate_family", ""))
    target_family = _controlled_candidate_family_for_mechanism(family)
    if target_family is None:
        raise HypothesisAlgorithmDesignerError(
            f"unsupported sandbox mechanism family: {family}"
        )
    blueprint = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED",
        "blueprint_id": f"adapted_{sandbox_item.get('blueprint_id', family)}",
        "source_algorithm_blueprint_id": sandbox_item.get("blueprint_id"),
        "source_hypothesis_id": sandbox_item.get("hypothesis_id"),
        "candidate_family": target_family,
        "parameter_grid": _controlled_parameter_grid_for_mechanism(family),
        "sandbox": {
            "max_inference_seconds": 120.0,
            "promotion_min_score_delta": 0.0,
        },
        "selection_objective": _selection_objective_for_mechanism(family),
        "algorithm_designer_metadata": {
            "mechanism_family": family,
            "executor_profile": sandbox_item.get("executor_profile"),
            "source_blueprint": sandbox_item.get("source_blueprint"),
            "non_arbitrary_code_guard": bool(
                sandbox_item.get("non_arbitrary_code_guard")
            ),
        },
    }
    _write_optional_json(output_file, blueprint)
    return blueprint


def _diagnose_failure_corpus(
    *,
    memories: list[dict[str, Any]],
    evidence_cards: list[dict[str, Any]] | None = None,
    logs: list[str],
    strategy_libraries: list[dict[str, Any]],
    literature_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_summaries = _evidence_summaries(
        memories=memories,
        evidence_cards=evidence_cards or [],
        logs=logs,
        strategy_libraries=strategy_libraries,
        literature_cards=literature_cards,
    )
    text = "\n".join(str(item.get("summary", "")) for item in evidence_summaries).lower()
    mechanisms = []
    if "phase" in text or "shock" in text:
        mechanisms.extend(["phase_drift", "shock_oversmoothing"])
    if "checkpoint" in text and ("unreliab" in text or "initial" in text):
        mechanisms.append("checkpoint_unreliability")
    if "runtime" in text or "time" in text:
        mechanisms.append("runtime_bottleneck")
    if "data mismatch" in text or "shape" in text:
        mechanisms.append("data_mismatch")
    if "long" in text or "late" in text:
        mechanisms.append("long_horizon_instability")
    if "static" in text or "window" in text or "blend" in text:
        mechanisms.append("static_window_adapter_saturation")
    mechanisms = _dedupe(mechanisms)
    if not mechanisms:
        mechanisms = ["unknown_plateau"]
    return {
        "failure_memory_count": len(memories),
        "explicit_evidence_count": len(evidence_cards or []),
        "log_count": len(logs),
        "strategy_library_count": len(strategy_libraries),
        "literature_card_count": len(literature_cards),
        "training_evidence_count": sum(
            1
            for item in evidence_summaries
            if item.get("source_type")
            in {
                "task1_short_window_training",
                "task1_short_window_family_selection",
                "fno_training_optimization",
            }
        ),
        "failure_mechanisms": mechanisms,
        "evidence_summaries": evidence_summaries,
    }


def _generate_mechanism_hypotheses(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_ids = [
        item["evidence_id"]
        for item in diagnosis.get("evidence_summaries", [])
        if isinstance(item, dict)
    ]
    rows = [
        {
            "hypothesis_id": "h_uncertainty_gating_001",
            "mechanism_family": "uncertainty_gating",
            "hypothesis": (
                "Checkpoint disagreement can be useful as a reliability signal, but "
                "only if a sample-conditioned gate decides when to trust it."
            ),
            "mechanism": "Fit or derive a gate from context features, checkpoint-baseline disagreement, and local spectral energy.",
            "failure_mechanisms_addressed": [
                "checkpoint_unreliability",
                "static_window_adapter_saturation",
            ],
            "avoid_repeating": [
                "global fixed-weight checkpoint blend",
                "unconditional static window replacement",
            ],
        },
        {
            "hypothesis_id": "h_residual_correction_001",
            "mechanism_family": "residual_correction",
            "hypothesis": (
                "A learned residual corrector can use checkpoint error structure "
                "without copying checkpoint trajectories directly."
            ),
            "mechanism": "Train a lightweight residual operator from safe training/reference windows and apply it to early-middle rollout windows.",
            "failure_mechanisms_addressed": [
                "checkpoint_unreliability",
                "shock_oversmoothing",
            ],
            "avoid_repeating": ["direct checkpoint residual scaling without fitting"],
        },
        {
            "hypothesis_id": "h_data_retrieval_nonlocal_001",
            "mechanism_family": "data_retrieval_nonlocal",
            "hypothesis": (
                "Front-window phase errors may require retrieving aligned training "
                "trajectories before applying any local correction."
            ),
            "mechanism": "Retrieve nonlocal neighbors using derivative and phase features, align them, then use their increments as a dynamics source.",
            "failure_mechanisms_addressed": ["phase_drift", "data_mismatch"],
            "avoid_repeating": ["nearest-neighbor retrieval without phase alignment"],
        },
        {
            "hypothesis_id": "h_time_window_decomposition_001",
            "mechanism_family": "time_window_decomposition",
            "hypothesis": (
                "Early, middle, and late windows have different failure modes and "
                "should not share a single correction rule."
            ),
            "mechanism": "Design separate adapters for 10:20, 20:35, and 35:57 with per-window observables and falsification criteria.",
            "failure_mechanisms_addressed": [
                "static_window_adapter_saturation",
                "long_horizon_instability",
            ],
            "avoid_repeating": ["single global candidate score as the only selection signal"],
        },
        {
            "hypothesis_id": "h_physics_constraint_001",
            "mechanism_family": "physics_constraint",
            "hypothesis": (
                "Burgers residual can reject corrections that improve local MSE by "
                "creating nonphysical shock or high-frequency artifacts."
            ),
            "mechanism": "Use target-free Burgers residual and spectral diagnostics as non-regression gates around learned corrections.",
            "failure_mechanisms_addressed": [
                "shock_oversmoothing",
                "long_horizon_instability",
            ],
            "avoid_repeating": ["score-only promotion without physics non-regression"],
        },
        {
            "hypothesis_id": "h_operator_structure_001",
            "mechanism_family": "operator_structure",
            "hypothesis": (
                "A low-rank or spectral operator adapter may capture front-window "
                "dynamics better than scalar window blending."
            ),
            "mechanism": "Fit a bounded POD/ridge or spectral correction operator over safe training windows and expose it as a sandbox blueprint.",
            "failure_mechanisms_addressed": ["phase_drift", "runtime_bottleneck"],
            "avoid_repeating": ["scalar-only correction families"],
        },
    ]
    for row in rows:
        row["source_evidence_ids"] = evidence_ids
        row["source_evidence_types"] = sorted(
            {
                str(item["source_type"])
                for item in diagnosis.get("evidence_summaries", [])
                if isinstance(item, dict) and isinstance(item.get("source_type"), str)
            }
        )
        row["evidence_reuse"] = {
            "inherits": list(row["failure_mechanisms_addressed"]),
            "differs_from_old_failures": list(row["avoid_repeating"]),
        }
    constraints = _next_generation_constraints_from_diagnosis(diagnosis)
    for order, row in enumerate(rows):
        family = str(row["mechanism_family"])
        action = _next_generation_action_for_family(
            family=family,
            constraints=constraints,
        )
        row["next_generation_action"] = action
        row["priority_score"] = _mechanism_priority_score(action)
        row["base_priority_order"] = order
        row["evidence_reuse"]["next_generation_constraints"] = constraints
    return sorted(
        rows,
        key=lambda item: (
            -float(item.get("priority_score", 0.0)),
            int(item.get("base_priority_order", 0)),
        ),
    )


def _algorithm_blueprint(*, index: int, hypothesis: dict[str, Any]) -> dict[str, Any]:
    family = str(hypothesis["mechanism_family"])
    return {
        "blueprint_id": f"bp_algorithm_designer_{index:03d}_{family}",
        "hypothesis_id": hypothesis["hypothesis_id"],
        "candidate_family": family,
        "source_evidence_ids": hypothesis.get("source_evidence_ids", []),
        "source_evidence_types": hypothesis.get("source_evidence_types", []),
        "algorithm_design": {
            "inputs": [
                "task_context",
                "safe_baseline_prediction",
                "official_training_or_safe_reference_windows",
                "optional_checkpoint_prediction",
            ],
            "outputs": ["task1_candidate_prediction", "evidence_artifact"],
            "training_or_fitting_process": _training_process_for_family(family),
            "inference_process": _inference_process_for_family(family),
            "data_dependencies": [
                "official_training_data",
                "safe_baseline_submission",
                "local_validation_target_for_selection_only",
            ],
            "complexity_budget": {
                "sandbox_level": 0,
                "max_wall_seconds_hint": 600,
                "macmini_timing_factor": 0.05,
            },
            "failure_risks": _failure_risks_for_family(family),
        },
        "execution_constraints": {
            "forbidden_data": list(FORBIDDEN_DATA),
            "allowed_target_use": "local_validation_selection_and_falsification_only",
            "requires_manifest": True,
            "requires_reproducible_parameters": True,
            "bounded_executor_required": True,
        },
        "evaluation_plan": {
            "expected_observables": _expected_observables_for_family(family),
            "segment_level_metrics": [
                "rel_mse_10_20",
                "rel_mse_20_35",
                "rel_mse_35_57",
                "task1_total_score_proxy",
            ],
            "physics_diagnostics": [
                "burgers_residual_mean",
                "high_frequency_energy_ratio",
                "shock_smoothing_indicator",
            ],
            "runtime_metrics": [
                "train_time_a100_equivalent",
                "inference_time_a100_equivalent",
                "source_materialization_seconds",
            ],
            "ablation_plan": [
                "disable candidate source",
                "disable non-regression gate",
                "compare against initial and last failed family",
            ],
            "falsification_criteria": _falsification_for_family(family),
        },
        "internalization_plan": {
            "algorithm_card_update": {
                "target_status": "tested",
                "mechanism_family": family,
            },
            "evidence_record": {
                "required_fields": [
                    "judge_report_path",
                    "source_manifest_path",
                    "segment_metrics",
                    "runtime_metrics",
                ],
            },
            "failure_taxonomy_update": {
                "on_failure": [
                    "candidate_no_segment_improvement",
                    "candidate_runtime_regression",
                    "candidate_physics_regression",
                ],
                "on_success": ["candidate_promoted_to_strategy_proposal"],
            },
        },
    }


def _designer_rubric(
    *,
    hypotheses: list[dict[str, Any]],
    blueprints: list[dict[str, Any]],
    diversity_audit: dict[str, Any],
    min_hypothesis_count: int,
    min_mechanism_family_count: int,
) -> dict[str, Any]:
    violations = _design_violations(
        hypotheses=hypotheses,
        blueprints=blueprints,
        diversity_audit=diversity_audit,
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
    )
    return {
        "hypothesis_count": len(hypotheses),
        "mechanism_family_count": len(_mechanism_families(hypotheses)),
        "blueprint_count": len(blueprints),
        "violations": violations,
        "decision": "repair_required" if violations else "ready_for_sandbox_planning",
    }


def _design_violations(
    *,
    hypotheses: list[dict[str, Any]],
    blueprints: list[dict[str, Any]],
    diversity_audit: Any,
    min_hypothesis_count: int,
    min_mechanism_family_count: int,
) -> list[str]:
    violations: list[str] = []
    if len(hypotheses) < min_hypothesis_count:
        violations.append("insufficient_hypothesis_count")
    if len(_mechanism_families(hypotheses)) < min_mechanism_family_count:
        violations.append("insufficient_mechanism_diversity")
    if (
        isinstance(diversity_audit, dict)
        and int(diversity_audit.get("near_duplicate_count", 0)) > 0
    ):
        violations.append("near_duplicate_hypotheses")
    for blueprint in blueprints:
        constraints = blueprint.get("execution_constraints")
        forbidden = constraints.get("forbidden_data") if isinstance(constraints, dict) else None
        if not isinstance(forbidden, list) or not set(FORBIDDEN_DATA).issubset(
            {str(item) for item in forbidden}
        ):
            violations.append("blueprint_missing_forbidden_data_constraints")
            break
    for blueprint in blueprints:
        if not _evaluation_plan_complete(blueprint.get("evaluation_plan")):
            violations.append("blueprint_missing_evaluation_plan")
            break
    return _dedupe(violations)


def _evaluation_plan_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = [
        "expected_observables",
        "segment_level_metrics",
        "physics_diagnostics",
        "runtime_metrics",
        "ablation_plan",
        "falsification_criteria",
    ]
    return all(isinstance(value.get(key), list) and value[key] for key in required)


def _diversity_audit(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    seen_signatures: set[str] = set()
    duplicates = []
    for item in hypotheses:
        signature = _normalized_signature(
            str(item.get("mechanism_family", "")) + " " + str(item.get("mechanism", ""))
        )
        if signature in seen_signatures:
            duplicates.append(item.get("hypothesis_id"))
        seen_signatures.add(signature)
    return {
        "mechanism_families": sorted(_mechanism_families(hypotheses)),
        "near_duplicate_count": len(duplicates),
        "near_duplicate_hypothesis_ids": duplicates,
    }


def _memory_summary(*, index: int, memory: dict[str, Any]) -> dict[str, Any]:
    state_hint = str(memory.get("state_hint", "unknown"))
    conclusion = str(memory.get("conclusion") or memory.get("recommendation") or "")
    return {
        "evidence_id": f"failure_memory_{index:03d}",
        "source_type": "failure_memory",
        "state_hint": state_hint,
        "summary": conclusion[:400],
        "avoid_repeating": _avoid_repeating_from_memory(memory),
    }


def _evidence_summaries(
    *,
    memories: list[dict[str, Any]],
    evidence_cards: list[dict[str, Any]] | None = None,
    logs: list[str],
    strategy_libraries: list[dict[str, Any]],
    literature_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        _memory_summary(index=index, memory=memory)
        for index, memory in enumerate(memories)
    ]
    for index, evidence in enumerate(evidence_cards or []):
        source_type = str(evidence.get("source_type") or "evidence")
        rows.append(
            {
                "evidence_id": f"explicit_evidence_{index:03d}",
                "source_type": source_type,
                "state_hint": evidence.get("state_hint"),
                "summary": json.dumps(evidence, ensure_ascii=False, sort_keys=True)[
                    :400
                ],
                "avoid_repeating": [],
                "raw_constraints": evidence.get("next_generation_constraints")
                if isinstance(evidence.get("next_generation_constraints"), dict)
                else {},
            }
        )
    for index, text in enumerate(logs):
        rows.append(
            {
                "evidence_id": f"run_log_{index:03d}",
                "source_type": "run_log",
                "summary": text[:400],
                "avoid_repeating": [],
            }
        )
    for index, library in enumerate(strategy_libraries):
        rows.append(
            {
                "evidence_id": f"strategy_library_{index:03d}",
                "source_type": "strategy_library",
                "summary": json.dumps(library, ensure_ascii=False, sort_keys=True)[:400],
                "avoid_repeating": [],
            }
        )
    for index, card in enumerate(literature_cards):
        source_type = str(card.get("source_type") or "literature_card")
        rows.append(
            {
                "evidence_id": f"literature_card_{index:03d}",
                "source_type": source_type,
                "summary": json.dumps(card, ensure_ascii=False, sort_keys=True)[:400],
                "avoid_repeating": [],
                "raw_constraints": card.get("next_generation_constraints")
                if isinstance(card.get("next_generation_constraints"), dict)
                else {},
            }
        )
    return rows


def _next_generation_constraints_from_diagnosis(
    diagnosis: dict[str, Any]
) -> dict[str, list[str]]:
    constraints = {
        "prefer_mechanism_families": [],
        "refine_mechanism_families": [],
        "near_miss_mechanism_families": [],
        "avoid_mechanism_families": [],
        "quarantine_mechanism_families": [],
    }
    for item in diagnosis.get("evidence_summaries", []):
        if not isinstance(item, dict):
            continue
        summary = item.get("raw_constraints")
        if not isinstance(summary, dict):
            continue
        for key in constraints:
            constraints[key].extend(_string_list(summary.get(key)))
        if item.get("source_type") in {
            "task1_short_window_training",
            "task1_short_window_family_selection",
            "fno_training_optimization",
        }:
            _merge_training_constraints(
                constraints=constraints,
                training_constraints=summary,
            )
    return {key: _dedupe(value) for key, value in constraints.items()}


def _merge_training_constraints(
    *,
    constraints: dict[str, list[str]],
    training_constraints: dict[str, Any],
) -> None:
    if _string_list(training_constraints.get("avoid_training_model_families")):
        constraints["avoid_mechanism_families"].append("operator_structure")
        return
    if _string_list(training_constraints.get("training_model_families")):
        constraints["prefer_mechanism_families"].append("operator_structure")
    if _string_list(training_constraints.get("prefer_training_objectives")):
        constraints["prefer_mechanism_families"].append("residual_correction")
    if _string_list(training_constraints.get("refine_training_horizons")):
        constraints["refine_mechanism_families"].append("time_window_decomposition")


def _next_generation_action_for_family(
    *, family: str, constraints: dict[str, list[str]]
) -> str:
    if family in constraints.get("quarantine_mechanism_families", []):
        return "quarantine"
    if family in constraints.get("avoid_mechanism_families", []):
        return "avoid"
    if family in constraints.get("prefer_mechanism_families", []):
        return "prefer"
    if family in constraints.get("refine_mechanism_families", []):
        return "refine"
    if family in constraints.get("near_miss_mechanism_families", []):
        return "near_miss"
    return "explore"


def _mechanism_priority_score(action: str) -> float:
    return {
        "prefer": 100.0,
        "refine": 80.0,
        "near_miss": 60.0,
        "explore": 40.0,
        "avoid": 10.0,
        "quarantine": 0.0,
    }.get(action, 40.0)


def _avoid_repeating_from_memory(memory: dict[str, Any]) -> list[str]:
    text = json.dumps(memory, ensure_ascii=False).lower()
    rows = []
    if "direct" in text or "replacement" in text:
        rows.append("direct window replacement")
    if "blend" in text:
        rows.append("fixed window blend")
    if "residual" in text and "scal" in text:
        rows.append("untrained residual scaling")
    if "initial" in text:
        rows.append("sources that fail local subwindow selection against initial")
    return _dedupe(rows)


def _mechanism_families(hypotheses: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("mechanism_family"))
        for item in hypotheses
        if isinstance(item.get("mechanism_family"), str) and item.get("mechanism_family")
    }


def _training_process_for_family(family: str) -> str:
    mapping = {
        "uncertainty_gating": "Fit a bounded sample/window reliability model from safe features and prior outcomes.",
        "residual_correction": "Fit a lightweight residual operator on official training or safe reference windows.",
        "data_retrieval_nonlocal": "Index safe trajectories by context, derivative, and phase features; estimate aligned increments.",
        "time_window_decomposition": "Fit or select independent source policies per early-middle subwindow.",
        "physics_constraint": "Calibrate target-free Burgers residual thresholds on safe baseline trajectories.",
        "operator_structure": "Fit low-rank or spectral operator coefficients on safe windows.",
    }
    return mapping.get(family, "Fit a bounded algorithmic adapter on safe data.")


def _inference_process_for_family(family: str) -> str:
    mapping = {
        "uncertainty_gating": "Compute reliability features per sample/window and choose or blend sources under non-regression gates.",
        "residual_correction": "Predict residual corrections and apply them only where segment and physics guards allow.",
        "data_retrieval_nonlocal": "Retrieve aligned neighbors from context and inject their early-middle increments.",
        "time_window_decomposition": "Route each window through its own source selector and join with handoff checks.",
        "physics_constraint": "Reject candidate corrections that increase Burgers residual or high-frequency energy.",
        "operator_structure": "Apply the fitted low-rank/spectral operator to context-derived features.",
    }
    return mapping.get(family, "Execute the bounded adapter and emit a candidate prediction.")


def _failure_risks_for_family(family: str) -> list[str]:
    common = ["local_validation_overfit", "runtime_regression"]
    mapping = {
        "uncertainty_gating": ["gate_overfits_validation_windows", "false_confidence_on_checkpoint"],
        "residual_correction": ["residual_overshoot", "shock_smoothing"],
        "data_retrieval_nonlocal": ["neighbor_mismatch", "phase_alignment_error"],
        "time_window_decomposition": ["handoff_discontinuity", "window_objective_conflict"],
        "physics_constraint": ["physics_proxy_mismatch", "overly_conservative_rejection"],
        "operator_structure": ["low_rank_underfit", "spectral_ringing"],
    }
    return mapping.get(family, []) + common


def _expected_observables_for_family(family: str) -> list[str]:
    mapping = {
        "uncertainty_gating": ["selected non-initial source only on high-confidence samples"],
        "residual_correction": ["lower rel-MSE in 20:35 or 35:57 without late regression"],
        "data_retrieval_nonlocal": ["lower phase-aligned derivative error"],
        "time_window_decomposition": ["different source choices across 10:20, 20:35, and 35:57"],
        "physics_constraint": ["reduced Burgers residual non-regression violations"],
        "operator_structure": ["stable low-rank coefficient norms"],
    }
    return mapping.get(family, ["observable segment-level improvement"])


def _falsification_for_family(family: str) -> list[str]:
    return [
        "no target subwindow beats initial under local validation selection",
        "task1 total score decreases beyond tolerance",
        "physics or runtime non-regression gate fails",
        f"{family} reproduces a previously rejected failure mechanism",
    ]


def _normalized_signature(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:12])


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise HypothesisAlgorithmDesignerError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisAlgorithmDesignerError(f"{label} invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HypothesisAlgorithmDesignerError(f"{label} must be a JSON object")
    return payload


def _read_text(path: Path) -> str:
    if not path.exists():
        raise HypothesisAlgorithmDesignerError(f"text artifact does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise HypothesisAlgorithmDesignerError(f"output already exists: {path}")


def _dedupe(items: list[str]) -> list[str]:
    rows = []
    seen = set()
    for item in items:
        if item not in seen:
            rows.append(item)
            seen.add(item)
    return rows


def _find_blueprint(design: dict[str, Any], blueprint_id: str) -> dict[str, Any] | None:
    for item in design.get("algorithm_blueprints", []):
        if isinstance(item, dict) and item.get("blueprint_id") == blueprint_id:
            return item
    return None


def _source_blueprint_id_from_sandbox(sandbox: dict[str, Any]) -> str:
    adapted = sandbox.get("adapted_blueprint")
    if isinstance(adapted, dict) and isinstance(
        adapted.get("source_algorithm_blueprint_id"),
        str,
    ):
        return adapted["source_algorithm_blueprint_id"]
    metadata = sandbox.get("algorithm_designer_metadata")
    if isinstance(metadata, dict) and isinstance(
        metadata.get("source_algorithm_blueprint_id"),
        str,
    ):
        return metadata["source_algorithm_blueprint_id"]
    return str(sandbox.get("blueprint_id", ""))


def _sandbox_score_delta(sandbox: dict[str, Any]) -> float | None:
    value = _float_or_none(sandbox.get("score_delta"))
    if value is not None:
        return value
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict):
        return _float_or_none(verifier.get("score_delta_vs_baseline"))
    return None


def _sandbox_evidence_summary(sandbox: dict[str, Any]) -> dict[str, Any]:
    metrics = _sandbox_metric_results(sandbox)
    score_delta = _sandbox_score_delta(sandbox)
    if score_delta is None:
        score_delta = _metric_float(metrics, "score_delta_vs_baseline")
    early_delta = _metric_float(metrics, "early_score_delta_vs_baseline")
    middle_delta = _metric_float(metrics, "middle_score_guardrail")
    late_delta = _metric_float(metrics, "late_rmse_guardrail")
    inference_time = _metric_float(metrics, "inference_time_budget")
    train_time = _metric_float(metrics, "train_time_budget")
    physics_delta = _metric_float(metrics, "physics_residual_rmse_delta")
    context_preserved = _metric_bool(
        metrics,
        "context_preserved",
        default=_sandbox_context_preserved(sandbox),
    )
    runtime_passed = _metric_passed(metrics, "inference_time_budget", default=True)
    physics_passed = _metric_passed(
        metrics,
        "physics_residual_rmse_delta",
        default=True,
    )
    rules_passed = _sandbox_rules_passed(sandbox)
    leakage_detected = _sandbox_leakage_detected(sandbox)
    repeated_similarity = _sandbox_repeated_failure_similarity(sandbox)
    segment_deltas = {
        "early": early_delta,
        "middle": middle_delta,
        "late": late_delta,
    }
    return {
        "rules_passed": rules_passed,
        "score_delta": score_delta,
        "task1_total_score_delta": score_delta,
        "segment_deltas": segment_deltas,
        "early_segment_delta": early_delta,
        "middle_segment_delta": middle_delta,
        "late_segment_delta": late_delta,
        "inference_time_delta": inference_time,
        "train_time_delta": train_time,
        "physics_residual_delta": physics_delta,
        "context_preserved": context_preserved,
        "leakage_detected": leakage_detected,
        "runtime_guard_passed": runtime_passed,
        "physics_guard_passed": physics_passed,
        "repeated_failure_similarity": repeated_similarity,
        "primary_improvement_axis": _primary_axis(segment_deltas, prefer_positive=True),
        "primary_failure_axis": _primary_axis(segment_deltas, prefer_positive=False),
    }


def _promotion_rubric_decision(
    *, evidence: dict[str, Any], mechanism_family: str
) -> dict[str, Any]:
    score_delta = _float_or_zero(evidence.get("score_delta"))
    early_delta = _float_or_zero(evidence.get("early_segment_delta"))
    middle_delta = _float_or_zero(evidence.get("middle_segment_delta"))
    late_delta = _float_or_zero(evidence.get("late_segment_delta"))
    repeated_similarity = _float_or_zero(evidence.get("repeated_failure_similarity"))
    hard_violation_codes = []
    if evidence.get("context_preserved") is False:
        hard_violation_codes.append("context_prefix_violation")
    if evidence.get("leakage_detected") is True:
        hard_violation_codes.append("leakage_risk_detected")
    if evidence.get("runtime_guard_passed") is False:
        hard_violation_codes.append("runtime_guard_failed")
    if evidence.get("physics_guard_passed") is False:
        hard_violation_codes.append("physics_guard_failed")
    if hard_violation_codes:
        decision = "quarantine"
        failure_codes = hard_violation_codes
    elif score_delta >= 0.1 and min(middle_delta, late_delta) >= 0.0:
        decision = "promote_to_strategy"
        failure_codes = []
    elif max(early_delta, middle_delta, late_delta) > 0.05 and score_delta > -0.1:
        decision = "promote_to_refinement"
        failure_codes = ["local_segment_only_improvement"]
    elif score_delta > 0.0 or max(early_delta, middle_delta, late_delta) > 0.0:
        decision = "record_near_miss"
        failure_codes = ["near_miss_insufficient_global_improvement"]
    else:
        decision = "reject_with_failure_memory"
        failure_codes = ["candidate_no_segment_improvement"]
    if repeated_similarity >= 0.85 and decision not in {"quarantine", "promote_to_strategy"}:
        if "repeated_failure_similarity_high" not in failure_codes:
            failure_codes.append("repeated_failure_similarity_high")
        if decision == "record_near_miss":
            decision = "reject_with_failure_memory"
    constraints = _next_generation_constraints_for_decision(
        decision=decision,
        mechanism_family=mechanism_family,
    )
    return {
        "schema_version": 1,
        "decision": decision,
        "failure_codes": failure_codes,
        "reason_codes": _dedupe(failure_codes + _rubric_reason_codes(evidence)),
        "next_generation_constraints": constraints,
        "next_command": _next_command_for_decision(decision),
    }


def _next_generation_constraints_for_decision(
    *, decision: str, mechanism_family: str
) -> dict[str, list[str]]:
    constraints = {
        "prefer_mechanism_families": [],
        "refine_mechanism_families": [],
        "near_miss_mechanism_families": [],
        "avoid_mechanism_families": [],
        "quarantine_mechanism_families": [],
    }
    if decision == "promote_to_strategy":
        constraints["prefer_mechanism_families"].append(mechanism_family)
    elif decision == "promote_to_refinement":
        constraints["refine_mechanism_families"].append(mechanism_family)
    elif decision == "record_near_miss":
        constraints["near_miss_mechanism_families"].append(mechanism_family)
    elif decision == "reject_with_failure_memory":
        constraints["avoid_mechanism_families"].append(mechanism_family)
    elif decision == "quarantine":
        constraints["quarantine_mechanism_families"].append(mechanism_family)
        constraints["avoid_mechanism_families"].append(mechanism_family)
    return constraints


def _next_command_for_decision(decision: str) -> str:
    mapping = {
        "promote_to_strategy": "record-strategy-proposal",
        "promote_to_refinement": "plan-refinement-sandbox",
        "record_near_miss": "build-hypothesis-algorithm-design",
        "reject_with_failure_memory": "record-failure-memory",
        "quarantine": "quarantine-hypothesis-algorithm",
    }
    return mapping.get(decision, "build-hypothesis-algorithm-design")


def _rubric_reason_codes(evidence: dict[str, Any]) -> list[str]:
    reasons = []
    if _float_or_zero(evidence.get("score_delta")) > 0.0:
        reasons.append("task1_total_score_improved")
    if evidence.get("runtime_guard_passed") is False:
        reasons.append("runtime_regression")
    if evidence.get("physics_guard_passed") is False:
        reasons.append("physics_regression")
    if evidence.get("primary_improvement_axis") != "none":
        reasons.append(f"{evidence['primary_improvement_axis']}_axis_improved")
    if evidence.get("primary_failure_axis") != "none":
        reasons.append(f"{evidence['primary_failure_axis']}_axis_failed")
    return reasons


def _sandbox_metric_results(sandbox: dict[str, Any]) -> dict[str, Any]:
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("metric_results"), dict):
        return verifier["metric_results"]
    return {}


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    metric = metrics.get(key)
    if isinstance(metric, dict):
        return _float_or_none(metric.get("value"))
    return None


def _metric_bool(metrics: dict[str, Any], key: str, *, default: bool) -> bool:
    metric = metrics.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("value"), bool):
        return bool(metric["value"])
    return default


def _metric_passed(metrics: dict[str, Any], key: str, *, default: bool) -> bool:
    metric = metrics.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("passed"), bool):
        return bool(metric["passed"])
    return default


def _sandbox_context_preserved(sandbox: dict[str, Any]) -> bool:
    if isinstance(sandbox.get("context_preserved"), bool):
        return bool(sandbox["context_preserved"])
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("context_preserved"), bool):
        return bool(verifier["context_preserved"])
    return True


def _sandbox_leakage_detected(sandbox: dict[str, Any]) -> bool:
    reference = sandbox.get("reference_audit")
    if isinstance(reference, dict) and reference.get("uses_target_as_reference") is True:
        return True
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict) and verifier.get("leakage_detected") is True:
        return True
    return False


def _sandbox_repeated_failure_similarity(sandbox: dict[str, Any]) -> float:
    for key in ["failure_memory_similarity", "prior_failure_similarity"]:
        value = sandbox.get(key)
        if isinstance(value, dict):
            similarity = _float_or_none(value.get("max_similarity"))
            if similarity is not None:
                return similarity
    return 0.0


def _primary_axis(
    segment_deltas: dict[str, float | None], *, prefer_positive: bool
) -> str:
    rows = {
        key: value
        for key, value in segment_deltas.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if not rows:
        return "none"
    if prefer_positive:
        key, value = max(rows.items(), key=lambda item: float(item[1]))
        return key if float(value) > 0.0 else "none"
    key, value = min(rows.items(), key=lambda item: float(item[1]))
    return key if float(value) < 0.0 else "none"


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _sandbox_rules_passed(sandbox: dict[str, Any]) -> bool:
    if isinstance(sandbox.get("rules_passed"), bool):
        return bool(sandbox["rules_passed"])
    verifier = sandbox.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("rules_passed"), bool):
        return bool(verifier["rules_passed"])
    return False


def _sandbox_artifact_value(sandbox: dict[str, Any], key: str) -> Any:
    if key in sandbox:
        return sandbox.get(key)
    artifacts = sandbox.get("artifacts")
    if isinstance(artifacts, dict):
        return artifacts.get(key)
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _executor_route_for_blueprint(blueprint: dict[str, Any]) -> dict[str, str] | None:
    family = str(blueprint.get("candidate_family", ""))
    route = SUPPORTED_EXECUTOR_ROUTES.get(family)
    if route is None:
        return None
    constraints = blueprint.get("execution_constraints")
    if not isinstance(constraints, dict):
        return None
    forbidden = constraints.get("forbidden_data")
    if not isinstance(forbidden, list):
        return None
    if not set(FORBIDDEN_DATA).issubset({str(item) for item in forbidden}):
        return None
    if constraints.get("bounded_executor_required") is not True:
        return None
    if not _evaluation_plan_complete(blueprint.get("evaluation_plan")):
        return None
    return route


def _controlled_candidate_family_for_mechanism(family: str) -> str | None:
    mapping = {
        "uncertainty_gating": "task1_early_refinement",
        "residual_correction": "learned_residual_correction",
        "data_retrieval_nonlocal": "task1_retrieval_refinement",
        "time_window_decomposition": "task1_early_refinement",
        "physics_constraint": "task1_physics_residual_refinement",
        "operator_structure": "linear_trend_spectral",
    }
    return mapping.get(family)


def _controlled_parameter_grid_for_mechanism(family: str) -> dict[str, Any]:
    if family == "residual_correction":
        return {
            "residual_shrinkage": [0.25, 0.5],
            "residual_spectral_strength": [0.0],
        }
    if family == "operator_structure":
        return {
            "slope_window": [5],
            "gain": [-0.075],
            "decay": [70.0],
            "spectral_strength": [0.5],
        }
    return {"candidate_params": _controlled_candidate_params_for_mechanism(family)}


def _controlled_candidate_params_for_mechanism(family: str) -> list[dict[str, Any]]:
    if family == "residual_correction":
        return []
    if family == "data_retrieval_nonlocal":
        return [
            {
                "k": 8,
                "mode": "affine",
                "early_blend_steps": 25,
                "persistence_weight_start": 0.25,
                "spectral_strength": 0.0,
                "early_anchor": "phase_aligned_retrieval",
                "handoff_curve": "cosine",
                "phase_shift_scale": 1.0,
            }
        ]
    if family == "physics_constraint":
        return [
            {
                "base_family": "task1_retrieval_refinement",
                "k": 48,
                "mode": "affine",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
                "correction_weight": 0.02,
                "stop_step": 105,
                "residual_spectral_strength": 0.0,
                "viscosity": 0.001,
                "dt": 0.05,
                "dx": 1.0 / 256.0,
            }
        ]
    if family == "operator_structure":
        return [{"window": 5, "growth": 0.0, "damping_cutoff": 70, "strength": 0.5}]
    return [
        {
            "k": 16,
            "mode": "affine",
            "early_blend_steps": 25,
            "persistence_weight_start": 0.25,
            "spectral_strength": 0.0,
            "early_anchor": "phase_aligned_retrieval",
            "handoff_curve": "cosine",
            "phase_shift_scale": 1.0,
        }
    ]


def _selection_objective_for_mechanism(family: str) -> str:
    if family in {"uncertainty_gating", "time_window_decomposition"}:
        return "early_segment_score"
    if family == "physics_constraint":
        return "pareto_physics_score"
    return "task1_total_score_proxy"
