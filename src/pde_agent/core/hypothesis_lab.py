"""Hypothesis Lab orchestration facade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pde_agent.core.hypothesis_lab_utils import (
    EARLY_HANDOFF_CURVES,
    EARLY_REFINEMENT_ANCHORS,
    HypothesisLabError,
    _clamp_float,
    _clamp_int,
    _clamp_task1_retrieval_k,
    _guard_output,
    _normalize_retrieval_refinement_params,
    _read_json,
    _require_state,
    _string_list_from_any,
    _write_json,
    _write_optional_json,
)
from pde_agent.core.hypothesis_lab_evidence import (
    _merge_card_list,
    _merge_ranked_evidence_row,
    _normalize_evidence_params,
    _rank_hypotheses,
    _ranked_evidence_params,
)
from pde_agent.core.hypothesis_lab_knowledge_cards import (
    _checkpoint_card,
    _checkpoint_family,
    _external_blend_card,
    _linear_trend_spectral_card,
    _literature_expansion_report_card,
    _literature_note_card,
    _parse_literature_note,
)
from pde_agent.core.hypothesis_lab_search_policy import (
    _adaptive_hypothesis_lab_search_policy_card,
    _adaptive_plateau_card,
    _adaptive_retrieval_refinement_card,
    _first_log_best_candidate,
    _scientific_log_card,
)
from pde_agent.core.hypothesis_lab_anchors import (
    _affine_context_calibration_anchor_prediction,
    _damped_linear_velocity_anchor_prediction,
    _early_handoff_weights,
    _linear_velocity_anchor_prediction,
    _mean_trend_bias_anchor_prediction,
    _phase_aligned_retrieval_anchor_prediction,
    _phase_derivative_hybrid_anchor_prediction,
    _retrieval_derivative_match_anchor_prediction,
    _second_order_velocity_anchor_prediction,
)
from pde_agent.core.hypothesis_lab_scoring import _task1_total_score_proxy
from pde_agent.core.hypothesis_lab_scoring import _sandbox_pareto_promotion_gate
from pde_agent.core.hypothesis_lab_predictions import (
    _base_prediction_for_physics_residual_refinement,
)
from pde_agent.core.hypothesis_lab_pareto import (
    _algorithm_candidate_cards,
    _executed_algorithm_candidate_evidence,
    _pareto_memory_cards,
    _record_algorithm_candidate_outcome,
    _record_hypothesis_strategy_growth_artifacts,
    _record_pareto_failure_memory,
)
from pde_agent.core.hypothesis_lab_candidate_pool import (
    _apply_early_memory_policy,
    _candidate_pool_action,
    _candidate_pool_attribution,
    _candidate_pool_policy_from_evidence,
    _candidate_pool_search_space_policy,
    _candidate_early_rel_mse,
    _dedupe_strings,
    _dedupe_values,
    _directed_retrieval_refinement_params,
    _early_memory_policy,
    _early_memory_rank_key,
    _early_param_sort_key,
    _early_params_match,
    _early_refinement_experiment_plan,
    _early_refinement_signals,
    _filter_retrieval_refinement_params_by_directives,
    _finite_float_or_none,
    _retrieval_refinement_avoid_rules_from_directives,
    _retrieval_refinement_center,
    _retrieval_refinement_experiment_plan,
    _retrieval_refinement_search_directives_from_evidence,
    _should_avoid_exact_memory_params,
)
from pde_agent.core.hypothesis_lab_candidate_params import (
    _default_retrieval_refinement_params,
    _early_refinement_params,
    _early_transition_expert_parameter_grid,
    _early_transition_expert_params,
    _front25_dynamics_candidate_params,
    _local_retrieval_refinement_params,
)
from pde_agent.core.hypothesis_lab_sandbox_runner import (
    _iter_parameter_grid,
    run_blueprint_sandbox,
)
from pde_agent.core.hypothesis_lab_selection import (
    _candidate_selection_key,
    _selection_objective,
)

CARDS_STATE_HINT = "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
HYPOTHESES_STATE_HINT = "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"
BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED"
VERIFIED_BLUEPRINTS_STATE_HINT = "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED"
SANDBOX_STATE_HINT = "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
RUN_STATE_HINT = "HYPOTHESIS_LAB_RAN"

# Re-exports for backward compatibility
from pde_agent.core.hypothesis_lab_knowledge import (
    build_knowledge_cards,
    _manual_candidates,
)

from pde_agent.core.hypothesis_lab_generation import (
    generate_hypotheses,
)

from pde_agent.core.hypothesis_lab_design import (
    design_algorithm_blueprints,
)

def run_hypothesis_lab(
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    failure_memory_output_path: str | Path | None = None,
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    output_dir: str | Path,
    submission_id: str,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_path = root / "hypothesis_lab_report.json"
    _guard_output(report_path, force=force)
    root.mkdir(parents=True, exist_ok=True)
    cards = build_knowledge_cards(
        manual_report_paths=manual_report_paths,
        adaptive_report_paths=adaptive_report_paths,
        log_paths=log_paths,
        checkpoint_paths=checkpoint_paths,
        literature_paths=literature_paths,
        algorithm_candidate_paths=algorithm_candidate_paths,
        failure_memory_paths=failure_memory_paths,
        output_path=root / "knowledge_cards.json",
        force=force,
    )
    hypotheses = generate_hypotheses(
        cards=cards,
        output_path=root / "hypotheses.json",
        force=force,
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=root / "blueprints.json",
        force=force,
    )
    if not blueprints["blueprints"]:
        raise HypothesisLabError("no executable blueprints generated")
    executable_blueprint = _first_executable_blueprint(blueprints["blueprints"])
    if executable_blueprint is None:
        raise HypothesisLabError("no sandbox-executable blueprints generated")
    sandbox = run_blueprint_sandbox(
        blueprint=executable_blueprint,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        output_dir=root / "sandbox",
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    verified_blueprints_path = root / "verified_blueprints.json"
    verified_blueprints = _verified_blueprints(
        blueprints=blueprints,
        sandbox=sandbox,
    )
    _write_json(verified_blueprints_path, verified_blueprints)
    candidate_outcome = _record_algorithm_candidate_outcome(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
        sandbox=sandbox,
        failure_memory_output_path=failure_memory_output_path,
        force=force,
    )
    pareto_memory = _record_pareto_failure_memory(
        sandbox=sandbox,
        failure_memory_output_path=failure_memory_output_path,
        algorithm_candidate_evidence=_executed_algorithm_candidate_evidence(
            cards=cards,
            hypotheses=hypotheses,
            executable_blueprint=executable_blueprint,
        ),
        force=force,
    )
    strategy_growth = _record_hypothesis_strategy_growth_artifacts(
        cards=cards,
        hypotheses=hypotheses,
        executable_blueprint=executable_blueprint,
        sandbox=sandbox,
        output_dir=root,
        force=force,
    )
    report = {
        "schema_version": 1,
        "state_hint": RUN_STATE_HINT,
        "cards": {
            "count": cards["card_count"],
            "path": str(root / "knowledge_cards.json"),
        },
        "hypotheses": {
            "count": hypotheses["hypothesis_count"],
            "path": str(root / "hypotheses.json"),
        },
        "blueprints": {
            "count": blueprints["blueprint_count"],
            "path": str(root / "blueprints.json"),
        },
        "sandbox": sandbox,
        "artifacts": {
            "hypothesis_lab_report_path": str(report_path),
            "sandbox_report_path": sandbox["artifacts"]["sandbox_report_path"],
            "best_submission_dir": sandbox["artifacts"]["best_submission_dir"],
            "verified_blueprint_path": str(verified_blueprints_path),
        },
        "next_command": strategy_growth["next_command"]
        if strategy_growth is not None
        else sandbox["next_command"],
    }
    if strategy_growth is not None:
        report["strategy_growth"] = strategy_growth
        report["artifacts"].update(strategy_growth["artifacts"])
    if candidate_outcome is not None:
        report["candidate_outcome"] = candidate_outcome
    if pareto_memory is not None:
        report["pareto_memory"] = pareto_memory
    _write_json(report_path, report)
    return report

def _first_executable_blueprint(blueprints: list[Any]) -> dict[str, Any] | None:
    for blueprint in blueprints:
        if not isinstance(blueprint, dict):
            continue
        sandbox = blueprint.get("sandbox")
        if not isinstance(sandbox, dict):
            continue
        if sandbox.get("executable_in_hypothesis_lab") is False:
            continue
        if blueprint.get("candidate_family") in {
            "linear_trend_spectral",
            "learned_residual_correction",
            "task1_retrieval_prior",
            "task1_retrieval_refinement",
            "task1_retrieval_refinement_local_search",
            "task1_early_refinement",
            "task1_early_transition_expert",
            "task1_sem_burgers_rollout",
            "task1_physics_residual_refinement",
            "task1_pi_deeponet_correction_head",
            "task1_front25_dynamics",
        }:
            return blueprint
    return None

def _verified_blueprints(
    *,
    blueprints: dict[str, Any],
    sandbox: dict[str, Any],
) -> dict[str, Any]:
    verified = json.loads(json.dumps(blueprints))
    verified["state_hint"] = VERIFIED_BLUEPRINTS_STATE_HINT
    verified["source_sandbox_report_path"] = sandbox["artifacts"]["sandbox_report_path"]
    best = sandbox["best_candidate"]
    for blueprint in verified.get("blueprints", []):
        if not isinstance(blueprint, dict):
            continue
        if blueprint.get("blueprint_id") != sandbox.get("blueprint_id"):
            continue
        seed = {
            "params": best["params"],
            "source": "sandbox_best",
            "candidate_id": best["candidate_id"],
            "task1_total_score_proxy": best["task1_total_score_proxy"],
            "segment_score": best["segment_score"],
        }
        existing = blueprint.get("seed_params")
        if not isinstance(existing, list):
            existing = []
        blueprint["seed_params"] = [seed, *_dedupe_seed_params(existing, seed["params"])]
    return verified

def _dedupe_seed_params(
    rows: list[Any],
    first_params: dict[str, Any],
) -> list[dict[str, Any]]:
    deduped = []
    first_key = json.dumps(first_params, sort_keys=True)
    seen = {first_key}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
            continue
        key = json.dumps(row["params"], sort_keys=True)
        if key in seen:
            continue
        deduped.append(row)
        seen.add(key)
    return deduped
