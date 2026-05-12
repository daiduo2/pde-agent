"""Bounded Task1 candidate-family factory for the Claude SDK agent."""


import json
import re
from pathlib import Path
from typing import Any


CARD_STATE_HINT = "TASK1_CANDIDATE_FAMILY_CARD_V1"
FAMILY_REPORT_STATE_HINT = "TASK1_CANDIDATE_FAMILIES_GENERATED"
SANDBOX_STATE_HINT = "TASK1_CANDIDATE_FAMILY_SANDBOX_PLANNED"
FACTORY_BENCHMARK_STATE_HINT = "TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED"
MATERIALIZED_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"

VALID_TARGET_SEGMENTS = {"early", "middle", "late", "full", "transition"}
VALID_MECHANISMS = {
    "front25_dynamics",
    "residual_corrector",
    "spectral_physics",
    "checkpoint_adapter",
    "ensemble_gate",
    "fno_spectral_truncation",
    "fno_modewise_correction",
    "fno_rollout_stabilization",
    "fno_multistep_consistency",
    "pinn_physics_residual_judge",
    "pi_deeponet_correction_signal",
    "literature_algorithm_blueprint",
}
VALID_SANDBOX_BUDGETS = {"quick", "medium", "full"}
VALID_STATUSES = {
    "proposed",
    "sandbox_planned",
    "sandbox_blocked",
    "promoted",
    "rejected",
}
SUPPORTED_HOOK_COMMANDS = {
    "hypothesis_lab_front25_dynamics_smoke": "run-hypothesis-lab",
    "hypothesis_lab_early_subsegment_patch_search": "run-hypothesis-lab",
    "official_checkpoint_family": "run-checkpoint-family-sandbox",
    "sem_window_gate_blend": "run-hypothesis-lab",
    "sem_window_blend": "run-hypothesis-lab",
    "task1_physics_residual_refinement": "run-hypothesis-lab",
    "task1_pi_deeponet_correction_head": "run-hypothesis-lab",
    "validate_executable_blueprints": "validate-executable-blueprints",
    "literature_algorithm_v2_pipeline": "run-literature-algorithm-v2-pipeline",
}

HOOK_EVALUATION_CHAINS = {
    "literature_algorithm_v2_pipeline": [
        {
            "command": "validate-algorithm-cards-v2",
            "required_artifact": "algorithm_cards_v2_audit_path",
        },
        {
            "command": "compile-algorithm-candidates",
            "required_artifact": "algorithm_candidates_path",
        },
        {
            "command": "validate-executable-blueprints",
            "required_artifact": "executable_blueprint_audit_path",
        },
        {
            "command": "run-bounded-executor-sandbox",
            "required_artifact": "bounded_executor_sandbox_path",
        },
        {
            "command": "record-bounded-executor-sandbox-event",
            "required_artifact": "literature_registry_event_path",
        },
    ],
}


class CandidateFactoryError(RuntimeError):
    pass


def validate_candidate_family_card(card: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(card, dict):
        raise CandidateFactoryError("candidate card must be a JSON object")
    normalized = dict(card)
    normalized["state_hint"] = normalized.get("state_hint") or CARD_STATE_HINT
    if normalized["state_hint"] != CARD_STATE_HINT:
        raise CandidateFactoryError("state_hint must be TASK1_CANDIDATE_FAMILY_CARD_V1")
    candidate_id = _required_text(normalized, "candidate_family_id")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,96}", candidate_id):
        raise CandidateFactoryError("candidate_family_id must be stable snake_case")
    _required_text(normalized, "family_name")
    task = _required_text(normalized, "task")
    if task != "task1":
        raise CandidateFactoryError("task must be task1")
    _require_enum(normalized, "target_segment", VALID_TARGET_SEGMENTS)
    _require_enum(normalized, "mechanism", VALID_MECHANISMS)
    _required_text(normalized, "hypothesis")
    if not isinstance(normalized.get("parameter_space"), dict) or not normalized[
        "parameter_space"
    ]:
        raise CandidateFactoryError("parameter_space must be a non-empty object")
    _required_text_list(normalized, "implementation_hooks")
    _required_text_list(normalized, "expected_observables")
    _required_text_list(normalized, "risk_factors")
    _require_enum(normalized, "sandbox_budget", VALID_SANDBOX_BUDGETS)
    source = normalized.get("source")
    if not isinstance(source, dict):
        raise CandidateFactoryError("source must be an object")
    _required_text(source, "type")
    _require_enum(normalized, "status", VALID_STATUSES)
    return normalized


def generate_task1_candidate_families(
    *,
    benchmark_path: str | Path,
    chain_report_path: str | Path,
    output_path: str | Path,
    prior_registry_path: str | Path | None = None,
    max_families: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    benchmark_file = _require_file(Path(benchmark_path), "benchmark")
    chain_file = _require_file(Path(chain_report_path), "chain report")
    benchmark = _read_json(benchmark_file)
    chain = _read_json(chain_file)
    prior_ids = _prior_candidate_family_ids(prior_registry_path)
    context = _generation_context(
        benchmark=benchmark,
        chain=chain,
        benchmark_path=benchmark_file,
        chain_report_path=chain_file,
    )
    seeds = _seed_cards(context=context)
    cards = []
    for seed in seeds[: max(0, max_families)]:
        card = dict(seed)
        base_id = str(card["candidate_family_id"])
        card["candidate_family_id"] = _next_candidate_id(base_id, prior_ids)
        prior_ids.add(card["candidate_family_id"])
        cards.append(validate_candidate_family_card(card))
    report = {
        "schema_version": 1,
        "state_hint": FAMILY_REPORT_STATE_HINT,
        "family_report_path": str(output_file),
        "decision": "generated" if cards else "no_candidates",
        "candidate_family_count": len(cards),
        "candidate_families": cards,
        "inputs": {
            "benchmark_path": str(benchmark_file),
            "chain_report_path": str(chain_file),
            "prior_registry_path": str(prior_registry_path)
            if prior_registry_path is not None
            else None,
            "benchmark_decision": benchmark.get("decision"),
            "chain_decision": chain.get("decision"),
        },
        "generation_context": context,
        "next_command": "run-candidate-family-sandbox",
    }
    _write_json(output_file, report)
    return report


def run_candidate_family_sandbox(
    *,
    family_report_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    family_file = _require_file(Path(family_report_path), "family report")
    family_report = _read_json(family_file)
    cards = [
        validate_candidate_family_card(card)
        for card in family_report.get("candidate_families", [])
        if isinstance(card, dict)
    ]
    items = [_sandbox_item(card) for card in cards]
    planned_count = sum(1 for item in items if item["status"] == "planned")
    blocked_count = sum(1 for item in items if item["status"] == "blocked")
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_STATE_HINT,
        "sandbox_report_path": str(output_file),
        "decision": "planned" if planned_count else "blocked",
        "family_report_path": str(family_file),
        "candidate_family_count": len(cards),
        "planned_count": planned_count,
        "blocked_count": blocked_count,
        "sandbox_items": items,
        "next_command": "evaluate-candidate-factory-capability",
    }
    _write_json(output_file, report)
    return report


def evaluate_candidate_factory_capability(
    *,
    family_report_path: str | Path,
    sandbox_report_path: str | Path,
    output_path: str | Path,
    min_family_count: int = 3,
    min_executable_count: int = 1,
    min_mechanism_count: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    family_file = _require_file(Path(family_report_path), "family report")
    sandbox_file = _require_file(Path(sandbox_report_path), "sandbox report")
    family_report = _read_json(family_file)
    sandbox_report = _read_json(sandbox_file)
    cards = [
        validate_candidate_family_card(card)
        for card in family_report.get("candidate_families", [])
        if isinstance(card, dict)
    ]
    mechanisms = sorted({str(card["mechanism"]) for card in cards})
    items = [
        item
        for item in sandbox_report.get("sandbox_items", [])
        if isinstance(item, dict)
    ]
    planned_count = sum(1 for item in items if item.get("status") == "planned")
    checks = {
        "candidate_families": {
            "actual": len(cards),
            "minimum": min_family_count,
            "passed": len(cards) >= min_family_count,
        },
        "mechanism_diversity": {
            "actual": len(mechanisms),
            "minimum": min_mechanism_count,
            "passed": len(mechanisms) >= min_mechanism_count,
        },
        "executable_families": {
            "actual": planned_count,
            "minimum": min_executable_count,
            "passed": planned_count >= min_executable_count,
        },
    }
    decision = "passed" if all(check["passed"] for check in checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "state_hint": FACTORY_BENCHMARK_STATE_HINT,
        "factory_benchmark_path": str(output_file),
        "decision": decision,
        "candidate_space": {
            "family_count": len(cards),
            "mechanisms": mechanisms,
            "mechanism_count": len(mechanisms),
        },
        "sandbox_readiness": {
            "planned_count": planned_count,
            "blocked_count": sum(1 for item in items if item.get("status") == "blocked"),
        },
        "capability_checks": checks,
        "inputs": {
            "family_report_path": str(family_file),
            "sandbox_report_path": str(sandbox_file),
        },
        "agent_runtime_handoff": {
            "ready": decision == "passed",
            "next_command": "run-claude-agent-sdk-runtime",
            "active_substrate": "claude_agent_sdk",
            "candidate_generator_fallback": "auto",
            "sandbox_report_path": str(sandbox_file),
            "planned_candidate_family_ids": [
                str(item["candidate_family_id"])
                for item in items
                if item.get("status") == "planned"
                and isinstance(item.get("candidate_family_id"), str)
            ],
        },
        "next_command": "run-claude-agent-sdk-runtime"
        if decision == "passed"
        else "generate-task1-candidate-families",
    }
    _write_json(output_file, report)
    return report


def materialize_candidate_family_item(
    *,
    sandbox_report_path: str | Path,
    output_path: str | Path,
    candidate_family_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    sandbox_file = _require_file(Path(sandbox_report_path), "candidate family sandbox")
    sandbox_report = _read_json(sandbox_file)
    if sandbox_report.get("state_hint") != SANDBOX_STATE_HINT:
        raise CandidateFactoryError("candidate family sandbox has invalid state_hint")
    item = _select_sandbox_item(
        sandbox_report=sandbox_report,
        candidate_family_id=candidate_family_id,
    )
    if item.get("status") != "planned":
        raise CandidateFactoryError("selected candidate family item is not planned")
    candidate = _algorithm_candidate_from_sandbox_item(
        item=item,
        sandbox_file=sandbox_file,
    )
    report = {
        "schema_version": 1,
        "state_hint": MATERIALIZED_CANDIDATES_STATE_HINT,
        "source_state_hint": SANDBOX_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "algorithm_candidate_count": 1,
        "algorithm_candidates": [candidate],
        "selected_candidate_family_id": item["candidate_family_id"],
        "selected_hook": item["selected_hook"],
        "inputs": {
            "sandbox_report_path": str(sandbox_file),
            "candidate_family_id": candidate_family_id,
        },
        "next_command": str(item["next_command"]),
    }
    _write_json(output_file, report)
    return report


def _seed_cards(*, context: dict[str, Any]) -> list[dict[str, Any]]:
    source = {
        "type": "agent_hypothesis",
        "artifact_paths": [
            str(context.get("benchmark_path")),
            str(context.get("chain_report_path")),
        ],
    }
    return [
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_front25_dynamics_v0",
            "family_name": "Front25 dynamics source",
            "task": "task1",
            "target_segment": "early",
            "mechanism": "front25_dynamics",
            "hypothesis": (
                "Use a bounded dynamics source to improve the first 25 predicted "
                "steps where current local families plateau."
            ),
            "parameter_space": {"window": [10, 15, 20, 25], "blend": [0.1, 0.2, 0.35]},
            "implementation_hooks": ["hypothesis_lab_front25_dynamics_smoke"],
            "expected_observables": ["early_rel_mse_down", "runtime_under_budget"],
            "risk_factors": ["early_oversmoothing"],
            "sandbox_budget": "quick",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_residual_corrector_v0",
            "family_name": "Residual corrector",
            "task": "task1",
            "target_segment": "transition",
            "mechanism": "residual_corrector",
            "hypothesis": (
                "Apply a low-cost residual correction over early transition "
                "windows instead of replacing the whole rollout."
            ),
            "parameter_space": {"scale": [0.05, 0.1, 0.2], "window": [20, 35, 50]},
            "implementation_hooks": ["hypothesis_lab_early_subsegment_patch_search"],
            "expected_observables": ["transition_rel_mse_down", "no_late_regression"],
            "risk_factors": ["late_drift_amplification"],
            "sandbox_budget": "quick",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_spectral_physics_v0",
            "family_name": "Spectral physics correction",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "spectral_physics",
            "hypothesis": (
                "Use spectral-energy and finite-difference residual signals as "
                "bounded correction gates for Burgers rollouts."
            ),
            "parameter_space": {"cutoff": [0.45, 0.55, 0.65], "damping": [0.01, 0.03]},
            "implementation_hooks": ["hypothesis_lab_early_subsegment_patch_search"],
            "expected_observables": ["late_rmse_down", "spectral_energy_stable"],
            "risk_factors": ["shock_oversmoothing"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_checkpoint_adapter_v0",
            "family_name": "Checkpoint adapter",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "checkpoint_adapter",
            "hypothesis": (
                "Wrap official checkpoint predictions with a small adapter or "
                "arbitration layer instead of retraining from scratch."
            ),
            "parameter_space": {"adapter": ["bias", "blend", "window_gate"], "window": [25, 50]},
            "implementation_hooks": ["official_checkpoint_family"],
            "expected_observables": ["score_delta_positive", "inference_under_budget"],
            "risk_factors": ["checkpoint_shape_mismatch"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_fno_spectral_truncation_v0",
            "family_name": "FNO spectral truncation",
            "task": "task1",
            "target_segment": "late",
            "mechanism": "fno_spectral_truncation",
            "hypothesis": (
                "Restrict high-frequency FNO modes during rollout arbitration "
                "to reduce late-horizon spectral noise without retraining."
            ),
            "parameter_space": {
                "mode_cutoff": [0.45, 0.55, 0.65],
                "damping": [0.0, 0.01, 0.03],
                "blend": [0.1, 0.25, 0.5],
            },
            "implementation_hooks": ["official_checkpoint_family"],
            "expected_observables": ["late_rmse_down", "high_frequency_energy_down"],
            "risk_factors": ["shock_oversmoothing", "early_segment_regression"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_fno_modewise_correction_v0",
            "family_name": "FNO mode-wise correction",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "fno_modewise_correction",
            "hypothesis": (
                "Apply small mode-wise amplitude corrections to official FNO "
                "checkpoint rollouts using validation residual spectra."
            ),
            "parameter_space": {
                "mode_band": ["low", "mid", "high"],
                "correction_scale": [0.02, 0.05, 0.1],
                "window": [25, 50, 100],
            },
            "implementation_hooks": ["official_checkpoint_family"],
            "expected_observables": ["segment_score_balance", "spectral_residual_down"],
            "risk_factors": ["validation_overfit", "mode_phase_mismatch"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_fno_rollout_stabilization_v0",
            "family_name": "FNO autoregressive rollout stabilization",
            "task": "task1",
            "target_segment": "late",
            "mechanism": "fno_rollout_stabilization",
            "hypothesis": (
                "Stabilize autoregressive FNO rollouts by damping residual "
                "growth only after the context handoff window."
            ),
            "parameter_space": {
                "handoff": [10, 15, 20],
                "residual_decay": [0.98, 0.995, 1.0],
                "stability_gate": ["energy", "shock", "spectral"],
            },
            "implementation_hooks": ["official_checkpoint_family"],
            "expected_observables": ["late_lorentzian_score_up", "energy_growth_down"],
            "risk_factors": ["underprediction_after_shock", "runtime_overhead"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_fno_multistep_consistency_v0",
            "family_name": "FNO multi-step consistency",
            "task": "task1",
            "target_segment": "transition",
            "mechanism": "fno_multistep_consistency",
            "hypothesis": (
                "Use short multi-step consistency checks to arbitrate between "
                "FNO checkpoint, retrieval, and current best predictions."
            ),
            "parameter_space": {
                "consistency_horizon": [2, 4, 8],
                "agreement_metric": ["l2", "spectral_l2", "gradient_l2"],
                "blend": [0.1, 0.25, 0.4],
            },
            "implementation_hooks": ["official_checkpoint_family"],
            "expected_observables": ["transition_rel_mse_down", "no_late_regression"],
            "risk_factors": ["agreement_proxy_misaligned", "extra_validation_cost"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_failure_conditioned_gate_v0",
            "family_name": "Failure-conditioned ensemble gate",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "ensemble_gate",
            "hypothesis": (
                "Choose between existing strategies using early shock, spectral, "
                "and residual proxies rather than one global blend."
            ),
            "parameter_space": {"proxy": ["shock", "spectral", "residual"], "threshold": [0.2, 0.4, 0.6]},
            "implementation_hooks": ["sem_window_gate_blend", "sem_window_blend"],
            "expected_observables": ["segment_score_balance", "reduced_regression_cases"],
            "risk_factors": ["gate_overfit_to_validation"],
            "sandbox_budget": "quick",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_pinn_physics_residual_judge_v0",
            "family_name": "PINN-style residual judge",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "pinn_physics_residual_judge",
            "hypothesis": (
                "Use Burgers residual proxies as a candidate judge and bounded "
                "correction signal rather than training a full PINN."
            ),
            "parameter_space": {
                "residual_weight": [0.05, 0.1, 0.2],
                "correction_scale": [0.02, 0.05, 0.1],
                "window": [25, 50, 100],
            },
            "implementation_hooks": ["task1_physics_residual_refinement"],
            "expected_observables": ["burgers_residual_down", "score_delta_positive"],
            "risk_factors": ["residual_proxy_misaligned", "oversmoothing_shocks"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_pi_deeponet_correction_signal_v0",
            "family_name": "PI-DeepONet correction signal",
            "task": "task1",
            "target_segment": "transition",
            "mechanism": "pi_deeponet_correction_signal",
            "hypothesis": (
                "Fit a bounded PI-DeepONet-style low-rank correction head that "
                "uses physics residual signals without retraining a large model."
            ),
            "parameter_space": {
                "rank": [2, 4, 8],
                "residual_weight": [0.05, 0.1, 0.2],
                "blend": [0.1, 0.25, 0.4],
            },
            "implementation_hooks": ["task1_pi_deeponet_correction_head"],
            "expected_observables": [
                "pi_deeponet_active_correction",
                "transition_rel_mse_down",
            ],
            "risk_factors": ["inactive_correction", "validation_overfit"],
            "sandbox_budget": "medium",
            "source": source,
            "status": "proposed",
        },
        {
            "state_hint": CARD_STATE_HINT,
            "candidate_family_id": "task1_literature_algorithm_blueprint_v0",
            "family_name": "Literature-to-algorithm blueprint",
            "task": "task1",
            "target_segment": "full",
            "mechanism": "literature_algorithm_blueprint",
            "hypothesis": (
                "Convert extracted AlgorithmCard v2 claims into bounded "
                "Blueprint DSL candidates before any promotion decision."
            ),
            "parameter_space": {
                "operator_family": ["fno", "pinn", "deeponet"],
                "blueprint_op": ["spectral_smooth", "checkpoint_calibration"],
                "sandbox_budget": ["quick"],
            },
            "implementation_hooks": ["literature_algorithm_v2_pipeline"],
            "expected_observables": [
                "blueprint_schema_valid",
                "sandbox_result_recorded",
            ],
            "risk_factors": ["blueprint_schema_invalid", "executor_missing"],
            "sandbox_budget": "quick",
            "source": source,
            "status": "proposed",
        },
    ]


def _sandbox_item(card: dict[str, Any]) -> dict[str, Any]:
    supported = [
        hook
        for hook in card["implementation_hooks"]
        if hook in SUPPORTED_HOOK_COMMANDS
    ]
    if not supported:
        return {
            "candidate_family_id": card["candidate_family_id"],
            "mechanism": card["mechanism"],
            "status": "blocked",
            "blocked_reason": "no_supported_hook",
            "implementation_hooks": card["implementation_hooks"],
        }
    hook = supported[0]
    evaluation_chain = _evaluation_chain(hook)
    return {
        "candidate_family_id": card["candidate_family_id"],
        "mechanism": card["mechanism"],
        "status": "planned",
        "selected_hook": hook,
        "next_command": SUPPORTED_HOOK_COMMANDS[hook],
        "sandbox_budget": card["sandbox_budget"],
        "evaluation_chain": evaluation_chain,
        "required_evidence": [
            step["required_artifact"]
            for step in evaluation_chain
            if isinstance(step.get("required_artifact"), str)
        ],
        "candidate_options": {
            "candidate_family_id": card["candidate_family_id"],
            "mechanism": card["mechanism"],
            "parameter_space": card["parameter_space"],
        },
    }


def _select_sandbox_item(
    *,
    sandbox_report: dict[str, Any],
    candidate_family_id: str | None,
) -> dict[str, Any]:
    items = [
        item
        for item in sandbox_report.get("sandbox_items", [])
        if isinstance(item, dict)
    ]
    if candidate_family_id is not None:
        matches = [
            item
            for item in items
            if item.get("candidate_family_id") == candidate_family_id
        ]
        if not matches:
            raise CandidateFactoryError(
                f"candidate family item not found: {candidate_family_id}"
            )
        return matches[0]
    for item in items:
        if item.get("status") == "planned":
            return item
    raise CandidateFactoryError("candidate family sandbox has no planned items")


def _algorithm_candidate_from_sandbox_item(
    *,
    item: dict[str, Any],
    sandbox_file: Path,
) -> dict[str, Any]:
    family_id = str(item["candidate_family_id"])
    mechanism = str(item.get("mechanism") or "candidate_family")
    selected_hook = str(item["selected_hook"])
    next_command = str(item["next_command"])
    candidate_options = (
        item.get("candidate_options") if isinstance(item.get("candidate_options"), dict)
        else {}
    )
    return {
        "algorithm_candidate_id": f"candidate_{family_id}",
        "candidate_family": "factory_candidate_family",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": _model_family_from_mechanism(mechanism),
        "hypothesis": (
            f"Evaluate candidate-family mechanism `{mechanism}` from the "
            "Task1 candidate factory under a bounded sandbox before promotion."
        ),
        "mechanism": (
            "Materialized from a planned candidate-family sandbox item. "
            f"Selected hook `{selected_hook}` with bounded parameter space."
        ),
        "executor_spec": {
            "cli_command": next_command,
            "implementation_hooks": [next_command, selected_hook],
            "candidate_options": candidate_options,
            "evaluation_chain": item.get("evaluation_chain", []),
        },
        "expected_observables": _expected_observables_from_mechanism(mechanism),
        "risk_controls": _risk_controls_from_mechanism(mechanism),
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": False,
        },
        "source_grounding": {
            "algorithm_card_id": family_id,
            "source_ids": [family_id],
            "source_artifacts": [str(sandbox_file)],
        },
    }


def _model_family_from_mechanism(mechanism: str) -> str:
    if mechanism.startswith("fno_"):
        return "FNO"
    if mechanism == "front25_dynamics":
        return "Task1Front25Dynamics"
    if mechanism in {"pinn_physics_residual_judge", "pi_deeponet_correction_signal"}:
        return "PhysicsInformedCorrection"
    return "Task1CandidateFactory"


def _expected_observables_from_mechanism(mechanism: str) -> list[str]:
    observables = ["sandbox_result_recorded", "context_prefix_exact"]
    if mechanism in {"front25_dynamics", "residual_corrector"}:
        observables.append("early_rel_mse_down")
    elif mechanism.startswith("fno_") or mechanism == "checkpoint_adapter":
        observables.append("score_delta_positive")
    elif "physics" in mechanism or "deeponet" in mechanism:
        observables.append("physical_residual_or_transition_score_improves")
    else:
        observables.append("task1_total_score_proxy_improves")
    return observables


def _risk_controls_from_mechanism(mechanism: str) -> list[str]:
    risks = ["do_not_promote_without_sandbox_evidence"]
    if mechanism in {"front25_dynamics", "residual_corrector"}:
        risks.extend(["early_oversmoothing", "middle_late_regression"])
    elif mechanism.startswith("fno_") or mechanism == "checkpoint_adapter":
        risks.extend(["checkpoint_shape_mismatch", "runtime_over_budget"])
    elif "physics" in mechanism or "deeponet" in mechanism:
        risks.extend(["residual_proxy_misaligned", "validation_overfit"])
    else:
        risks.append("candidate_family_overfit")
    return risks


def _evaluation_chain(hook: str) -> list[dict[str, str]]:
    if hook in HOOK_EVALUATION_CHAINS:
        return [dict(step) for step in HOOK_EVALUATION_CHAINS[hook]]
    return [
        {
            "command": SUPPORTED_HOOK_COMMANDS[hook],
            "required_artifact": "sandbox_report_path",
        }
    ]


def _generation_context(
    *,
    benchmark: dict[str, Any],
    chain: dict[str, Any],
    benchmark_path: Path,
    chain_report_path: Path,
) -> dict[str, Any]:
    score = benchmark.get("score_progress") if isinstance(benchmark, dict) else {}
    if not isinstance(score, dict):
        score = {}
    chain_metrics = chain.get("metrics") if isinstance(chain, dict) else {}
    if not isinstance(chain_metrics, dict):
        chain_metrics = {}
    sandbox = benchmark.get("sandbox_validation") if isinstance(benchmark, dict) else {}
    if not isinstance(sandbox, dict):
        sandbox = {}
    outcomes = sandbox.get("outcome_counts") if isinstance(sandbox, dict) else {}
    if not isinstance(outcomes, dict):
        outcomes = {}
    return {
        "benchmark_path": str(benchmark_path),
        "chain_report_path": str(chain_report_path),
        "benchmark_decision": benchmark.get("decision"),
        "chain_decision": chain.get("decision"),
        "best_score": _float(score.get("best_score")),
        "score_delta": _float(score.get("score_delta")),
        "stagnation_count": int(score.get("stagnation_count") or 0),
        "candidate_space_exhausted_count": int(
            chain_metrics.get("candidate_space_exhausted_count")
            or outcomes.get("candidate_space_exhausted")
            or 0
        ),
    }


def _prior_candidate_family_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    file = Path(path)
    if not file.exists():
        return set()
    payload = _read_json(file)
    values = payload.get("candidate_families", payload.get("cards", []))
    return {
        str(card.get("candidate_family_id"))
        for card in values
        if isinstance(card, dict) and card.get("candidate_family_id")
    }


def _next_candidate_id(base_id: str, used: set[str]) -> str:
    if base_id not in used:
        return base_id
    match = re.fullmatch(r"(.+)_v(\d+)", base_id)
    stem = match.group(1) if match else base_id
    version = int(match.group(2)) + 1 if match else 1
    candidate = f"{stem}_v{version}"
    while candidate in used:
        version += 1
        candidate = f"{stem}_v{version}"
    return candidate


def _require_enum(card: dict[str, Any], key: str, allowed: set[str]) -> str:
    value = _required_text(card, key)
    if value not in allowed:
        raise CandidateFactoryError(f"{key} must be one of {sorted(allowed)}")
    return value


def _required_text(card: dict[str, Any], key: str) -> str:
    value = card.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CandidateFactoryError(f"{key} must be a non-empty string")
    return value


def _required_text_list(card: dict[str, Any], key: str) -> list[str]:
    value = card.get(key)
    if not isinstance(value, list) or not value:
        raise CandidateFactoryError(f"{key} must be a non-empty list")
    values = [item for item in value if isinstance(item, str) and item.strip()]
    if len(values) != len(value):
        raise CandidateFactoryError(f"{key} must contain only non-empty strings")
    return values


def _float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CandidateFactoryError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise CandidateFactoryError(f"{label} not found: {path}")
    return path


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise CandidateFactoryError(f"output already exists: {path}")
