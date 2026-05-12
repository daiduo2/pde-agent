"""Offline artifact backbone for PDE literature-to-algorithm candidates."""


import ast
import hashlib
import importlib.util
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_algorithm_card_extraction import (
    _markdown_heading_blocks,
    extract_algorithm_cards,
    ingest_mineru_parse,
)
from pde_agent.core.literature_algorithm_card_validation import (
    validate_algorithm_cards_v2,
)
from pde_agent.core.literature_bounded_executor import (
    audit_bounded_executor_authoring_task,
    complete_bounded_executor_authoring_task,
    create_bounded_executor_design_spec,
    materialize_bounded_executor_template,
    prepare_bounded_executor_authoring_task,
    record_bounded_executor_sandbox_event,
    record_bounded_executor_sandbox_evidence,
    run_bounded_executor_sandbox,
    validate_bounded_executor_implementation,
)
from pde_agent.core.literature_blueprint_validation import (
    _first_candidate_id,
    _materialize_executable_blueprints_from_candidates,
    audit_executable_blueprint_payload,
    validate_executable_blueprints,
)
from pde_agent.core.literature_candidate_compilation import (
    compile_algorithm_candidates,
    plan_candidate_sandbox,
)
from pde_agent.core.literature_registry import (
    build_literature_source_records,
    record_literature_registry_event,
    update_literature_registry,
)
from pde_agent.core.literature_candidate_promotion import (
    evaluate_algorithm_candidate_promotion,
)
from pde_agent.core.literature_source_acquisition import (
    MinerUHttpClient,
    download_mineru_parse_result,
    download_source_document,
    normalize_arxiv_search_results,
    normalize_github_search_results,
    normalize_openalex_search_results,
    poll_mineru_parse_task,
    run_arxiv_mineru_parse_pipeline,
    submit_mineru_parse_task,
)


SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
SOURCE_SEARCH_STATE_HINT = "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
MINERU_TASK_SUBMITTED_STATE_HINT = "MINERU_PARSE_TASK_SUBMITTED"
MINERU_TASK_POLLED_STATE_HINT = "MINERU_PARSE_TASK_POLLED"
MINERU_RESULT_DOWNLOADED_STATE_HINT = "MINERU_PARSE_RESULT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT = "ALGORITHM_CARDS_V2_VALIDATED"
ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT = "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
LITERATURE_REGISTRY_EVENT_STATE_HINT = "LITERATURE_REGISTRY_EVENT_RECORDED"
EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT = "EXECUTABLE_BLUEPRINTS_VALIDATED"
SANDBOX_PLAN_STATE_HINT = "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"
EXECUTOR_TAXONOMY_STATE_HINT = "EXECUTOR_TAXONOMY_BUILT"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
EXECUTOR_DESIGN_REQUEST_STATE_HINT = "EXECUTOR_DESIGN_REQUEST_CREATED"
BLOCKED_CANDIDATE_HANDOFF_STATE_HINT = "BLOCKED_CANDIDATE_HANDOFF_BUILT"
LITERATURE_REGISTRY_UPDATED_STATE_HINT = "LITERATURE_REGISTRY_UPDATED"
LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT = "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"
ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT = "ARXIV_MINERU_PARSE_PIPELINE_RAN"
LITERATURE_EXTRACTION_CONTEXT_STATE_HINT = "LITERATURE_EXTRACTION_CONTEXT_PREPARED"
FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT = "FNO_LITERATURE_ADAPTATION_TASK_PREPARED"
DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT = (
    "DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED"
)
UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT = "UNET_LITERATURE_ADAPTATION_TASK_PREPARED"
BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT = "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED"
BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT = "BOUNDED_EXECUTOR_TEMPLATE_MATERIALIZED"
BOUNDED_EXECUTOR_AUDIT_STATE_HINT = "BOUNDED_EXECUTOR_IMPLEMENTATION_VALIDATED"
BOUNDED_EXECUTOR_SANDBOX_STATE_HINT = "BOUNDED_EXECUTOR_SANDBOX_RAN"
BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT = "BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED"
BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT = (
    "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED"
)
BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT = (
    "BOUNDED_EXECUTOR_AUTHORING_COMPLETED"
)

VALID_SOURCE_TYPES = {
    "openalex_work",
    "arxiv_paper",
    "github_repo",
    "competition_resource",
    "manual_source",
}

MINERU_AGENT_URL_ENDPOINT = "https://mineru.net/api/v1/agent/parse/url"
MINERU_AGENT_POLL_ENDPOINT = "https://mineru.net/api/v1/agent/parse"
MINERU_PRECISE_URL_ENDPOINT = "https://mineru.net/api/v4/extract/task"

HOOK_TO_FAMILY = {
    "official_checkpoint_calibration": {
        "candidate_family": "official_checkpoint_family",
        "model_family": "checkpoint",
        "cli_command": "run-checkpoint-family-sandbox",
        "sandbox_level": 1,
    },
    "learned_residual_correction": {
        "candidate_family": "lightweight_residual_corrector",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "spectral_rollout_stabilization": {
        "candidate_family": "spectral_rollout_stabilization",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "physics_residual_regularization": {
        "candidate_family": "physics_residual_regularization",
        "model_family": "FNO",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 1,
    },
    "scheduled_rollout_training": {
        "candidate_family": "scheduled_rollout_training",
        "model_family": "FNO",
        "cli_command": "execute-fno-finetune-candidate",
        "sandbox_level": 1,
    },
    "operator_correction_postprocess": {
        "candidate_family": "operator_correction_postprocess",
        "model_family": "postprocess",
        "cli_command": "run-hypothesis-lab",
        "sandbox_level": 0,
    },
}

HOOK_ALIASES = {
    "physics_informed_loss": "physics_residual_regularization",
    "pinn_residual_loss": "physics_residual_regularization",
    "pi_deeponet_physics_loss": "physics_residual_regularization",
    "scheduled_sampling": "scheduled_rollout_training",
    "rollout_training": "scheduled_rollout_training",
    "autoregressive_rollout_training": "scheduled_rollout_training",
    "postprocess_operator_correction": "operator_correction_postprocess",
    "residual_blend_postprocess": "operator_correction_postprocess",
}

HOOK_TO_ALGORITHM_CARD_V2 = {
    "official_checkpoint_calibration": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "checkpoint",
        "mathematical_object": "checkpoint_weights",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": True,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "learned_residual_correction": {
        "operator_target": "residual_to_correction",
        "mechanism_type": "postprocess",
        "mathematical_object": "residual_operator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": True,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "spectral_rollout_stabilization": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "spectral_filter",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "physics_residual_regularization": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "loss",
        "mathematical_object": "burgers_residual",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "scheduled_rollout_training": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "rollout",
        "mathematical_object": "time_integrator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "operator_correction_postprocess": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "correction_operator",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "fourier_neural_operator_architecture": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "architecture",
        "mathematical_object": "fourier_modes",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": True,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
}

ALGORITHM_CARD_V2_OPERATOR_TARGETS = {
    "ic_to_trajectory",
    "coefficient_to_solution",
    "residual_to_correction",
    "context_to_rollout",
    "postprocess_to_solution",
}

ALGORITHM_CARD_V2_MECHANISM_TYPES = {
    "architecture",
    "loss",
    "rollout",
    "postprocess",
    "conditioning",
    "checkpoint",
    "hybrid",
}

ALGORITHM_CARD_V2_EVIDENCE_STATUSES = {
    "extracted",
    "candidate_compiled",
    "sandbox_planned",
    "tested",
    "promoted",
    "rejected",
    "blocked",
}

ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS = {
    "requires_training",
    "requires_checkpoint",
    "requires_target_data",
    "requires_new_executor",
    "requires_external_code",
    "can_run_as_postprocess",
}

FAILURE_TAXONOMY = {
    "source_download_failed",
    "parse_failed",
    "parse_quality_insufficient",
    "no_algorithmic_claim",
    "unsupported_executor",
    "data_contract_incompatible",
    "requires_external_code",
    "candidate_schema_invalid",
    "blueprint_schema_invalid",
    "sandbox_rule_failed",
    "context_mismatch",
    "target_leakage_risk",
    "score_regressed",
    "runtime_budget_exceeded",
    "physics_regressed",
    "ablation_missing_or_weak",
    "insufficient_reproducibility",
    "runtime_exceeded",
    "duplicate_family",
    "insufficient_evidence",
    "reproducibility_failed",
}

EXECUTABLE_BLUEPRINT_BASELINES = {
    "persistence",
    "fno_checkpoint",
    "retrieval_prior",
}

EXECUTABLE_BLUEPRINT_TRANSFORMS = {
    "linear_trend",
    "spectral_smooth",
    "residual_fit",
    "retrieval_prior",
    "retrieval_refinement",
    "checkpoint_calibration",
}

EXECUTABLE_BLUEPRINT_TRAINING_MODES = {
    "none",
    "finetune",
    "supervised",
    "physics_loss",
}

EXECUTABLE_BLUEPRINT_EXECUTORS = {
    "controlled_numpy_operator",
    "controlled_numpy_residual_operator",
    "controlled_numpy_retrieval_prior",
    "controlled_numpy_retrieval_refinement",
    "controlled_numpy_task1_early_refinement",
    "controlled_numpy_sem_burgers_rollout",
    "controlled_numpy_burgers_residual_refinement",
    "controlled_numpy_pi_deeponet_correction_head",
    "official_checkpoint_candidate_pipeline",
}

EXECUTOR_DESIGN_HOOKS = {
    "fourier_neural_operator_architecture": {
        "route": "requires_executor_design",
        "requested_executor_id": "bounded_fno_architecture_executor",
        "fallback_route": "official_checkpoint_family",
        "rationale": (
            "A paper-level FNO architecture claim needs bounded local implementation "
            "or an official-checkpoint adapter before sandbox execution."
        ),
    }
}


def record_agent_algorithm_cards(
    *,
    cards_json: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    try:
        payload = json.loads(cards_json)
    except json.JSONDecodeError as exc:
        raise LiteratureAlgorithmLabError(f"cards_json is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LiteratureAlgorithmLabError("cards_json must decode to an object")
    if payload.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "agent algorithm cards must have ALGORITHM_CARDS_EXTRACTED state_hint"
        )
    cards = payload.get("algorithm_cards")
    if not isinstance(cards, list):
        raise LiteratureAlgorithmLabError("algorithm_cards must be a list")
    payload.setdefault("schema_version", 1)
    payload["algorithm_cards_path"] = str(output_file)
    payload["algorithm_card_count"] = len(cards)
    payload.setdefault("extraction_status", "extracted" if cards else "empty")
    payload.setdefault("next_command", "validate-algorithm-cards-v2")
    _write_json(output_file, payload)
    return payload


def prepare_literature_extraction_context(
    *,
    paper_parse_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    parse_file = _require_file(Path(paper_parse_path), "paper parse")
    parse = _read_json(parse_file)
    if parse.get("state_hint") != PAPER_PARSE_STATE_HINT:
        raise LiteratureAlgorithmLabError("paper parse has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    source_id = _nonempty_text(parse.get("source_id"), "source_id")
    markdown_file = _require_file(
        Path(_nonempty_text(parse.get("markdown_path"), "markdown_path")),
        "parsed markdown",
    )
    markdown = markdown_file.read_text(encoding="utf-8")
    context_units = []
    for block in _markdown_heading_blocks(markdown):
        section_title = block.get("section_title")
        text = str(block["markdown"]).strip()
        if (
            int(block["block_index"]) == 0
            and block["line_start"] == block["line_end"]
            and text.startswith("# ")
        ):
            section_title = "Document Title"
        unit_id = f"section_{int(block['block_index']):03d}_{_slug(str(section_title or 'preamble'))}"
        context_units.append(
            {
                "unit_id": unit_id,
                "source_id": source_id,
                "section_title": section_title,
                "block_index": block["block_index"],
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "char_count": len(text),
                "contains_formula": "$$" in text or "\\(" in text or "\\[" in text,
                "text": text,
            }
        )
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_EXTRACTION_CONTEXT_STATE_HINT,
        "paper_parse_path": str(parse_file),
        "source_id": source_id,
        "markdown_path": str(markdown_file),
        "parse_quality": parse.get("parse_quality", {}),
        "section_index": parse.get("section_index", {}),
        "context_unit_count": len(context_units),
        "context_units": context_units,
        "agent_contract": {
            "output_state_hint": ALGORITHM_CARDS_STATE_HINT,
            "output_schema": "AlgorithmCard v2 report",
            "grounding_rules": [
                "Every algorithmic claim must cite evidence_spans from context_units.",
                "Do not invent paper claims that are absent from the context text.",
                "Use domain expertise to interpret mechanisms, but preserve source grounding.",
                "Return extraction_status empty with failure_code no_algorithmic_claim when no actionable algorithm is present.",
            ],
            "allowed_existing_hooks": sorted(HOOK_TO_FAMILY),
            "unsupported_hook_policy": (
                "Use a new descriptive hook only when the paper mechanism cannot be "
                "represented by an allowed_existing_hook; such hooks will be routed "
                "through executor design before sandbox execution."
            ),
            "required_card_fields": [
                "algorithm_card_id",
                "source_ids",
                "claim",
                "core_mechanism",
                "operation",
                "implementation_hooks",
                "expected_observables",
                "failure_modes",
                "minimal_experiment",
                "evidence_spans",
                "evidence_status",
                "operator_target",
                "mechanism_type",
                "mathematical_object",
                "implementation_requirements",
            ],
            "optional_card_fields": {
                "mutation_evidence": {
                    "purpose": (
                        "Use only when the extracted algorithm is a deliberate "
                        "variant of a previously deprioritized candidate family."
                    ),
                    "required_when_present": [
                        "parent_candidate_family",
                        "mutation_type",
                        "reason",
                    ],
                    "notes": [
                        "mutation_type must name the concrete algorithmic change.",
                        "reason should cite the failure or deprioritization signal.",
                    ],
                }
            },
        },
        "next_command": "agent-extract-algorithm-cards",
    }
    _write_json(output_file, report)
    return report


OPERATOR_LITERATURE_ADAPTATION_PROFILES: dict[str, dict[str, Any]] = {
    "fno": {
        "state_hint": FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "fno_literature_adaptation_task_path",
        "operator_family": "FNO",
        "skill_name": "pde-fno-literature-adapter",
        "next_command": "agent-extract-fno-algorithm-cards",
        "keywords": [
            "fourier",
            "spectral",
            "operator",
            "convolution",
            "mode",
            "resolution",
            "burgers",
        ],
        "reading_focus": [
            "spectral convolution and Fourier mode truncation",
            "lifting and projection around Fourier layers",
            "resolution or discretization generalization claims",
            "Burgers-specific rollout stability and high-frequency behavior",
            "training rollout, checkpoint, or calibration procedures",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "spectral_rollout_stabilization",
                "official_checkpoint_calibration",
                "physics_residual_regularization",
                "scheduled_rollout_training",
            ],
            "requires_bounded_executor": [
                "fourier_neural_operator_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "ic_to_trajectory",
                "context_to_rollout",
                "postprocess_to_solution",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "rollout",
                "postprocess",
                "checkpoint",
                "loss",
            ],
            "required_observables": [
                "local judge score or segment score",
                "late-horizon RMSE or spectral-energy stability",
                "runtime under task budget",
                "first-context-step identity preservation",
            ],
        },
        "task": (
            "Use FNO domain knowledge to turn grounded context units into "
            "AlgorithmCard v2 records without fabricating claims."
        ),
    },
    "deeponet": {
        "state_hint": DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "deeponet_literature_adaptation_task_path",
        "operator_family": "DeepONet",
        "skill_name": "pde-deeponet-literature-adapter",
        "next_command": "agent-extract-deeponet-algorithm-cards",
        "keywords": [
            "deeponet",
            "deep operator",
            "branch",
            "trunk",
            "sensor",
            "coordinate",
            "pi-deeponet",
            "physics-informed",
            "operator",
            "burgers",
        ],
        "reading_focus": [
            "branch network encoding of input functions or sensors",
            "trunk network coordinate basis and branch-trunk merge rule",
            "PI-DeepONet residual or physics-informed loss terms",
            "rollout construction from pointwise operator evaluations",
            "Burgers context-window conditioning and target coordinate layout",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "physics_residual_regularization",
                "scheduled_rollout_training",
                "operator_correction_postprocess",
            ],
            "requires_bounded_executor": [
                "deeponet_branch_trunk_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "ic_to_trajectory",
                "context_to_rollout",
                "coordinate_to_solution",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "loss",
                "conditioning",
                "rollout",
                "hybrid",
            ],
            "required_observables": [
                "local judge score or segment score",
                "coordinate-grid reconstruction error",
                "physics residual when a residual loss is claimed",
                "runtime under task budget",
                "first-context-step identity preservation",
            ],
        },
        "task": (
            "Use DeepONet and PI-DeepONet domain knowledge to turn grounded "
            "context units into AlgorithmCard v2 records without fabricating claims."
        ),
    },
    "unet": {
        "state_hint": UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "unet_literature_adaptation_task_path",
        "operator_family": "UNet",
        "skill_name": "pde-unet-literature-adapter",
        "next_command": "agent-extract-unet-algorithm-cards",
        "keywords": [
            "u-net",
            "unet",
            "encoder",
            "decoder",
            "skip",
            "convolution",
            "rollout",
            "frame",
            "checkpoint",
            "burgers",
        ],
        "reading_focus": [
            "encoder-decoder hierarchy and skip connection mechanism",
            "temporal frame prediction, push-forward, or autoregressive rollout",
            "checkpoint, finetuning, or calibration procedures",
            "stability controls for long-horizon generated frames",
            "Burgers context-window conditioning and output-frame semantics",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "official_checkpoint_calibration",
                "scheduled_rollout_training",
                "operator_correction_postprocess",
            ],
            "requires_bounded_executor": [
                "unet_encoder_decoder_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "context_to_rollout",
                "postprocess_to_solution",
                "frame_to_frame",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "rollout",
                "checkpoint",
                "postprocess",
                "hybrid",
            ],
            "required_observables": [
                "local judge score or segment score",
                "late-horizon RMSE or drift",
                "runtime under task budget",
                "context preservation",
                "failure evidence for oversmoothing or unstable rollout",
            ],
        },
        "task": (
            "Use UNet operator-family knowledge to turn grounded context units "
            "into AlgorithmCard v2 records without fabricating claims."
        ),
    },
}


def prepare_fno_literature_adaptation_task(
    *,
    literature_extraction_context_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    return _prepare_operator_literature_adaptation_task(
        profile_key="fno",
        literature_extraction_context_path=literature_extraction_context_path,
        output_path=output_path,
        force=force,
    )


def prepare_deeponet_literature_adaptation_task(
    *,
    literature_extraction_context_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    return _prepare_operator_literature_adaptation_task(
        profile_key="deeponet",
        literature_extraction_context_path=literature_extraction_context_path,
        output_path=output_path,
        force=force,
    )


def prepare_unet_literature_adaptation_task(
    *,
    literature_extraction_context_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    return _prepare_operator_literature_adaptation_task(
        profile_key="unet",
        literature_extraction_context_path=literature_extraction_context_path,
        output_path=output_path,
        force=force,
    )


def _prepare_operator_literature_adaptation_task(
    *,
    profile_key: str,
    literature_extraction_context_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    profile = OPERATOR_LITERATURE_ADAPTATION_PROFILES[profile_key]
    context_file = _require_file(
        Path(literature_extraction_context_path),
        "literature extraction context",
    )
    context = _read_json(context_file)
    if context.get("state_hint") != LITERATURE_EXTRACTION_CONTEXT_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "literature extraction context has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    context_units = _object_list(context.get("context_units"), "context_units")
    selected_units = _operator_relevant_context_units(
        context_units,
        keywords=profile["keywords"],
    )
    selected_ids = [
        _nonempty_text(unit.get("unit_id"), "unit_id")
        for unit in selected_units
    ]
    report = {
        "schema_version": 1,
        "state_hint": profile["state_hint"],
        profile["path_key"]: str(output_file),
        "literature_extraction_context_path": str(context_file),
        "source_id": context.get("source_id"),
        "operator_family": profile["operator_family"],
        "selected_context_unit_count": len(selected_units),
        "selected_context_unit_ids": selected_ids,
        "reading_focus": profile["reading_focus"],
        "recommended_hooks": profile["recommended_hooks"],
        "algorithm_card_guidance": profile["algorithm_card_guidance"],
        "agent_contract": {
            "skill_name": profile["skill_name"],
            "base_skill_name": "pde-algorithm-card-extractor",
            "input_state_hint": LITERATURE_EXTRACTION_CONTEXT_STATE_HINT,
            "output_state_hint": ALGORITHM_CARDS_STATE_HINT,
            "task": profile["task"],
            "forbidden_actions": [
                "do not infer implementation details absent from the paper context",
                "do not bypass AlgorithmCard v2 validation",
                "do not turn an architecture claim into runnable code without bounded executor sandboxing",
            ],
        },
        "next_command": profile["next_command"],
    }
    _write_json(output_file, report)
    return report


def run_literature_algorithm_v2_pipeline(
    *,
    algorithm_cards_path: str | Path,
    workspace_dir: str | Path,
    registry_path: str | Path,
    current_best_score: float,
    algorithm_candidate_id: str | None = None,
    sandbox_evidence_paths: list[str | Path] | None = None,
    paper_parse_paths: list[str | Path] | None = None,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.literature_expansion import (
        summarize_literature_registry,
    )

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    cards_file = _require_file(Path(algorithm_cards_path), "algorithm cards")
    card_audit_path = workspace / "algorithm_cards_v2_audit.json"
    candidates_path = workspace / "algorithm_candidates.json"
    blueprints_path = workspace / "executable_blueprints.json"
    blueprint_audit_path = workspace / "executable_blueprint_audit.json"
    registry_file = Path(registry_path)
    summary_path = workspace / "literature_registry_summary.json"

    card_audit = validate_algorithm_cards_v2(
        algorithm_cards_path=cards_file,
        output_path=card_audit_path,
        force=force,
    )
    candidates = compile_algorithm_candidates(
        algorithm_cards_path=cards_file,
        output_path=candidates_path,
        force=force,
    )
    blueprints = _materialize_executable_blueprints_from_candidates(
        candidates=candidates,
        output_path=blueprints_path,
        force=force,
    )
    blueprint_audit = validate_executable_blueprints(
        blueprints_path=blueprints["executable_blueprints_path"],
        output_path=blueprint_audit_path,
        force=force,
    )

    artifacts: dict[str, Any] = {
        "algorithm_cards_v2_audit_path": card_audit["algorithm_card_v2_audit_path"],
        "algorithm_candidates_path": candidates["algorithm_candidates_path"],
        "executable_blueprints_path": blueprints["executable_blueprints_path"],
        "executable_blueprint_audit_path": blueprint_audit[
            "executable_blueprint_audit_path"
        ],
    }
    evidence_paths = sandbox_evidence_paths or []
    promotion = None
    registry_event = None
    if evidence_paths:
        selected_candidate_id = algorithm_candidate_id or _first_candidate_id(candidates)
        promotion_path = workspace / "candidate_promotion.json"
        promotion = evaluate_algorithm_candidate_promotion(
            algorithm_candidates_path=candidates_path,
            algorithm_candidate_id=selected_candidate_id,
            sandbox_evidence_paths=evidence_paths,
            output_path=promotion_path,
            current_best_score=float(current_best_score),
            force=force,
        )
        event_path = workspace / "literature_registry_event.json"
        registry_event = record_literature_registry_event(
            event_source_path=promotion_path,
            output_path=event_path,
            force=force,
        )
        artifacts["candidate_promotion_path"] = promotion[
            "algorithm_candidate_promotion_path"
        ]
        artifacts["literature_registry_event_path"] = registry_event[
            "literature_registry_event_path"
        ]

    event_paths = (
        [registry_event["literature_registry_event_path"]]
        if registry_event is not None
        else []
    )
    registry = update_literature_registry(
        registry_path=registry_file,
        paper_parse_paths=[Path(path) for path in (paper_parse_paths or [])],
        algorithm_cards_paths=[cards_file],
        algorithm_candidates_paths=[candidates_path],
        event_paths=event_paths,
        force=force,
    )
    summary = summarize_literature_registry(
        registry_path=registry_file,
        output_path=summary_path,
        current_best_score=float(current_best_score),
        force=force,
    )
    artifacts["literature_registry_path"] = registry["registry_path"]
    artifacts["literature_registry_summary_path"] = summary["summary_path"]
    pipeline_decision = (
        "promotion_recorded" if promotion is not None else "awaiting_sandbox_evidence"
    )
    next_command = (
        str(promotion["next_command"])
        if promotion is not None
        else str(blueprint_audit["next_command"])
    )
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT,
        "workspace_dir": str(workspace),
        "algorithm_cards_path": str(cards_file),
        "current_best_score": float(current_best_score),
        "pipeline_decision": pipeline_decision,
        "artifacts": artifacts,
        "metrics": {
            "card_issue_count": card_audit["issue_count"],
            "candidate_count": candidates["algorithm_candidate_count"],
            "blueprint_count": blueprints["blueprint_count"],
            "blueprint_issue_count": blueprint_audit["issue_count"],
            "registry_event_count": registry["event_count"],
            "candidate_action_count": summary["metrics"][
                "candidate_action_count"
            ],
        },
        "next_command": next_command,
    }
    _write_json(workspace / "literature_algorithm_v2_pipeline.json", report)
    return report


def build_executor_taxonomy(
    *,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    hook_mappings: dict[str, dict[str, Any]] = {}
    for hook, family_spec in HOOK_TO_FAMILY.items():
        hook_mappings[hook] = {
            "route": "existing_executor",
            "candidate_family": family_spec["candidate_family"],
            "model_family": family_spec["model_family"],
            "cli_command": family_spec["cli_command"],
            "sandbox_level": family_spec["sandbox_level"],
        }
    hook_mappings.update(EXECUTOR_DESIGN_HOOKS)
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_TAXONOMY_STATE_HINT,
        "executor_taxonomy_path": str(output_file),
        "hook_mappings": hook_mappings,
        "route_counts": {
            "existing_executor": len(HOOK_TO_FAMILY),
            "requires_executor_design": len(EXECUTOR_DESIGN_HOOKS),
        },
        "next_command": "create-executor-design-request",
    }
    _write_json(output_file, report)
    return report


def adapt_fno_architecture_candidate(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        selected_id,
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("unsupported_hook") != "fourier_neural_operator_architecture":
        raise LiteratureAlgorithmLabError(
            "FNO architecture adapter requires a fourier_neural_operator_architecture candidate"
        )
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint resource")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    adapted_source_grounding = {
        **source_grounding,
        "adapter_source_candidate_id": selected_id,
        "adapter_route": "official_checkpoint_family",
    }
    adapted_candidate = {
        "algorithm_candidate_id": f"{selected_id}_official_checkpoint_adapter",
        "parent_algorithm_card_id": candidate.get("parent_algorithm_card_id"),
        "candidate_family": "official_checkpoint_family",
        "target_problem": candidate.get("target_problem", "PDE_Burgers"),
        "target_task": candidate.get("target_task", "task1"),
        "model_family": "checkpoint",
        "hypothesis": candidate.get("hypothesis"),
        "mechanism": (
            f"{candidate.get('mechanism')} Adapted through a local official "
            "checkpoint-family sandbox instead of unbounded architecture code."
        ),
        "executor_spec": _executor_spec("run-checkpoint-family-sandbox"),
        "parameter_grid": _default_parameter_grid("official_checkpoint_family"),
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
            "do not execute external repo code",
            "verify checkpoint tensor convention before promotion",
        ],
        "source_grounding": adapted_source_grounding,
    }
    report = {
        "schema_version": 1,
        "state_hint": FNO_CANDIDATE_ADAPTED_STATE_HINT,
        "adapted_candidate_path": str(output_file),
        "algorithm_candidates_path": str(output_file),
        "source_algorithm_candidate_id": selected_id,
        "algorithm_candidate_count": 1,
        "adapted_candidate": adapted_candidate,
        "algorithm_candidates": [adapted_candidate],
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    _write_json(output_file, report)
    return report


def create_executor_design_request(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("cli_command") != "blocked_requires_executor_design":
        raise LiteratureAlgorithmLabError(
            "executor design request requires a blocked candidate"
        )
    hook = _nonempty_text(executor.get("unsupported_hook"), "unsupported_hook")
    hook_spec = EXECUTOR_DESIGN_HOOKS.get(hook, {})
    requested_executor_id = str(
        hook_spec.get("requested_executor_id", f"bounded_{_slug(hook)}_executor")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    strategy_id = f"external_fno_executor_design_{_slug(str(candidate['algorithm_candidate_id']))}"
    hypothesis = _nonempty_text(candidate.get("hypothesis"), "hypothesis")
    mechanism = _nonempty_text(candidate.get("mechanism"), "mechanism")
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_DESIGN_REQUEST_STATE_HINT,
        "executor_design_request_path": str(output_file),
        "algorithm_candidate_id": candidate["algorithm_candidate_id"],
        "candidate_family": candidate.get("candidate_family"),
        "unsupported_hook": hook,
        "requested_executor_id": requested_executor_id,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "source_grounding": source_grounding,
        "reason_codes": [
            "unsupported_algorithm_hook",
            "requires_bounded_executor",
        ],
        "executor_contract": {
            "inputs": [
                "algorithm_candidate_path",
                "validation_windows_path",
                "optional_checkpoint_path",
                "output_dir",
            ],
            "outputs": [
                "candidate_prediction_path",
                "candidate_execution_report_path",
                "sandbox_result_path",
            ],
            "forbidden_actions": [
                "execute external repository code directly",
                "mutate source training data",
                "skip first-10-step identity validation",
            ],
        },
        "minimal_experiment": {
            "level": 2,
            "description": (
                "Implement a bounded executor adapter, run it on local Burgers "
                "validation windows, then compare against persistence and current best."
            ),
        },
        "risk_controls": list(candidate.get("risk_controls", []))
        + [
            "require local sandbox score before promotion",
            "record failure even when executor design is abandoned",
        ],
        "recommended_intermediate_actions": [
            "search official FNO checkpoint resource",
            "map checkpoint resource to run-checkpoint-family-sandbox when available",
            "otherwise design bounded_fno_architecture_executor",
        ],
        "strategy_proposal_fields": {
            "strategy_id": strategy_id,
            "title": "Design bounded FNO executor from literature candidate",
            "rationale": hypothesis,
            "implementation_plan": [
                "Translate the literature FNO mechanism into a bounded local executor contract.",
                "Prefer official checkpoint-family execution if a compatible checkpoint is available.",
                "Run a local Burgers sandbox and require score, runtime, and first-context identity evidence before promotion.",
            ],
            "expected_observables": [
                "higher local judge score than persistence or current best",
                "stable inference under the two-minute task budget",
            ],
            "known_risks": [
                "FNO tensor conventions may not match official checkpoints",
                "architecture reproduction may consume budget without improving long-horizon score",
            ],
            "source_type": "external_retrieval",
            "source_grounding": source_grounding,
        },
        "next_command": "build-blocked-candidate-handoff",
    }
    _write_json(output_file, report)
    return report


def build_blocked_candidate_handoff(
    *,
    executor_design_request_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    request = _read_json(
        _require_file(Path(executor_design_request_path), "executor design request")
    )
    if request.get("state_hint") != EXECUTOR_DESIGN_REQUEST_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "executor design request has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    proposal_fields = request.get("strategy_proposal_fields")
    if not isinstance(proposal_fields, dict):
        raise LiteratureAlgorithmLabError(
            "executor design request missing strategy_proposal_fields"
        )
    report = {
        "schema_version": 1,
        "state_hint": BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
        "blocked_candidate_handoff_path": str(output_file),
        "algorithm_candidate_id": request.get("algorithm_candidate_id"),
        "record_strategy_proposal_fields": proposal_fields,
        "blocked_evidence_summary": {
            "blocked": True,
            "requires_executor": request.get("requested_executor_id"),
            "reason_codes": request.get("reason_codes", []),
            "source_grounding": request.get("source_grounding", {}),
        },
        "next_command": "record-strategy-proposal",
    }
    _write_json(output_file, report)
    return report


def _blocked_candidate(*, card: dict[str, Any], hook: str) -> dict[str, Any]:
    candidate_id = _algorithm_candidate_id(
        card_id=str(card["algorithm_card_id"]),
        family="blocked",
        hook=hook,
    )
    return {
        "algorithm_candidate_id": candidate_id,
        "parent_algorithm_card_id": card["algorithm_card_id"],
        "candidate_family": "unsupported",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "hybrid",
        "hypothesis": card["claim"],
        "mechanism": card["core_mechanism"],
        "executor_spec": {
            "cli_command": "blocked_requires_executor_design",
            "unsupported_hook": hook,
            "required_options": [],
        },
        "parameter_grid": {},
        "benchmark_plan": {
            "judge": "local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "candidate_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": ["blocked until a bounded executor is designed"],
        "source_grounding": {
            "source_ids": card["source_ids"],
            "algorithm_card_id": card["algorithm_card_id"],
        },
    }


def _algorithm_candidate_id(
    *,
    card_id: str,
    family: str,
    hook: str | None = None,
) -> str:
    raw_slug = re.sub(r"[^a-zA-Z0-9]+", "_", card_id.lower()).strip("_")
    if len(raw_slug) <= 80:
        return f"candidate_{raw_slug or 'source'}_{family}"
    digest_source = "|".join([card_id, family, hook or ""])
    digest = _sha256_text(digest_source)[:8]
    return f"candidate_{_slug(card_id)}_{family}_{digest}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _sandbox_level(candidate_family: str) -> int:
    if candidate_family in {
        "lightweight_residual_corrector",
        "spectral_rollout_stabilization",
        "operator_correction_postprocess",
    }:
        return 0
    if candidate_family in {
        "official_checkpoint_family",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return 1
    return 2


def _source_type(value: Any) -> str:
    text = _nonempty_text(value, "source_type")
    if text not in VALID_SOURCE_TYPES:
        raise LiteratureAlgorithmLabError(f"unsupported source_type: {text}")
    return text


def _source_ids_from_source_artifact(payload: dict[str, Any]) -> set[Any]:
    if payload.get("state_hint") == SOURCE_DOCUMENT_STATE_HINT:
        record = payload.get("source_record")
        if isinstance(record, dict):
            return {record.get("source_id")}
        return set()
    return {
        record.get("source_id")
        for record in payload.get("source_records", [])
        if isinstance(record, dict)
    }


def _source_ids_from_algorithm_cards_report(payload: dict[str, Any]) -> list[str]:
    source_ids = []
    for source_id in payload.get("source_ids", []):
        if isinstance(source_id, str) and source_id.strip():
            source_ids.append(source_id)
    if source_ids:
        return sorted(set(source_ids))
    audit = payload.get("extraction_audit")
    if not isinstance(audit, dict):
        return []
    for key in ["skipped_blocks", "skipped_parses"]:
        value = audit.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                source_ids.append(source_id)
    return sorted(set(source_ids))


def _find_by_id(value: Any, selected_id: str, id_key: str) -> dict[str, Any]:
    for item in _object_list(value, "items"):
        if item.get(id_key) == selected_id:
            return item
    raise LiteratureAlgorithmLabError(f"{id_key} not found: {selected_id}")


def _fno_relevant_context_units(context_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _operator_relevant_context_units(
        context_units,
        keywords=OPERATOR_LITERATURE_ADAPTATION_PROFILES["fno"]["keywords"],
    )


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


def _source_candidate(
    *,
    source_id: str,
    source_type: str,
    title: str,
    url: str,
    document_url: str,
    retrieval_query: str,
    discovery_provider: str,
    license_note: str,
    external_ids: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "title": title,
        "url": url,
        "document_url": document_url,
        "retrieval_query": retrieval_query,
        "discovery_provider": discovery_provider,
        "license_note": license_note,
        "external_ids": {
            key: value for key, value in external_ids.items() if value is not None
        },
    }


def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise LiteratureAlgorithmLabError("expected at least one non-empty text value")


def _xml_text(element: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    child = element.find(path, namespace)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _dedupe_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    index = 2
    while f"{base}_{index}" in seen:
        index += 1
    return f"{base}_{index}"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        result.append(_nonempty_text(item, label))
    return result
