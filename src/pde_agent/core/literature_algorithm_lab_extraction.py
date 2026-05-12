"""Literature algorithm lab extraction functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_algorithm_card_extraction import (
    _markdown_heading_blocks,
    extract_algorithm_cards,
    ingest_mineru_parse,
)
from pde_agent.core.literature_algorithm_lab_helpers import (
    _guard_output,
    _nonempty_text,
    _object_list,
    _operator_relevant_context_units,
    _read_json,
    _require_file,
    _slug,
    _write_json,
)

PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
LITERATURE_EXTRACTION_CONTEXT_STATE_HINT = "LITERATURE_EXTRACTION_CONTEXT_PREPARED"
FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT = "FNO_LITERATURE_ADAPTATION_TASK_PREPARED"
DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT = (
    "DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED"
)
UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT = "UNET_LITERATURE_ADAPTATION_TASK_PREPARED"
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
