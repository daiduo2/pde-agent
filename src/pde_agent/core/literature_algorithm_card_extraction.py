"""literature_algorithm_card_extraction facade - re-exports from submodules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError

# Re-export constants
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
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

# Re-exports from submodules
from pde_agent.core.literature_algorithm_card_extraction_helpers import (
    _guard_output,
    _nonempty_text,
    _read_json,
    _require_file,
    _write_json,
)

from pde_agent.core.literature_algorithm_card_extraction_ingest import (
    ingest_mineru_parse,
    _parse_quality_flags,
    _section_index,
    _source_ids_from_source_artifact,
)

from pde_agent.core.literature_algorithm_card_extraction_extract import (
    extract_algorithm_cards,
    _algorithm_card_id,
    _algorithm_card_v2_enum_or_default,
    _algorithm_card_v2_fields,
    _attach_section_to_spans,
    _evidence_spans_for_fields,
    _extract_colon_field_spans,
    _extract_colon_fields,
    _first_line_matching,
    _has_complete_algorithm_card_fields,
    _implementation_requirements_from_fields,
    _infer_algorithm_card_v2_defaults,
    _infer_algorithm_fields,
    _infer_nonstructured_algorithm_field_block,
    _infer_nonstructured_algorithm_fields,
    _is_low_signal_algorithm_section,
    _line_span_from_match,
    _looks_like_fno_architecture_section,
    _looks_like_physics_residual_section,
    _looks_like_residual_correction_section,
    _looks_like_scheduled_rollout_section,
    _looks_like_spectral_rollout_section,
    _markdown_heading_blocks,
    _missing_algorithm_card_fields,
    _non_heading_lines,
    _nonempty_or_default,
    _normalize_algorithm_hook,
    _parse_bool_field,
    _scan_algorithm_card_field_blocks,
    _slug,
)
