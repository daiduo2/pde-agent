"""task1_short_window_training facade - re-exports from submodules."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import h5py
import numpy as np
from pde_agent.core.local_windows import build_local_windows
from pde_agent.core.task1_fno_training_policy import (
    _any_guardrail_evaluated,
    _anchor_family_challenge_candidates,
    _anchor_neighborhood_candidates,
    _anchor_only_int_values,
    _anchor_train_step_values,
    _attempted_fno_config_signatures,
    _candidate_rationale,
    _cap_positive_int_values,
    _decision_reasons,
    _dedupe,
    _diversified_model_family_values,
    _fno_candidate_record,
    _fno_config_signature,
    _fno_directive_action,
    _fno_directive_reason_codes,
    _fno_graft_evidence_summary,
    _fno_model_family_values,
    _fno_optimization_constraints,
    _fno_planning_preferences,
    _fno_regression_repair_candidates,
    _first_available_loss,
    _first_available_string,
    _fno_selection_components,
    _fno_selection_reason_codes,
    _fno_selection_score,
    _fno_training_directive,
    _guardrail_quality,
    _has_expanded_capacity_axis,
    _has_positive_late_safe_graft_evidence,
    _loss_type_list,
    _medium_training_budget_values,
    _merge_preferred_ints,
    _merge_preferred_strings,
    _normalization_list,
    _normalize_fno_candidate_config,
    _operator_policy_from_search_guidance,
    _ordered_values,
    _prefers_medium_training_budget,
    _prioritize_new_dynamics_families,
    _preferred_probe_forecast,
    _select_fno_candidate,
    _short_window_metrics,
    _string_list,
    _training_constraints,
    _training_decision,
)
from pde_agent.core.task1_short_window_common import (
    FAMILY_SELECTION_STATE_HINT,
    STATE_HINT,
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
from pde_agent.core.task1_short_window_family_policy import (
    family_record as _family_record,
    family_selection_constraints as _family_selection_constraints,
    family_training_directive as _family_training_directive,
    select_family as _select_family,
)
from pde_agent.core.task1_short_window_hypothesis_bridge import (
    convert_hypothesis_algorithm_evidence_for_candidate_promotion,
    materialize_training_windows_for_hypothesis_sandbox,
)
from pde_agent.core.task1_short_window_models import (
    build_model as _build_model,
    denormalize_prediction as _denormalize_prediction,
    normalize_windows as _normalize_windows,
    parameter_count as _parameter_count,
    training_loss as _training_loss,
)

# Re-exports from submodules
from pde_agent.core.task1_short_window_training_helpers import (
    _hdf5_dataset_sample_count,
    _repair_training_sample_budget,
    _write_hdf5_tensor,
    _slug,
    _float_or_default_local,
    _build_fno_training_iteration_prompt,
    _load_windows,
    _source_horizon_guard,
    _guard_artifacts,
    _positive_int,
    _positive_float,
    _positive_int_list,
    _load_json,
    _guard_output,
    _write_optional_json,
    _join_command,
    _require_torch,
)

from pde_agent.core.task1_short_window_training_loops import (
    run_task1_short_window_training_loop,
    run_task1_short_window_family_selection_loop,
)

from pde_agent.core.task1_short_window_training_planning import (
    plan_fno_training_candidates,
    run_fno_training_optimization_loop,
    record_fno_training_optimization_evidence,
)

from pde_agent.core.task1_short_window_training_contracts import (
    build_fno_training_harness_contract,
    build_fno_training_iteration_contract_from_evidence,
    _fno_graft_harness_actions,
    _resolve_fno_graft_inputs,
)
