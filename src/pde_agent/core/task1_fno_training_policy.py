"""task1_fno_training_policy facade - re-exports from submodules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from pde_agent.core.task1_short_window_common import (
    SUPPORTED_FNO_LOSS_TYPES,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_NORMALIZATION_TYPES,
    Task1ShortWindowTrainingError,
)
from pde_agent.core.task1_short_window_family_policy import family_record as _family_record

# Re-exports from submodules
from pde_agent.core.task1_fno_training_policy_core import (
    _fno_candidate_record,
    _select_fno_candidate,
    _fno_selection_reason_codes,
    _fno_selection_components,
    _fno_selection_score,
    _fno_early_middle_selection_score,
    _guardrail_quality,
    _fno_optimization_constraints,
    _graft_plateau_requires_new_dynamics_source,
    _has_positive_late_safe_graft_evidence,
    _fno_graft_evidence_summary,
    _graft_refine_horizons,
    _graft_training_objectives,
    _any_guardrail_evaluated,
    _fno_planning_preferences,
    _operator_policy_from_search_guidance,
    _best_late_safe_graft_evidence,
    _prioritize_graft_anchor_preferences,
    _move_preference_to_front,
    _has_capacity_expansion_regression,
    _same_fno_capacity,
    _fno_regression_aware_preferences,
    _anchor_train_step_values,
    _medium_training_budget_values,
    _cap_positive_int_values,
    _float_or_default_local,
    _load_json,
    _positive_int,
    _positive_int_list_optional,
    _fno_model_family_list_optional,
    _training_memory_anchor_list_optional,
    _fno_model_family_values,
    _loss_type_list,
    _normalization_list_optional,
    _string_list,
    _dedupe,
)

from pde_agent.core.task1_fno_training_policy_directive import (
    _fno_training_directive,
    _fno_directive_action,
    _fno_directive_reason_codes,
    _architecture_search_exhausted,
)

from pde_agent.core.task1_fno_training_policy_metrics import (
    _short_window_metrics,
    _rel_mse,
)

from pde_agent.core.task1_fno_training_policy_utils import (
    _attempted_fno_config_signatures,
    _fno_config_signature,
    _preferred_probe_forecast,
    _fno_regression_repair_candidates,
    _anchor_neighborhood_candidates,
    _anchor_family_challenge_candidates,
    _anchor_only_int_values,
    _prefers_medium_training_budget,
    _has_expanded_capacity_axis,
    _first_available_loss,
    _first_available_string,
    _candidate_rationale,
    _ordered_values,
    _merge_preferred_ints,
    _merge_preferred_strings,
    _training_decision,
    _training_constraints,
    _decision_reasons,
    _diversified_model_family_values,
    _prioritize_new_dynamics_families,
    _normalize_fno_candidate_config,
    _normalization_list,
)
