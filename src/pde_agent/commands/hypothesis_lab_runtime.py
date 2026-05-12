"""hypothesis_lab_runtime facade - re-exports from submodules."""

from __future__ import annotations

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pde_agent.core.hypothesis_doom_loop import (
    STATE_HINT as DOOM_LOOP_STATE_HINT,
)
from pde_agent.core.availability_audit import (
    STATE_HINT as AGENT_RUNTIME_AVAILABILITY_STATE_HINT,
)
from pde_agent.core.availability_audit import audit_agent_runtime_availability
from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
from pde_agent.core.hypothesis_candidate_planner import (
    STATE_HINT as CANDIDATE_PLAN_STATE_HINT,
)
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_algorithm_designer import (
    DESIGN_STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGN_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer import (
    VALIDATION_STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGN_VALIDATION_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer import (
    adapt_hypothesis_algorithm_sandbox_item_to_blueprint,
    build_hypothesis_algorithm_design,
    plan_hypothesis_algorithm_sandbox,
    record_hypothesis_algorithm_evidence,
    validate_hypothesis_algorithm_design,
)
from pde_agent.core.hypothesis_algorithm_designer_stability import (
    STATE_HINT as HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_STATE_HINT,
)
from pde_agent.core.hypothesis_algorithm_designer_stability import (
    run_hypothesis_algorithm_designer_stability_batch,
)
from pde_agent.core.hypothesis_kpi import STATE_HINT as KPI_STATE_HINT
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
from pde_agent.core.hypothesis_lab_trace import (
    HypothesisLabTraceStore,
    SUMMARY_STATE_HINT,
    summarize_hypothesis_lab_trace,
)
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox
from pde_agent.core.json_output import success_payload
from pde_agent.core.pde_literature_extraction_skill import (
    STATE_HINT as PDE_LITERATURE_EXTRACTION_SKILL_STATE_HINT,
)
from pde_agent.core.pde_literature_extraction_skill import (
    prepare_pde_literature_extraction_skill,
)
from pde_agent.core.scientific_tool_router import (
    STATE_HINT as TOOL_CATALOG_STATE_HINT,
)
from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog
from pde_agent.core.task1_short_window_training import (
    FAMILY_SELECTION_STATE_HINT as TASK1_SHORT_WINDOW_FAMILY_SELECTION_STATE_HINT,
    STATE_HINT as TASK1_SHORT_WINDOW_TRAINING_STATE_HINT,
)
from pde_agent.core.task1_short_window_training import (
    build_fno_training_harness_contract,
    plan_fno_training_candidates,
    record_fno_training_optimization_evidence,
    run_fno_training_optimization_loop,
    run_task1_short_window_family_selection_loop,
    run_task1_short_window_training_loop,
)

TRACE_EVENT_STATE_HINT = "HYPOTHESIS_LAB_TRACE_EVENT_RECORDED"

# Re-exports from submodules
from pde_agent.commands.hypothesis_lab_runtime_fno import (
    build_fno_training_harness_contract_command,
    plan_fno_training_candidates_command,
    run_fno_training_optimization_loop_command,
    record_fno_training_optimization_evidence_command,
    run_task1_short_window_training_loop_command,
    run_task1_short_window_family_selection_loop_command,
    build_fno_training_iteration_contract_command,
)

from pde_agent.commands.hypothesis_lab_runtime_algorithm import (
    build_hypothesis_algorithm_design_command,
    validate_hypothesis_algorithm_design_command,
    record_hypothesis_algorithm_evidence_command,
    plan_hypothesis_algorithm_sandbox_command,
    adapt_hypothesis_algorithm_sandbox_item_command,
    run_hypothesis_algorithm_sandbox_command,
    run_hypothesis_algorithm_designer_stability_batch_command,
)

from pde_agent.commands.hypothesis_lab_runtime_planning import (
    plan_hypothesis_candidates_command,
    build_hypothesis_kpi_report_command,
)

from pde_agent.commands.hypothesis_lab_runtime_trace import (
    record_hypothesis_lab_trace_event_command,
    summarize_hypothesis_lab_trace_command,
)

from pde_agent.commands.hypothesis_lab_runtime_misc import (
    detect_hypothesis_doom_loop_command,
    build_scientific_tool_catalog_command,
    audit_agent_runtime_availability_command,
    prepare_pde_literature_extraction_skill_command,
    materialize_training_windows_for_hypothesis_sandbox_command,
    convert_hypothesis_algorithm_evidence_for_candidate_promotion_command,
    _parse_json_objects,
    _parse_json_object,
    _hypothesis_lab_policy_constraints,
    _dedupe_strings,
)
