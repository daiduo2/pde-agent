"""literature_algorithm_lab facade - re-exports from submodules."""

from __future__ import annotations

from __future__ import annotations
import os
from pathlib import Path
from pde_agent.core.json_output import success_payload
from pde_agent.core.hypothesis_lab_trace import HypothesisLabTraceStore
from pde_agent.core.literature_algorithm_lab import (
    ALGORITHM_CANDIDATES_STATE_HINT,
    ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
    ALGORITHM_CARDS_STATE_HINT,
    ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
    ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT,
    BOUNDED_EXECUTOR_AUDIT_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT,
    BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT,
    BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT,
    BOUNDED_EXECUTOR_SANDBOX_STATE_HINT,
    BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT,
    BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
    DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    EXECUTOR_DESIGN_REQUEST_STATE_HINT,
    EXECUTOR_TAXONOMY_STATE_HINT,
    EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT,
    FNO_CANDIDATE_ADAPTED_STATE_HINT,
    FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT,
    LITERATURE_EXTRACTION_CONTEXT_STATE_HINT,
    LITERATURE_REGISTRY_UPDATED_STATE_HINT,
    LITERATURE_REGISTRY_EVENT_STATE_HINT,
    MINERU_RESULT_DOWNLOADED_STATE_HINT,
    MINERU_TASK_POLLED_STATE_HINT,
    MINERU_TASK_SUBMITTED_STATE_HINT,
    PAPER_PARSE_STATE_HINT,
    SANDBOX_PLAN_STATE_HINT,
    SOURCE_DOCUMENT_STATE_HINT,
    SOURCE_SEARCH_STATE_HINT,
    SOURCE_RECORDS_STATE_HINT,
    UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
    adapt_fno_architecture_candidate,
    audit_bounded_executor_authoring_task,
    build_blocked_candidate_handoff,
    build_executor_taxonomy,
    build_literature_source_records,
    complete_bounded_executor_authoring_task,
    compile_algorithm_candidates,
    create_bounded_executor_design_spec,
    create_executor_design_request,
    download_mineru_parse_result,
    download_source_document,
    evaluate_algorithm_candidate_promotion,
    extract_algorithm_cards,
    ingest_mineru_parse,
    normalize_arxiv_search_results,
    normalize_github_search_results,
    normalize_openalex_search_results,
    plan_candidate_sandbox,
    poll_mineru_parse_task,
    prepare_bounded_executor_authoring_task,
    prepare_deeponet_literature_adaptation_task,
    prepare_fno_literature_adaptation_task,
    prepare_literature_extraction_context,
    prepare_unet_literature_adaptation_task,
    record_bounded_executor_sandbox_evidence,
    materialize_bounded_executor_template,
    record_literature_registry_event,
    record_agent_algorithm_cards,
    record_bounded_executor_sandbox_event,
    run_arxiv_mineru_parse_pipeline,
    run_bounded_executor_sandbox,
    run_literature_algorithm_v2_pipeline,
    submit_mineru_parse_task,
    update_literature_registry,
    validate_algorithm_cards_v2,
    validate_bounded_executor_implementation,
    validate_executable_blueprints,
)


# Re-exports from submodules
from pde_agent.commands.literature_algorithm_lab_source import (
    normalize_openalex_sources_command,
    normalize_arxiv_sources_command,
    normalize_github_sources_command,
    run_arxiv_mineru_parse_pipeline_command,
    download_source_document_command,
    build_literature_source_records_command,
    _source_search_payload,
)

from pde_agent.commands.literature_algorithm_lab_mineru import (
    submit_mineru_parse_command,
    poll_mineru_parse_command,
    download_mineru_result_command,
)

from pde_agent.commands.literature_algorithm_lab_extraction import (
    prepare_literature_extraction_context_command,
    prepare_fno_literature_adaptation_task_command,
    prepare_deeponet_literature_adaptation_task_command,
    prepare_unet_literature_adaptation_task_command,
    ingest_mineru_parse_command,
    extract_algorithm_cards_command,
    record_agent_algorithm_cards_command,
)

from pde_agent.commands.literature_algorithm_lab_validation import (
    validate_algorithm_cards_v2_command,
    validate_executable_blueprints_command,
    run_literature_algorithm_v2_pipeline_command,
)

from pde_agent.commands.literature_algorithm_lab_candidate import (
    compile_algorithm_candidates_command,
    plan_candidate_sandbox_command,
    evaluate_algorithm_candidate_promotion_command,
)

from pde_agent.commands.literature_algorithm_lab_executor import (
    build_executor_taxonomy_command,
    adapt_fno_architecture_candidate_command,
    create_executor_design_request_command,
    build_blocked_candidate_handoff_command,
    create_bounded_executor_design_spec_command,
    materialize_bounded_executor_template_command,
    validate_bounded_executor_implementation_command,
    run_bounded_executor_sandbox_command,
    record_bounded_executor_sandbox_event_command,
    record_bounded_executor_sandbox_evidence_command,
    prepare_bounded_executor_authoring_task_command,
    audit_bounded_executor_authoring_task_command,
    complete_bounded_executor_authoring_task_command,
    _completion_algorithm_candidate_id,
    _completion_trace_payload,
    _completion_source_artifact_path,
    _completion_number,
)

from pde_agent.commands.literature_algorithm_lab_registry import (
    update_literature_registry_command,
    record_literature_registry_event_command,
)
