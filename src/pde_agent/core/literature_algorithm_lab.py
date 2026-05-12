"""literature_algorithm_lab facade - re-exports from submodules."""

from __future__ import annotations

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

# Re-exports from submodules
from pde_agent.core.literature_algorithm_lab_constants import *

from pde_agent.core.literature_algorithm_lab_helpers import (
    _find_by_id,
    _operator_relevant_context_units,
    _executor_spec,
    _default_parameter_grid,
    _slug,
    _read_json,
    _write_json,
    _guard_output,
    _require_file,
    _nonempty_text,
    _object_list,
)

from pde_agent.core.literature_algorithm_lab_extraction import (
    record_agent_algorithm_cards,
    prepare_literature_extraction_context,
    prepare_fno_literature_adaptation_task,
    prepare_deeponet_literature_adaptation_task,
    prepare_unet_literature_adaptation_task,
    _prepare_operator_literature_adaptation_task,
)

from pde_agent.core.literature_algorithm_lab_pipeline import (
    run_literature_algorithm_v2_pipeline,
)

from pde_agent.core.literature_algorithm_lab_taxonomy import (
    build_executor_taxonomy,
    adapt_fno_architecture_candidate,
    create_executor_design_request,
    build_blocked_candidate_handoff,
)
