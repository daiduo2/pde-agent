"""Literature algorithm lab pipeline functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_card_validation import (
    validate_algorithm_cards_v2,
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
from pde_agent.core.literature_algorithm_lab_helpers import (
    _require_file,
    _write_json,
)

LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT = "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"

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
