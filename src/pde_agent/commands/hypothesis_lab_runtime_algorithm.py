"""Command functions for algorithm domain."""

from __future__ import annotations

import json
from pathlib import Path
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
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox
from pde_agent.core.json_output import success_payload

def build_hypothesis_algorithm_design_command(
    *,
    output_path: str,
    failure_memory_paths: list[str],
    evidence_paths: list[str] | None = None,
    log_paths: list[str] | None = None,
    strategy_library_paths: list[str] | None = None,
    literature_card_paths: list[str] | None = None,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, object]:
    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[Path(path) for path in failure_memory_paths],
        evidence_paths=[Path(path) for path in evidence_paths or []],
        log_paths=[Path(path) for path in log_paths or []],
        strategy_library_paths=[Path(path) for path in strategy_library_paths or []],
        literature_card_paths=[Path(path) for path in literature_card_paths or []],
        output_path=Path(output_path),
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-hypothesis-algorithm-design",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGN_STATE_HINT,
        artifacts={
            "algorithm_design_json": output_path,
            "algorithm_design_path": output_path,
            "mechanism_hypotheses": report.get("mechanism_hypotheses", []),
            "algorithm_blueprints": report.get("algorithm_blueprints", []),
            "diversity_audit": report.get("diversity_audit", {}),
        },
        metrics={
            "hypothesis_count": len(report.get("mechanism_hypotheses", [])),
            "blueprint_count": len(report.get("algorithm_blueprints", [])),
            "mechanism_family_count": report.get("designer_rubric", {}).get(
                "mechanism_family_count",
                0,
            )
            if isinstance(report.get("designer_rubric"), dict)
            else 0,
            "decision": report.get("designer_rubric", {}).get("decision")
            if isinstance(report.get("designer_rubric"), dict)
            else None,
            "training_evidence_count": report.get("diagnosis", {}).get(
                "training_evidence_count",
                0,
            )
            if isinstance(report.get("diagnosis"), dict)
            else 0,
            "explicit_evidence_count": report.get("diagnosis", {}).get(
                "explicit_evidence_count",
                0,
            )
            if isinstance(report.get("diagnosis"), dict)
            else 0,
            "source_evidence_types": sorted(
                {
                    str(item.get("source_type"))
                    for item in report.get("diagnosis", {}).get(
                        "evidence_summaries",
                        [],
                    )
                    if isinstance(item, dict) and item.get("source_type")
                }
            )
            if isinstance(report.get("diagnosis"), dict)
            else [],
        },
        next_hint=report["next_command"],
    )

def validate_hypothesis_algorithm_design_command(
    *,
    design_path: str,
    output_path: str,
    min_hypothesis_count: int = 5,
    min_mechanism_family_count: int = 4,
    force: bool = False,
) -> dict[str, object]:
    report = validate_hypothesis_algorithm_design(
        design_path=Path(design_path),
        output_path=Path(output_path),
        min_hypothesis_count=min_hypothesis_count,
        min_mechanism_family_count=min_mechanism_family_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="validate-hypothesis-algorithm-design",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGN_VALIDATION_STATE_HINT,
        artifacts={
            "algorithm_design_audit_json": output_path,
            "algorithm_design_audit_path": output_path,
            "design_path": design_path,
            "violations": report.get("violations", []),
        },
        metrics={
            "decision": report["decision"],
            "hypothesis_count": report["hypothesis_count"],
            "mechanism_family_count": report["mechanism_family_count"],
            "blueprint_count": report["blueprint_count"],
            "violation_count": len(report.get("violations", [])),
        },
        next_hint=report["next_command"],
    )

def record_hypothesis_algorithm_evidence_command(
    *,
    design_path: str,
    sandbox_report_path: str,
    output_path: str,
    force: bool = False,
) -> dict[str, object]:
    report = record_hypothesis_algorithm_evidence(
        design_path=Path(design_path),
        sandbox_report_path=Path(sandbox_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="record-hypothesis-algorithm-evidence",
        state_hint="HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
        artifacts={
            "algorithm_evidence_json": output_path,
            "algorithm_evidence_path": output_path,
            "algorithm_card": report["algorithm_card"],
            "evidence": report["evidence"],
            "failure_taxonomy": report["failure_taxonomy"],
            "next_generation_constraints": report["next_generation_constraints"],
        },
        metrics={
            "decision": report["decision"],
            "failure_code_count": len(report["failure_taxonomy"]["failure_codes"]),
            "avoid_mechanism_family_count": len(
                report["next_generation_constraints"]["avoid_mechanism_families"]
            ),
        },
        next_hint=report["next_command"],
    )

def plan_hypothesis_algorithm_sandbox_command(
    *,
    design_path: str,
    output_path: str,
    max_ready_items: int = 5,
    force: bool = False,
) -> dict[str, object]:
    report = plan_hypothesis_algorithm_sandbox(
        design_path=Path(design_path),
        output_path=Path(output_path),
        max_ready_items=max_ready_items,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="plan-hypothesis-algorithm-sandbox",
        state_hint="HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED",
        artifacts={
            "algorithm_sandbox_plan_json": output_path,
            "algorithm_sandbox_plan_path": output_path,
            "sandbox_items": report["sandbox_items"],
            "blocked_items": report["blocked_items"],
        },
        metrics={
            "ready_count": report["ready_count"],
            "blocked_count": report["blocked_count"],
        },
        next_hint=report["next_command"],
    )

def adapt_hypothesis_algorithm_sandbox_item_command(
    *,
    sandbox_plan_path: str,
    output_path: str,
    blueprint_id: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    plan = json.loads(Path(sandbox_plan_path).read_text(encoding="utf-8"))
    items = plan.get("sandbox_items") if isinstance(plan, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("sandbox plan missing sandbox_items")
    selected = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if blueprint_id is None or item.get("blueprint_id") == blueprint_id:
            selected = item
            break
    if selected is None:
        raise RuntimeError("sandbox item not found")
    blueprint = adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
        sandbox_item=selected,
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="adapt-hypothesis-algorithm-sandbox-item",
        state_hint="HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED",
        artifacts={
            "adapted_blueprint_json": output_path,
            "adapted_blueprint_path": output_path,
            "source_algorithm_blueprint_id": blueprint["source_algorithm_blueprint_id"],
            "candidate_family": blueprint["candidate_family"],
        },
        metrics={
            "candidate_family": blueprint["candidate_family"],
            "mechanism_family": blueprint["algorithm_designer_metadata"][
                "mechanism_family"
            ],
        },
        next_hint="run-blueprint-sandbox",
    )

def run_hypothesis_algorithm_sandbox_command(
    *,
    adapted_blueprint_path: str,
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
) -> dict[str, object]:
    blueprint = json.loads(Path(adapted_blueprint_path).read_text(encoding="utf-8"))
    if blueprint.get("state_hint") != "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED":
        raise RuntimeError("adapted blueprint has invalid state_hint")
    report = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        output_dir=Path(output_dir),
        submission_id=submission_id,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    report["adapted_blueprint"] = {
        "adapted_blueprint_path": adapted_blueprint_path,
        "source_algorithm_blueprint_id": blueprint.get("source_algorithm_blueprint_id"),
        "source_hypothesis_id": blueprint.get("source_hypothesis_id"),
        "mechanism_family": (
            blueprint.get("algorithm_designer_metadata", {}).get("mechanism_family")
            if isinstance(blueprint.get("algorithm_designer_metadata"), dict)
            else None
        ),
    }
    artifacts_for_writeback = (
        report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    )
    sandbox_report_path = artifacts_for_writeback.get("sandbox_report_path")
    if isinstance(sandbox_report_path, str):
        Path(sandbox_report_path).write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    verifier = report.get("verifier") if isinstance(report.get("verifier"), dict) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    return success_payload(
        run_id="local",
        command="run-hypothesis-algorithm-sandbox",
        state_hint="HYPOTHESIS_ALGORITHM_SANDBOX_RAN",
        artifacts={
            "sandbox_report_path": artifacts.get("sandbox_report_path"),
            "judge_report_path": artifacts.get("judge_report_path"),
            "best_submission_dir": artifacts.get("best_submission_dir"),
            "adapted_blueprint_path": adapted_blueprint_path,
        },
        metrics={
            "decision": verifier.get("decision"),
            "score_delta_vs_baseline": verifier.get("score_delta_vs_baseline"),
            "best_candidate_id": report.get("best_candidate", {}).get("candidate_id")
            if isinstance(report.get("best_candidate"), dict)
            else None,
        },
        next_hint="record-hypothesis-algorithm-evidence",
    )

def run_hypothesis_algorithm_designer_stability_batch_command(
    *,
    memory_set_json: str,
    context_path: str,
    target_path: str,
    output_dir: str,
    submission_id: str,
    reference_path: str | None = None,
    reference_sample_limit: int | None = None,
    initial_submission_dir: str | None = None,
    max_sandbox_items_per_set: int = 1,
    force: bool = False,
) -> dict[str, object]:
    memory_sets = json.loads(memory_set_json)
    if not isinstance(memory_sets, list):
        raise RuntimeError("memory_set_json must be a list of path lists")
    normalized_memory_sets = []
    for row in memory_sets:
        if not isinstance(row, list):
            raise RuntimeError("memory_set_json entries must be path lists")
        normalized_memory_sets.append([Path(path) for path in row])
    report = run_hypothesis_algorithm_designer_stability_batch(
        memory_sets=normalized_memory_sets,
        context_path=Path(context_path),
        target_path=Path(target_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        reference_sample_limit=reference_sample_limit,
        initial_submission_dir=Path(initial_submission_dir)
        if initial_submission_dir is not None
        else Path(""),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        max_sandbox_items_per_set=max_sandbox_items_per_set,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-hypothesis-algorithm-designer-stability-batch",
        state_hint=HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_STATE_HINT,
        artifacts={
            "stability_report_json": report["stability_report_path"],
            "stability_report_path": report["stability_report_path"],
        },
        metrics=report["summary"],
        next_hint=report["next_command"],
    )
