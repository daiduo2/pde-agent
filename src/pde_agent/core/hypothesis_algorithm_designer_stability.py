"""Stability batch runner for Hypothesis Algorithm Designer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_algorithm_designer import (
    adapt_hypothesis_algorithm_sandbox_item_to_blueprint,
    build_hypothesis_algorithm_design,
    plan_hypothesis_algorithm_sandbox,
    record_hypothesis_algorithm_evidence,
    validate_hypothesis_algorithm_design,
)
from pde_agent.core.hypothesis_lab import run_blueprint_sandbox


STATE_HINT = "HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_BATCH_RAN"


class HypothesisAlgorithmDesignerStabilityError(RuntimeError):
    pass


def run_hypothesis_algorithm_designer_stability_batch(
    *,
    memory_sets: list[list[str | Path]],
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None,
    reference_sample_limit: int | None,
    initial_submission_dir: str | Path,
    output_dir: str | Path,
    submission_id: str,
    max_sandbox_items_per_set: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    if not memory_sets:
        raise HypothesisAlgorithmDesignerStabilityError("memory_sets must not be empty")
    if max_sandbox_items_per_set < 1:
        raise HypothesisAlgorithmDesignerStabilityError(
            "max_sandbox_items_per_set must be positive"
        )
    root = Path(output_dir)
    report_path = root / "stability_report.json"
    if report_path.exists() and not force:
        raise HypothesisAlgorithmDesignerStabilityError(
            f"stability report already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for index, memory_paths in enumerate(memory_sets):
        run_dir = root / f"memory_set_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        runs.append(
            _run_one_memory_set(
                index=index,
                memory_paths=[Path(path) for path in memory_paths],
                run_dir=run_dir,
                context_path=Path(context_path),
                target_path=Path(target_path),
                reference_path=Path(reference_path) if reference_path is not None else None,
                reference_sample_limit=reference_sample_limit,
                initial_submission_dir=Path(initial_submission_dir),
                submission_id=submission_id,
                max_sandbox_items=max_sandbox_items_per_set,
                force=force,
            )
        )
    summary = _summary(runs)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "summary": summary,
        "runs": runs,
        "next_command": "build-hypothesis-algorithm-design"
        if summary["sandbox_success_rate"] < 1.0
        else "record-hypothesis-lab-trace-event",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["stability_report_path"] = str(report_path)
    return report


def _run_one_memory_set(
    *,
    index: int,
    memory_paths: list[Path],
    run_dir: Path,
    context_path: Path,
    target_path: Path,
    reference_path: Path | None,
    reference_sample_limit: int | None,
    initial_submission_dir: Path,
    submission_id: str,
    max_sandbox_items: int,
    force: bool,
) -> dict[str, Any]:
    design_path = run_dir / "design.json"
    audit_path = run_dir / "design_audit.json"
    plan_path = run_dir / "sandbox_plan.json"
    run_record: dict[str, Any] = {
        "memory_set_index": index,
        "failure_memory_paths": [str(path) for path in memory_paths],
        "design_success": False,
        "validation_success": False,
        "sandbox_plan_success": False,
        "sandbox_success_count": 0,
        "evidence_writeback_success_count": 0,
        "planned_mechanism_families": [],
        "sandboxed_mechanism_families": [],
        "evidence_mechanism_families": [],
        "promoted_mechanism_families": [],
        "refinement_mechanism_families": [],
        "near_miss_mechanism_families": [],
        "rejected_mechanism_families": [],
        "quarantined_mechanism_families": [],
        "evidence_decisions": [],
        "decision_rationales": [],
        "errors": [],
    }
    try:
        design = build_hypothesis_algorithm_design(
            failure_memory_paths=memory_paths,
            log_paths=[],
            strategy_library_paths=[],
            literature_card_paths=[],
            output_path=design_path,
            force=force,
        )
        run_record["design_success"] = True
        run_record["hypothesis_count"] = len(design["mechanism_hypotheses"])
        run_record["mechanism_family_count"] = len(
            design["diversity_audit"]["mechanism_families"]
        )
    except Exception as exc:  # pragma: no cover - surfaced in report
        run_record["errors"].append(f"design_failed:{exc}")
        return _finalize_run_record(run_record)
    try:
        validation = validate_hypothesis_algorithm_design(
            design_path=design_path,
            output_path=audit_path,
            force=force,
        )
        run_record["validation_success"] = validation["decision"] == "accept"
        run_record["validation_violations"] = validation["violations"]
    except Exception as exc:  # pragma: no cover - surfaced in report
        run_record["errors"].append(f"validation_failed:{exc}")
        return _finalize_run_record(run_record)
    try:
        plan = plan_hypothesis_algorithm_sandbox(
            design_path=design_path,
            output_path=plan_path,
            max_ready_items=max_sandbox_items,
            force=force,
        )
        run_record["sandbox_plan_success"] = True
        run_record["ready_count"] = plan["ready_count"]
        run_record["blocked_count"] = plan["blocked_count"]
        run_record["planned_mechanism_families"] = _families_from_sandbox_items(
            plan.get("sandbox_items", [])
        )
    except Exception as exc:  # pragma: no cover - surfaced in report
        run_record["errors"].append(f"sandbox_plan_failed:{exc}")
        return _finalize_run_record(run_record)
    for item_index, item in enumerate(plan["sandbox_items"][:max_sandbox_items]):
        try:
            mechanism_family = str(item.get("candidate_family", "unknown"))
            adapted_path = run_dir / f"adapted_{item_index:03d}.json"
            adapted = adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
                sandbox_item=item,
                output_path=adapted_path,
                force=force,
            )
            sandbox = run_blueprint_sandbox(
                blueprint=adapted,
                context_path=context_path,
                target_path=target_path,
                reference_path=reference_path,
                reference_sample_limit=reference_sample_limit,
                output_dir=run_dir / f"sandbox_{item_index:03d}",
                submission_id=submission_id,
                timing_profile=None,
                timing_factor=None,
                force=force,
            )
            sandbox["adapted_blueprint"] = {
                "adapted_blueprint_path": str(adapted_path),
                "source_algorithm_blueprint_id": adapted.get(
                    "source_algorithm_blueprint_id"
                ),
            }
            artifacts = sandbox.get("artifacts")
            if isinstance(artifacts, dict) and isinstance(
                artifacts.get("sandbox_report_path"),
                str,
            ):
                sandbox_report_path = Path(artifacts["sandbox_report_path"])
            else:
                sandbox_report_path = (
                    run_dir
                    / f"sandbox_{item_index:03d}"
                    / "hypothesis_algorithm_sandbox_report.json"
                )
                sandbox.setdefault("artifacts", {})
                if isinstance(sandbox["artifacts"], dict):
                    sandbox["artifacts"]["sandbox_report_path"] = str(
                        sandbox_report_path
                    )
            sandbox_report_path.write_text(
                json.dumps(sandbox, allow_nan=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            run_record["sandbox_success_count"] += 1
            run_record["sandboxed_mechanism_families"].append(mechanism_family)
            evidence = record_hypothesis_algorithm_evidence(
                design_path=design_path,
                sandbox_report_path=sandbox_report_path,
                output_path=run_dir / f"evidence_{item_index:03d}.json",
                force=force,
            )
            if evidence["state_hint"] == "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED":
                run_record["evidence_writeback_success_count"] += 1
                evidence_family = _evidence_mechanism_family(evidence)
                decision = str(evidence.get("decision", "unknown"))
                run_record["evidence_mechanism_families"].append(evidence_family)
                run_record["evidence_decisions"].append(decision)
                run_record["decision_rationales"].append(
                    _decision_rationale_from_evidence(evidence)
                )
                if decision in {"promoted", "promote_to_strategy"}:
                    run_record["promoted_mechanism_families"].append(evidence_family)
                elif decision == "promote_to_refinement":
                    run_record["refinement_mechanism_families"].append(evidence_family)
                elif decision == "record_near_miss":
                    run_record["near_miss_mechanism_families"].append(evidence_family)
                elif decision == "quarantine":
                    run_record["quarantined_mechanism_families"].append(evidence_family)
                else:
                    run_record["rejected_mechanism_families"].append(evidence_family)
        except Exception as exc:  # pragma: no cover - surfaced in report
            run_record["errors"].append(f"sandbox_item_{item_index}_failed:{exc}")
    return _finalize_run_record(run_record)


def _finalize_run_record(run_record: dict[str, Any]) -> dict[str, Any]:
    run_record.setdefault("hypothesis_count", 0)
    run_record.setdefault("mechanism_family_count", 0)
    run_record.setdefault("ready_count", 0)
    run_record.setdefault("blocked_count", 0)
    for key in [
        "planned_mechanism_families",
        "sandboxed_mechanism_families",
        "evidence_mechanism_families",
        "promoted_mechanism_families",
        "refinement_mechanism_families",
        "near_miss_mechanism_families",
        "rejected_mechanism_families",
        "quarantined_mechanism_families",
        "evidence_decisions",
    ]:
        run_record[key] = _dedupe_strings(run_record.get(key, []))
    if isinstance(run_record.get("decision_rationales"), list):
        run_record["decision_rationales"] = [
            item
            for item in run_record["decision_rationales"]
            if isinstance(item, dict)
        ]
    run_record["sandbox_success"] = run_record["sandbox_success_count"] > 0
    run_record["evidence_writeback_success"] = (
        run_record["evidence_writeback_success_count"] > 0
    )
    return run_record


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(runs)
    sandboxed = _unique_families(runs, "sandboxed_mechanism_families")
    evidence = _unique_families(runs, "evidence_mechanism_families")
    promoted = _unique_families(runs, "promoted_mechanism_families")
    refinement = _unique_families(runs, "refinement_mechanism_families")
    near_miss = _unique_families(runs, "near_miss_mechanism_families")
    rejected = _unique_families(runs, "rejected_mechanism_families")
    quarantined = _unique_families(runs, "quarantined_mechanism_families")
    decisions = _unique_families(runs, "evidence_decisions")
    warnings = _stability_warnings(runs, sandboxed)
    return {
        "memory_set_count": count,
        "design_success_rate": _rate(runs, "design_success"),
        "validation_success_rate": _rate(runs, "validation_success"),
        "sandbox_plan_success_rate": _rate(runs, "sandbox_plan_success"),
        "sandbox_success_rate": _rate(runs, "sandbox_success"),
        "evidence_writeback_success_rate": _rate(
            runs,
            "evidence_writeback_success",
        ),
        "min_hypothesis_count": min(int(item["hypothesis_count"]) for item in runs),
        "min_mechanism_family_count": min(
            int(item["mechanism_family_count"]) for item in runs
        ),
        "total_error_count": sum(len(item["errors"]) for item in runs),
        "unique_sandboxed_mechanism_families": sandboxed,
        "unique_sandboxed_mechanism_family_count": len(sandboxed),
        "unique_evidence_mechanism_families": evidence,
        "unique_evidence_mechanism_family_count": len(evidence),
        "unique_promoted_mechanism_families": promoted,
        "unique_promoted_mechanism_family_count": len(promoted),
        "unique_refinement_mechanism_families": refinement,
        "unique_refinement_mechanism_family_count": len(refinement),
        "unique_near_miss_mechanism_families": near_miss,
        "unique_near_miss_mechanism_family_count": len(near_miss),
        "unique_rejected_mechanism_families": rejected,
        "unique_rejected_mechanism_family_count": len(rejected),
        "unique_quarantined_mechanism_families": quarantined,
        "unique_quarantined_mechanism_family_count": len(quarantined),
        "unique_evidence_decisions": decisions,
        "unique_evidence_decision_count": len(decisions),
        "warnings": warnings,
    }


def _rate(runs: list[dict[str, Any]], key: str) -> float:
    return sum(1 for item in runs if item.get(key)) / float(len(runs))


def _families_from_sandbox_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return _dedupe_strings(
        [
            str(item.get("candidate_family", "unknown"))
            for item in items
            if isinstance(item, dict)
        ]
    )


def _evidence_mechanism_family(evidence: dict[str, Any]) -> str:
    card = evidence.get("algorithm_card")
    if isinstance(card, dict) and isinstance(card.get("candidate_family"), str):
        return card["candidate_family"]
    taxonomy = evidence.get("failure_taxonomy")
    if isinstance(taxonomy, dict) and isinstance(taxonomy.get("mechanism_family"), str):
        return taxonomy["mechanism_family"]
    return "unknown"


def _decision_rationale_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    family = _evidence_mechanism_family(evidence)
    rubric = evidence.get("promotion_rubric")
    constraints = evidence.get("next_generation_constraints")
    return {
        "mechanism_family": family,
        "decision": str(evidence.get("decision", "unknown")),
        "reason_codes": _dedupe_strings(
            rubric.get("reason_codes", []) if isinstance(rubric, dict) else []
        ),
        "next_generation_constraints": constraints
        if isinstance(constraints, dict)
        else {},
        "next_command": str(evidence.get("next_command", "")),
    }


def _unique_families(runs: list[dict[str, Any]], key: str) -> list[str]:
    rows: list[str] = []
    for run in runs:
        rows.extend(_dedupe_strings(run.get(key, [])))
    return sorted(set(rows))


def _stability_warnings(
    runs: list[dict[str, Any]], sandboxed_families: list[str]
) -> list[str]:
    warnings: list[str] = []
    if sandboxed_families and len(sandboxed_families) == 1:
        warnings.append("repeated_first_family_only")
    if any(
        int(run.get("ready_count", 0)) > 1
        and len(_dedupe_strings(run.get("sandboxed_mechanism_families", []))) < 2
        for run in runs
    ):
        warnings.append("ready_diversity_not_executed")
    if not _unique_families(runs, "promoted_mechanism_families"):
        warnings.append("no_promoted_mechanism_family")
    return warnings


def _dedupe_strings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item)
        if value and value not in seen:
            rows.append(value)
            seen.add(value)
    return rows
