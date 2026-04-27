from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROPOSAL_STATE_HINT = "STRATEGY_PROPOSAL_RECORDED"
RUBRIC_STATE_HINT = "STRATEGY_RUBRIC_BUILT"
PROMOTION_STATE_HINT = "STRATEGY_PROMOTION_EVALUATED"
LEDGER_STATE_HINT = "STRATEGY_LEDGER_APPENDED"
SKILL_CANDIDATE_STATE_HINT = "SKILL_CANDIDATE_CREATED"
SKILL_REVIEW_STATE_HINT = "SKILL_CANDIDATE_REVIEWED"
SKILL_DRAFT_STATE_HINT = "SKILL_DRAFT_MATERIALIZED"
VALID_SOURCE_TYPES = {"local_library", "external_retrieval", "agent_hypothesis"}
REQUIRED_PROMOTION_METRICS = [
    "evaluation_window_count",
    "mean_score_delta",
    "score_delta_std",
    "long_window_score_delta",
    "physics_residual_delta",
    "inference_time_regression",
    "failure_rate",
]


class StrategyGrowthError(ValueError):
    """Raised when a strategy-growth artifact violates the protocol."""


def record_strategy_proposal(
    *,
    strategy_id: str,
    title: str,
    source_type: str,
    problem_context: str,
    hypothesis: str,
    mechanism: str,
    implementation_plan: dict[str, Any],
    expected_observables: list[str],
    known_risks: list[str],
    output_path: Path,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _require_nonempty("strategy_id", strategy_id)
    _require_nonempty("title", title)
    _require_nonempty("problem_context", problem_context)
    _require_nonempty("hypothesis", hypothesis)
    _require_nonempty("mechanism", mechanism)
    if source_type not in VALID_SOURCE_TYPES:
        raise StrategyGrowthError(f"unsupported source_type: {source_type}")
    if not isinstance(implementation_plan, dict) or not implementation_plan:
        raise StrategyGrowthError("implementation_plan must be a non-empty object")
    _require_nonempty_list("expected_observables", expected_observables)
    _require_nonempty_list("known_risks", known_risks)
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": PROPOSAL_STATE_HINT,
        "strategy_id": strategy_id,
        "title": title,
        "source_type": source_type,
        "problem_context": problem_context,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "implementation_plan": implementation_plan,
        "expected_observables": expected_observables,
        "known_risks": known_risks,
        "strategy_proposal_path": str(output_file),
        "next_command": "build-strategy-rubric",
    }
    _write_json(output_file, report)
    return report


def build_strategy_rubric(
    *,
    proposal_path: Path,
    output_path: Path,
    min_evaluation_windows: int,
    max_inference_time_regression: float,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    output_file = Path(output_path)
    proposal = _read_json(proposal_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if min_evaluation_windows < 2:
        raise StrategyGrowthError("min_evaluation_windows must be at least 2")
    if max_inference_time_regression < 0:
        raise StrategyGrowthError("max_inference_time_regression must be non-negative")
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    rubric: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": RUBRIC_STATE_HINT,
        "strategy_id": proposal["strategy_id"],
        "strategy_proposal_path": str(proposal_file),
        "experiment_rubric_path": str(output_file),
        "required_metrics": REQUIRED_PROMOTION_METRICS,
        "promotion_thresholds": {
            "min_evaluation_windows": min_evaluation_windows,
            "min_mean_score_delta": 0.0,
            "min_long_window_score_delta": 0.0,
            "max_score_delta_std": 2.0,
            "max_physics_residual_delta": 0.0,
            "max_inference_time_regression": max_inference_time_regression,
            "max_failure_rate": 0.0,
        },
        "next_command": "evaluate-strategy-promotion",
    }
    _write_json(output_file, rubric)
    return rubric


def evaluate_strategy_promotion(
    *,
    proposal_path: Path,
    rubric_path: Path,
    evidence_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    rubric_file = Path(rubric_path)
    evidence_file = Path(evidence_summary_path)
    output_file = Path(output_path)
    proposal = _read_json(proposal_file)
    rubric = _read_json(rubric_file)
    evidence = _read_json(evidence_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if rubric.get("state_hint") != RUBRIC_STATE_HINT:
        raise StrategyGrowthError("rubric artifact has invalid state_hint")
    if proposal.get("strategy_id") != rubric.get("strategy_id"):
        raise StrategyGrowthError("proposal and rubric strategy_id mismatch")
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    thresholds = rubric["promotion_thresholds"]
    reason_codes: list[str] = []
    missing = [
        metric for metric in rubric["required_metrics"] if metric not in evidence
    ]
    for metric in missing:
        reason_codes.append(f"missing_metric:{metric}")

    if not missing:
        if evidence["evaluation_window_count"] < thresholds["min_evaluation_windows"]:
            reason_codes.append("insufficient_evaluation_windows")
        if evidence["mean_score_delta"] < thresholds["min_mean_score_delta"]:
            reason_codes.append("negative_mean_score_delta")
        if evidence["long_window_score_delta"] < thresholds["min_long_window_score_delta"]:
            reason_codes.append("negative_long_window_score_delta")
        if evidence["failure_rate"] > thresholds["max_failure_rate"]:
            reason_codes.append("nonzero_failure_rate")
        if (
            evidence["inference_time_regression"]
            > thresholds["max_inference_time_regression"]
        ):
            reason_codes.append("inference_time_regression_too_high")
        if evidence["physics_residual_delta"] > thresholds["max_physics_residual_delta"]:
            reason_codes.append("physics_residual_regressed")
        if evidence["score_delta_std"] > thresholds["max_score_delta_std"]:
            reason_codes.append("score_delta_variance_too_high")

    hard_reject_reasons = {
        code
        for code in reason_codes
        if code.startswith("missing_metric:")
        or code
        in {
            "insufficient_evaluation_windows",
            "negative_mean_score_delta",
            "negative_long_window_score_delta",
            "nonzero_failure_rate",
            "inference_time_regression_too_high",
        }
    }
    if hard_reject_reasons:
        decision = "reject"
        promote = False
    elif reason_codes:
        decision = "revise"
        promote = False
    else:
        decision = "promote"
        promote = True
        reason_codes = ["all_thresholds_passed"]

    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": PROMOTION_STATE_HINT,
        "strategy_id": proposal["strategy_id"],
        "strategy_proposal_path": str(proposal_file),
        "experiment_rubric_path": str(rubric_file),
        "evidence_summary_path": str(evidence_file),
        "promotion_report_path": str(output_file),
        "decision": decision,
        "promote_to_skill_candidate": promote,
        "reason_codes": reason_codes,
        "metric_results": {
            metric: evidence.get(metric) for metric in rubric["required_metrics"]
        },
        "next_command": "append-strategy-ledger",
    }
    _write_json(output_file, report)
    return report


def append_strategy_ledger(
    *,
    proposal_path: Path,
    rubric_path: Path,
    evidence_summary_path: Path,
    promotion_report_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    rubric_file = Path(rubric_path)
    evidence_file = Path(evidence_summary_path)
    promotion_file = Path(promotion_report_path)
    ledger_file = Path(ledger_path)
    proposal = _read_json(proposal_file)
    rubric = _read_json(rubric_file)
    evidence = _read_json(evidence_file)
    promotion = _read_json(promotion_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if rubric.get("state_hint") != RUBRIC_STATE_HINT:
        raise StrategyGrowthError("rubric artifact has invalid state_hint")
    if promotion.get("state_hint") != PROMOTION_STATE_HINT:
        raise StrategyGrowthError("promotion artifact has invalid state_hint")
    strategy_id = proposal.get("strategy_id")
    if strategy_id != rubric.get("strategy_id") or strategy_id != promotion.get(
        "strategy_id"
    ):
        raise StrategyGrowthError("strategy_id mismatch across growth artifacts")
    promote = bool(promotion["promote_to_skill_candidate"])
    row: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": LEDGER_STATE_HINT,
        "strategy_id": strategy_id,
        "strategy_proposal_path": str(proposal_file),
        "experiment_rubric_path": str(rubric_file),
        "evidence_summary_path": str(evidence_file),
        "promotion_report_path": str(promotion_file),
        "strategy_ledger_path": str(ledger_file),
        "decision": promotion["decision"],
        "promote_to_skill_candidate": promote,
        "reason_codes": promotion["reason_codes"],
        "evidence_metric_count": len(evidence),
        "next_command": "create-skill-candidate"
        if promote
        else "record-strategy-proposal",
    }
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    with ledger_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def create_skill_candidate(
    *,
    proposal_path: Path,
    rubric_path: Path,
    evidence_summary_path: Path,
    promotion_report_path: Path,
    output_path: Path,
    skill_name: str,
    skill_dir: str,
) -> dict[str, Any]:
    proposal_file = Path(proposal_path)
    rubric_file = Path(rubric_path)
    evidence_file = Path(evidence_summary_path)
    promotion_file = Path(promotion_report_path)
    output_file = Path(output_path)
    proposal = _read_json(proposal_file)
    rubric = _read_json(rubric_file)
    evidence = _read_json(evidence_file)
    promotion = _read_json(promotion_file)
    if proposal.get("state_hint") != PROPOSAL_STATE_HINT:
        raise StrategyGrowthError("proposal artifact has invalid state_hint")
    if rubric.get("state_hint") != RUBRIC_STATE_HINT:
        raise StrategyGrowthError("rubric artifact has invalid state_hint")
    if promotion.get("state_hint") != PROMOTION_STATE_HINT:
        raise StrategyGrowthError("promotion artifact has invalid state_hint")
    strategy_id = proposal.get("strategy_id")
    if strategy_id != rubric.get("strategy_id") or strategy_id != promotion.get(
        "strategy_id"
    ):
        raise StrategyGrowthError("strategy_id mismatch across growth artifacts")
    if not promotion.get("promote_to_skill_candidate"):
        raise StrategyGrowthError("promotion report did not promote this strategy")
    _require_nonempty("skill_name", skill_name)
    _require_nonempty("skill_dir", skill_dir)
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    candidate: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": SKILL_CANDIDATE_STATE_HINT,
        "strategy_id": strategy_id,
        "skill_name": skill_name,
        "suggested_skill_dir": skill_dir,
        "skill_candidate_path": str(output_file),
        "title": proposal["title"],
        "source_type": proposal["source_type"],
        "scientific_basis": {
            "problem_context": proposal["problem_context"],
            "hypothesis": proposal["hypothesis"],
            "mechanism": proposal["mechanism"],
        },
        "trigger_conditions": proposal["expected_observables"],
        "procedure_steps": _procedure_steps(proposal),
        "failure_boundaries": proposal["known_risks"],
        "promotion_reason_codes": promotion["reason_codes"],
        "evidence_sources": {
            "strategy_proposal_path": str(proposal_file),
            "experiment_rubric_path": str(rubric_file),
            "evidence_summary_path": str(evidence_file),
            "promotion_report_path": str(promotion_file),
        },
        "metric_snapshot": {
            metric: evidence.get(metric) for metric in rubric["required_metrics"]
        },
        "next_command": "review-skill-candidate",
    }
    _write_json(output_file, candidate)
    return candidate


def review_skill_candidate(
    *,
    skill_candidate_path: Path,
    output_path: Path,
    reviewer_id: str,
) -> dict[str, Any]:
    candidate_file = Path(skill_candidate_path)
    output_file = Path(output_path)
    candidate = _read_json(candidate_file)
    if candidate.get("state_hint") != SKILL_CANDIDATE_STATE_HINT:
        raise StrategyGrowthError("skill candidate artifact has invalid state_hint")
    _require_nonempty("reviewer_id", reviewer_id)
    if output_file.exists():
        raise StrategyGrowthError(f"output file already exists: {output_file}")

    checklist = {
        "has_evidence_sources": _nonempty_dict(candidate.get("evidence_sources")),
        "has_failure_boundaries": _nonempty_list(candidate.get("failure_boundaries")),
        "has_procedure_steps": _nonempty_list(candidate.get("procedure_steps")),
        "has_trigger_conditions": _nonempty_list(candidate.get("trigger_conditions")),
        "has_scientific_basis": _nonempty_dict(candidate.get("scientific_basis")),
    }
    reason_codes = [
        reason
        for passed, reason in (
            (checklist["has_evidence_sources"], "missing_evidence_sources"),
            (checklist["has_failure_boundaries"], "missing_failure_boundaries"),
            (checklist["has_procedure_steps"], "missing_procedure_steps"),
            (checklist["has_trigger_conditions"], "missing_trigger_conditions"),
            (checklist["has_scientific_basis"], "missing_scientific_basis"),
        )
        if not passed
    ]
    approved = not reason_codes
    if approved:
        reason_codes = ["candidate_review_passed"]

    review: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": SKILL_REVIEW_STATE_HINT,
        "strategy_id": candidate["strategy_id"],
        "skill_name": candidate["skill_name"],
        "skill_candidate_path": str(candidate_file),
        "skill_review_path": str(output_file),
        "reviewer_id": reviewer_id,
        "decision": "approve" if approved else "reject",
        "approved_for_draft": approved,
        "reason_codes": reason_codes,
        "review_checklist": checklist,
        "next_command": "materialize-skill-draft"
        if approved
        else "revise-skill-candidate",
    }
    _write_json(output_file, review)
    return review


def materialize_skill_draft(
    *,
    skill_candidate_path: Path,
    skill_review_path: Path,
    draft_root: Path,
) -> dict[str, Any]:
    candidate_file = Path(skill_candidate_path)
    review_file = Path(skill_review_path)
    root = Path(draft_root)
    candidate = _read_json(candidate_file)
    review = _read_json(review_file)
    if candidate.get("state_hint") != SKILL_CANDIDATE_STATE_HINT:
        raise StrategyGrowthError("skill candidate artifact has invalid state_hint")
    if review.get("state_hint") != SKILL_REVIEW_STATE_HINT:
        raise StrategyGrowthError("skill review artifact has invalid state_hint")
    if candidate.get("strategy_id") != review.get("strategy_id"):
        raise StrategyGrowthError("strategy_id mismatch across skill artifacts")
    if not review.get("approved_for_draft"):
        raise StrategyGrowthError("skill review is not approved for draft materialization")

    skill_dir = root / candidate["skill_name"]
    skill_md = skill_dir / "SKILL.md"
    manifest_path = skill_dir / "skill_draft_manifest.json"
    if skill_md.exists() or manifest_path.exists():
        raise StrategyGrowthError(f"skill draft already exists: {skill_dir}")

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(_skill_markdown(candidate), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": SKILL_DRAFT_STATE_HINT,
        "strategy_id": candidate["strategy_id"],
        "skill_name": candidate["skill_name"],
        "skill_draft_dir": str(skill_dir),
        "skill_md_path": str(skill_md),
        "skill_draft_manifest_path": str(manifest_path),
        "skill_candidate_path": str(candidate_file),
        "skill_review_path": str(review_file),
        "review_decision": review["decision"],
        "next_command": "human-review-skill-draft",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StrategyGrowthError(f"{name} must be a non-empty string")


def _require_nonempty_list(name: str, value: list[str]) -> None:
    if not isinstance(value, list) or not value:
        raise StrategyGrowthError(f"{name} must be a non-empty list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise StrategyGrowthError(f"{name} must contain only non-empty strings")


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _nonempty_dict(value: object) -> bool:
    return isinstance(value, dict) and bool(value)


def _procedure_steps(proposal: dict[str, Any]) -> list[str]:
    plan = proposal["implementation_plan"]
    commands = plan.get("allowed_commands", [])
    artifacts = plan.get("expected_artifacts", [])
    steps = [proposal["mechanism"]]
    if isinstance(commands, list) and commands:
        steps.append("Run only the approved commands: " + ", ".join(map(str, commands)))
    if isinstance(artifacts, list) and artifacts:
        steps.append("Verify expected artifacts: " + ", ".join(map(str, artifacts)))
    return steps


def _skill_markdown(candidate: dict[str, Any]) -> str:
    basis = candidate["scientific_basis"]
    lines = [
        "---",
        f"name: {candidate['skill_name']}",
        "description: "
        + candidate["title"].replace("\n", " ").strip(),
        "---",
        "",
        "# Purpose",
        "",
        basis["hypothesis"],
        "",
        "# When To Use",
        "",
    ]
    lines.extend(f"- {item}" for item in candidate["trigger_conditions"])
    lines.extend(["", "# Procedure", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(candidate["procedure_steps"], start=1))
    lines.extend(["", "# Failure Boundaries", ""])
    lines.extend(f"- {item}" for item in candidate["failure_boundaries"])
    lines.extend(["", "# Evidence Sources", ""])
    for name, path in candidate["evidence_sources"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StrategyGrowthError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StrategyGrowthError(f"JSON artifact must be an object: {path}")
    return payload
