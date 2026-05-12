"""Escalation pipeline: bridge training failures to literature expansion."""

import json
from pathlib import Path
from typing import Any

from pde_agent.core.fno_training_memory import (
    build_training_iteration_memory,
    merge_training_memory_with_hypothesis_feedback,
)
from pde_agent.core.local_strategy_library import build_local_strategy_library


STATE_HINT = "FNO_TRAINING_ESCALATION_PIPELINE_RAN"
NO_CANDIDATES_STATE_HINT = "FNO_TRAINING_ESCALATION_NO_NEW_CANDIDATES"
CANDIDATES_DISCOVERED_STATE_HINT = "FNO_TRAINING_ESCALATION_CANDIDATES_DISCOVERED"


class FnoTrainingEscalationError(RuntimeError):
    pass


def run_fno_training_escalation_pipeline(
    *,
    workspace_dir: str | Path,
    memory_path: str | Path,
    output_path: str | Path | None = None,
    max_strategy_candidates: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).expanduser().resolve()
    memory_file = Path(memory_path).expanduser().resolve()
    out_file = Path(output_path) if output_path is not None else workspace / "fno_training_escalation_report.json"

    memory = _load_json(memory_file)
    if memory is None:
        raise FnoTrainingEscalationError(f"could not load memory: {memory_file}")

    family_outcomes = memory.get("family_outcomes", {})
    blocked_families = [
        family
        for family, outcome in family_outcomes.items()
        if isinstance(outcome, dict) and outcome.get("status") == "blocked"
    ]
    health_feedback = _load_json(workspace / "fno_training_health_feedback.json") or {}
    health_blocked_families = _blocked_families_from_health_feedback(health_feedback)
    for family in health_blocked_families:
        if family not in blocked_families:
            blocked_families.append(family)
    failure_reasons: list[str] = []
    for outcome in family_outcomes.values():
        if isinstance(outcome, dict):
            for reason in outcome.get("failure_reasons", []):
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
    if health_blocked_families and "blocked_by_fno_training_health_feedback" not in failure_reasons:
        failure_reasons.append("blocked_by_fno_training_health_feedback")

    library_path = workspace / "local_strategy_library.json"
    if not library_path.exists():
        build_local_strategy_library(output_path=library_path, force=True)

    candidates_path = workspace / "escalation_algorithm_candidates.json"
    try:
        from pde_agent.core.literature_expansion import (
            build_local_asset_algorithm_candidates,
            run_literature_strategy_expansion,
        )
        candidates = build_local_asset_algorithm_candidates(
            local_strategy_library_path=library_path,
            checkpoint_paths=None,
            output_path=candidates_path,
            max_strategy_candidates=max_strategy_candidates,
            force=force,
        )
        candidate_count = candidates.get("algorithm_candidate_count", 0)
        expansion_error = None
        if candidate_count == 0:
            expansion = None
        else:
            try:
                expansion = run_literature_strategy_expansion(
                    algorithm_candidates_path=candidates_path,
                    workspace_dir=workspace / "fno_training_literature_expansion",
                    checkpoint_paths=None,
                    failure_memory_path=None,
                    registry_summary_path=None,
                    force=force,
                )
            except Exception as exc:
                expansion = None
                expansion_error = str(exc)
    except Exception as exc:
        raise FnoTrainingEscalationError(
            f"candidate discovery failed: {exc}"
        ) from exc

    if candidate_count == 0:
        report = {
            "schema_version": 1,
            "state_hint": NO_CANDIDATES_STATE_HINT,
            "workspace_dir": str(workspace),
            "memory_path": str(memory_file),
            "blocked_families": blocked_families,
            "failure_reasons": failure_reasons,
            "message": "No new algorithm candidates discovered from local assets.",
            "agent_instructions": [
                "Consider manual strategy exploration or parameter grid expansion.",
                "Resume training with run-fno-training-iteration and broader search.",
            ],
        }
        _write_json(out_file, report, force=force)
        return report

    hypothesis_report = {
        "state_hint": "HYPOTHESIS_LAB_STRATEGY_FEEDBACK_INTEGRATED",
        "feedback": {
            "blocked_families": blocked_families,
            "search_guidance": [
                {
                    "guidance_type": "escalation_discovered_candidates",
                    "reason": f"Escalation discovered {candidate_count} local asset candidates",
                    "algorithm_candidates_path": str(candidates_path),
                    "expansion_report_path": expansion.get("expansion_report_path") if expansion else None,
                    "sandbox_queue": expansion.get("sandbox_queue", []) if expansion else [],
                    "candidate_families": [
                        c.get("candidate_family")
                        for c in candidates.get("algorithm_candidates", [])
                        if isinstance(c, dict)
                    ],
                }
            ],
        },
    }
    merged_memory = merge_training_memory_with_hypothesis_feedback(
        memory=memory,
        hypothesis_report=hypothesis_report,
    )
    merged_memory_path = workspace / "fno_training_memory_escalated.json"
    _write_json(merged_memory_path, merged_memory, force=force)

    report = {
        "schema_version": 1,
        "state_hint": CANDIDATES_DISCOVERED_STATE_HINT,
        "workspace_dir": str(workspace),
        "memory_path": str(memory_file),
        "merged_memory_path": str(merged_memory_path),
        "algorithm_candidates_path": str(candidates_path),
        "expansion_report_path": expansion.get("expansion_report_path") if expansion else None,
        "sandbox_queue": expansion.get("sandbox_queue", []) if expansion else [],
        "strategy_growth_queue": expansion.get("strategy_growth_queue", []) if expansion else [],
        "candidate_count": candidate_count,
        "expansion_action_count": expansion.get("action_count", 0) if expansion else 0,
        "expansion_executable_count": expansion.get("executable_count", 0) if expansion else 0,
        "expansion_proposal_count": expansion.get("proposal_count", 0) if expansion else 0,
        "expansion_error": expansion_error,
        "blocked_families": blocked_families,
        "failure_reasons": failure_reasons,
        "agent_instructions": [
            "Escalation pipeline discovered new candidates.",
            "Run sandbox queue from expansion_report_path before resuming training.",
            "Use merged_memory_path for the next training iteration after sandbox evaluation.",
        ],
        "next_command": (
            "run-expansion-sandbox-queue"
            if expansion and expansion.get("sandbox_queue")
            else "run-fno-training-iteration"
        ),
    }
    _write_json(out_file, report, force=force)
    return report


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _blocked_families_from_health_feedback(feedback: dict[str, Any]) -> list[str]:
    guidance_items = feedback.get("search_guidance")
    if not isinstance(guidance_items, list):
        return []
    rows: list[str] = []
    for guidance in guidance_items:
        if not isinstance(guidance, dict):
            continue
        blocked = guidance.get("blocked_families")
        if not isinstance(blocked, list):
            continue
        for family in blocked:
            if isinstance(family, str) and family and family not in rows:
                rows.append(family)
    return rows


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FnoTrainingEscalationError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
