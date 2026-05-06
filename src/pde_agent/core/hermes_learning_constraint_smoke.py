"""Smoke test artifact for Hypothesis Lab learning constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_literature_expansion import run_candidate_expansion_handoff
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report


STATE_HINT = "HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"


class HermesLearningConstraintSmokeError(RuntimeError):
    pass


def run_hypothesis_learning_constraint_smoke(
    *,
    trace_path: str | Path,
    output_dir: str | Path,
    local_strategy_library_path: str | Path,
    checkpoint_paths: list[str | Path] | None,
    candidate_generator: str,
    min_viable_candidate_count: int,
    force: bool,
) -> dict[str, Any]:
    if min_viable_candidate_count < 1:
        raise HermesLearningConstraintSmokeError(
            "min_viable_candidate_count must be positive"
        )
    root = Path(output_dir)
    report_path = root / "hypothesis_learning_constraint_smoke.json"
    if report_path.exists() and not force:
        raise HermesLearningConstraintSmokeError(f"smoke report exists: {report_path}")
    root.mkdir(parents=True, exist_ok=True)

    kpi = build_hypothesis_kpi_report(
        trace_path=Path(trace_path),
        output_path=root / "hypothesis_kpi_report.json",
        recent_window=50,
        force=force,
    )
    plan = build_hypothesis_candidate_plan(
        output_path=root / "candidate_plan.json",
        candidate_generator=candidate_generator,
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=root / "hypothesis_kpi_report.json",
        preferred_error_modes=[],
        max_schedule_length=5,
        min_viable_candidate_count=min_viable_candidate_count,
        force=force,
    )
    handoff = run_candidate_expansion_handoff(
        candidate_plan_path=Path(plan["candidate_plan_path"]),
        workspace_dir=root / "expansion_handoff",
        local_strategy_library_path=Path(local_strategy_library_path),
        checkpoint_paths=checkpoint_paths or [],
        max_strategy_candidates=None,
        failure_memory_path=None,
        registry_summary_path=None,
        force=force,
    )
    candidates = _read_json(Path(handoff["artifacts"]["algorithm_candidates_path"]))
    families = _candidate_families(candidates)
    skipped_count = int(
        handoff.get("metrics", {}).get("learning_constraint_skipped_count", 0)
    )
    learning_constraints = plan.get("learning_constraints")
    excluded_count = _learning_constraint_excluded_count(plan)
    passed = (
        isinstance(learning_constraints, dict)
        and (
            bool(learning_constraints.get("avoid_candidate_families"))
            or bool(learning_constraints.get("avoid_algorithm_candidate_ids"))
        )
        and (skipped_count > 0 or excluded_count > 0)
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "smoke_report_path": str(report_path),
        "trace_path": str(trace_path),
        "decision": "passed" if passed else "failed",
        "artifacts": {
            "smoke_report_path": str(report_path),
            "hypothesis_kpi_report_path": str(root / "hypothesis_kpi_report.json"),
            "candidate_plan_path": plan["candidate_plan_path"],
            "expansion_handoff_report_path": handoff["handoff_report_path"],
            "algorithm_candidates_path": handoff["artifacts"][
                "algorithm_candidates_path"
            ],
        },
        "metrics": {
            "candidate_count": candidates.get("algorithm_candidate_count"),
            "learning_constraint_skipped_count": skipped_count,
            "learning_constraint_excluded_count": excluded_count,
            "deprioritized_family_count": len(
                kpi.get("deprioritized_candidate_families", [])
            ),
        },
        "learning_constraints": learning_constraints,
        "candidate_families_after_handoff": families,
        "next_command": "run-hermes-runtime-closed-loop"
        if passed
        else "inspect-hypothesis-learning-constraint-smoke",
    }
    _write_json(report_path, report)
    return report


def _learning_constraint_excluded_count(plan: dict[str, Any]) -> int:
    rationale = plan.get("selection_rationale")
    if not isinstance(rationale, list):
        return 0
    return sum(
        1
        for item in rationale
        if isinstance(item, dict)
        and item.get("decision") == "excluded"
        and item.get("reason") == "blocked_by_learning_constraint"
    )


def _candidate_families(candidates: dict[str, Any]) -> list[str]:
    values = candidates.get("algorithm_candidates")
    if not isinstance(values, list):
        return []
    return sorted(
        {
            str(candidate.get("candidate_family"))
            for candidate in values
            if isinstance(candidate, dict) and candidate.get("candidate_family")
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HermesLearningConstraintSmokeError(f"artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesLearningConstraintSmokeError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesLearningConstraintSmokeError(f"artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
