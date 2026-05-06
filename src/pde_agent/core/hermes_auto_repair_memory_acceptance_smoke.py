"""End-to-end acceptance smoke for Hermes auto-repair memory internalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hermes_runtime_audit_chain import run_hermes_runtime_audit_chain
from pde_agent.core.hermes_runtime_closed_loop_audit import (
    audit_hermes_runtime_closed_loop,
)
from pde_agent.core.hermes_repair_policy_summary import build_repair_policy_summary
from pde_agent.core.hypothesis_candidate_planner import build_hypothesis_candidate_plan
from pde_agent.core.hypothesis_kpi import build_hypothesis_kpi_report
from pde_agent.core.research_chronicle import build_research_chronicle


STATE_HINT = "HERMES_AUTO_REPAIR_MEMORY_ACCEPTANCE_SMOKE_RAN"
MEMORY_GATE = "runtime_memory_bridge_smoke_gate"


class HermesAutoRepairMemoryAcceptanceSmokeError(RuntimeError):
    pass


def run_hermes_auto_repair_memory_acceptance_smoke(
    *,
    output_dir: str | Path,
    trace_path: str | Path,
    initial_submission_dir: str | Path,
    task1_test_path: str | Path,
    task1_target_path: str | Path,
    candidate_generator: str = "postprocess",
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    smoke_path = root / "hermes_auto_repair_memory_acceptance_smoke.json"
    if smoke_path.exists() and not force:
        raise HermesAutoRepairMemoryAcceptanceSmokeError(
            f"acceptance smoke exists: {smoke_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    _seed_failed_memory_smoke(root)

    chain = run_hermes_runtime_audit_chain(
        output_dir=root,
        trace_path=Path(trace_path),
        initial_submission_dir=Path(initial_submission_dir),
        candidate_generator=candidate_generator,
        candidate_generator_fallback="auto",
        iterations=1,
        max_batches=1,
        recent_window=50,
        repeated_family_threshold=3,
        failed_family_threshold=2,
        min_score_delta=0.1,
        max_rounds_per_batch=1,
        batch_deadline_seconds=30.0,
        submit_score_threshold=130.0,
        learning_objective="score",
        task1_test_path=Path(task1_test_path),
        task1_target_path=Path(task1_target_path),
        task1_reference_path=None,
        task2_test_path=None,
        task2_target_path=None,
        timing_profile=None,
        timing_factor=None,
        auto_repair=True,
        force=force,
    )
    kpi_path = root / "kpi_after_auto_repair.json"
    repair_policy_path = root / "repair_policy_after_auto_repair.json"
    kpi = build_hypothesis_kpi_report(
        trace_path=Path(trace_path),
        output_path=kpi_path,
        recent_window=50,
        force=True,
    )
    repair_policy = build_repair_policy_summary(
        trace_path=Path(trace_path),
        output_path=repair_policy_path,
        recent_window=50,
        force=True,
    )
    chronicle = build_research_chronicle(
        artifact_paths=[Path(chain["chain_report_path"])],
        output_path=root / "research_chronicle_after_auto_repair.md",
        force=True,
    )
    chronicle_path = Path(str(chronicle["artifacts"]["chronicle_path"]))
    candidate_plan = build_hypothesis_candidate_plan(
        output_path=root / "candidate_plan_after_acceptance.json",
        candidate_generator=candidate_generator,
        doom_loop_report_path=None,
        hypothesis_kpi_report_path=kpi_path,
        repair_policy_summary_path=repair_policy_path,
        research_chronicle_path=chronicle_path,
        expansion_handoff_report_path=None,
        audit_repair_tasks_path=None,
        preferred_error_modes=[],
        max_schedule_length=3,
        min_viable_candidate_count=1,
        force=True,
    )
    runtime_report_path = _auto_repair_runtime_report_path(chain)
    strict_audit = audit_hermes_runtime_closed_loop(
        runtime_report_path=runtime_report_path,
        output_path=root / "strict_runtime_memory_bridge_audit.json",
        require_guard_effect=False,
        require_learning_constraint_smoke=False,
        require_runtime_memory_bridge_smoke=True,
        force=True,
    )
    checks = _checks(
        chain=chain,
        kpi=kpi,
        repair_policy=repair_policy,
        chronicle_path=chronicle_path,
        candidate_plan=candidate_plan,
        strict_audit=strict_audit,
    )
    passed = all(checks.values())
    artifacts = {
        "smoke_report_path": str(smoke_path),
        "chain_report_path": str(chain["chain_report_path"]),
        "hypothesis_kpi_report_path": str(kpi_path),
        "repair_policy_summary_path": str(repair_policy_path),
        "research_chronicle_path": str(chronicle_path),
        "candidate_plan_path": str(candidate_plan["candidate_plan_path"]),
        "strict_audit_report_path": str(strict_audit["audit_report_path"]),
        "runtime_loop_report_path": str(runtime_report_path),
    }
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": "passed" if passed else "failed",
        "checks": checks,
        "artifacts": artifacts,
        "metrics": {
            "passed_check_count": sum(1 for value in checks.values() if value),
            "check_count": len(checks),
            "auto_repair_memory_bridge_success_count": _nested(
                kpi,
                ["metrics", "auto_repair_memory_bridge_success_count"],
                0,
            ),
            "repair_policy_recommended_action_count": len(
                _list_dicts(repair_policy.get("recommended_actions"))
            ),
        },
        "next_command": "run-hermes-runtime-audit-chain"
        if passed
        else "inspect-hermes-auto-repair-memory-acceptance-smoke",
    }
    smoke_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _seed_failed_memory_smoke(root: Path) -> None:
    smoke_dir = root / "runtime_memory_bridge_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = smoke_dir / "hermes_runtime_memory_bridge_smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_RUNTIME_MEMORY_BRIDGE_SMOKE_RAN",
                "decision": "failed",
                "artifacts": {},
                "memory_bridge": {"passed": False},
                "next_command": "inspect-hermes-runtime-memory-bridge-smoke",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _checks(
    *,
    chain: dict[str, Any],
    kpi: dict[str, Any],
    repair_policy: dict[str, Any],
    chronicle_path: Path,
    candidate_plan: dict[str, Any],
    strict_audit: dict[str, Any],
) -> dict[str, bool]:
    actions = _list_dicts(repair_policy.get("recommended_actions"))
    planner_actions = _list_dicts(
        _nested(
            candidate_plan,
            ["learning_constraints", "repair_policy_recommended_actions"],
            [],
        )
    )
    chronicle_text = (
        chronicle_path.read_text(encoding="utf-8") if chronicle_path.exists() else ""
    )
    return {
        "auto_repair_passed": chain.get("decision") == "repaired"
        and _nested(chain, ["metrics", "auto_repair_decision"], "") == "passed",
        "memory_bridge_passed": _nested(
            chain,
            ["metrics", "auto_repair_memory_bridge_passed"],
            False,
        )
        is True,
        "kpi_internalized": _nested(
            kpi,
            ["metrics", "auto_repair_memory_bridge_success_count"],
            0,
        )
        > 0,
        "repair_policy_internalized": MEMORY_GATE
        in _string_list(repair_policy.get("converged_repair_errors"))
        and any(action.get("audit_error") == MEMORY_GATE for action in actions),
        "chronicle_internalized": MEMORY_GATE in chronicle_text
        and "repair_action source=auto_repair" in chronicle_text,
        "planner_internalized": MEMORY_GATE
        in _string_list(
            _nested(
                candidate_plan,
                ["learning_constraints", "repair_convergence_errors"],
                [],
            )
        )
        and any(action.get("audit_error") == MEMORY_GATE for action in planner_actions),
        "strict_audit_passed": strict_audit.get("decision") == "passed"
        and _nested(strict_audit, ["summary", "failed_check_count"], 1) == 0,
    }


def _auto_repair_runtime_report_path(chain: dict[str, Any]) -> Path:
    path = _nested(
        chain,
        ["auto_repair", "artifacts", "runtime_loop_report_path"],
        None,
    )
    if not isinstance(path, str) or not path:
        raise HermesAutoRepairMemoryAcceptanceSmokeError(
            "chain missing auto_repair runtime_loop_report_path"
        )
    return Path(path)


def _nested(payload: dict[str, Any], keys: list[str], default: Any) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
