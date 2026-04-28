"""Audit Hermes strategy-growth autopilot reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_STRATEGY_GROWTH_AUTOPILOT_AUDITED"
EXPECTED_COMMANDS = [
    "generate-experiment-config",
    "audit-experiment-config",
    "run-spectral-residual-window-stratification",
    "recommend-strategy-next-action",
    "evaluate-strategy-promotion",
    "append-strategy-ledger",
    "create-skill-candidate",
    "review-skill-candidate",
    "materialize-skill-draft",
    "review-skill-draft",
    "install-skill-draft",
    "register-strategy-skill",
]


class HermesStrategyGrowthAutopilotAuditError(RuntimeError):
    pass


def audit_hermes_strategy_growth_autopilot_report(
    *,
    autopilot_report_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    report_file = Path(autopilot_report_path)
    report = _read_json(report_file, "autopilot report")
    registry = _read_json(Path(report["registry_path"]), "strategy registry")
    install = _read_json(
        Path(report["installed_skill_manifest_path"]), "skill install manifest"
    )
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "autopilot_completed",
        report.get("state_hint") == "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED",
        {"state_hint": report.get("state_hint")},
    )
    _check(
        checks,
        "final_state_registered",
        report.get("final_state_hint") == "STRATEGY_SKILL_REGISTERED",
        {"final_state_hint": report.get("final_state_hint")},
    )
    _check(
        checks,
        "no_human_review_required",
        report.get("human_review_required") is False,
        {"human_review_required": report.get("human_review_required")},
    )
    commands = [
        step.get("command")
        for step in report.get("steps", [])
        if isinstance(step, dict)
    ]
    _check(
        checks,
        "expected_command_sequence",
        commands == EXPECTED_COMMANDS,
        {"expected": EXPECTED_COMMANDS, "actual": commands},
    )
    _check(
        checks,
        "registry_updated",
        registry.get("state_hint") == "STRATEGY_SKILL_REGISTRY_UPDATED"
        and registry.get("entry_count", 0) >= 1,
        {"state_hint": registry.get("state_hint"), "entry_count": registry.get("entry_count")},
    )
    installed_skill_md = install.get("installed_skill_md_path")
    _check(
        checks,
        "installed_skill_exists",
        isinstance(installed_skill_md, str) and Path(installed_skill_md).is_file(),
        {"installed_skill_md_path": installed_skill_md},
    )
    registry_entries = registry.get("entries", [])
    installed_strategy_id = install.get("strategy_id")
    installed_skill_name = install.get("skill_name")
    _check(
        checks,
        "registry_contains_installed_skill",
        any(
            isinstance(entry, dict)
            and entry.get("strategy_id") == installed_strategy_id
            and entry.get("skill_name") == installed_skill_name
            for entry in registry_entries
        ),
        {
            "strategy_id": installed_strategy_id,
            "skill_name": installed_skill_name,
        },
    )
    issues = [check for check in checks if check["status"] != "passed"]
    audit = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "autopilot_report_path": str(report_file),
        "decision": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "checks": checks,
        "summary": {
            "step_count": report.get("step_count"),
            "final_state_hint": report.get("final_state_hint"),
            "registry_path": report.get("registry_path"),
            "installed_skill_manifest_path": report.get(
                "installed_skill_manifest_path"
            ),
        },
        "next_command": "review-autopilot-results",
    }
    if output_path is not None:
        output_file = Path(output_path)
        audit["audit_path"] = str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return audit


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> None:
    checks.append(
        {
            "name": name,
            "status": "passed" if passed else "failed",
            "details": details,
        }
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HermesStrategyGrowthAutopilotAuditError(f"missing {label} path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HermesStrategyGrowthAutopilotAuditError(
            f"{label} must be a JSON object"
        )
    return data
