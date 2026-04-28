import json
from pathlib import Path


def test_audit_hermes_strategy_growth_autopilot_report_passes(tmp_path: Path) -> None:
    from pde_agent.core.hermes_strategy_growth_autopilot_audit import (
        audit_hermes_strategy_growth_autopilot_report,
    )

    skill_md = tmp_path / "installed" / "skill" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text("---\nname: skill\n---\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "state_hint": "STRATEGY_SKILL_REGISTRY_UPDATED",
                "entry_count": 1,
                "entries": [
                    {
                        "state_hint": "STRATEGY_SKILL_REGISTERED",
                        "strategy_id": "spectral_residual_m16_ridge1e-03",
                        "skill_name": "spectral-residual-window-stratified-burgers",
                        "installed_skill_md_path": str(skill_md),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    install_manifest = tmp_path / "install.json"
    install_manifest.write_text(
        json.dumps(
            {
                "state_hint": "SKILL_INSTALLED",
                "strategy_id": "spectral_residual_m16_ridge1e-03",
                "skill_name": "spectral-residual-window-stratified-burgers",
                "installed_skill_md_path": str(skill_md),
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "autopilot.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_STRATEGY_GROWTH_AUTOPILOT_COMPLETED",
                "final_state_hint": "STRATEGY_SKILL_REGISTERED",
                "human_review_required": False,
                "step_count": 12,
                "steps": [
                    {"command": command, "state_hint": "ok", "artifact_path": str(tmp_path)}
                    for command in [
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
                ],
                "registry_path": str(registry),
                "installed_skill_manifest_path": str(install_manifest),
            }
        ),
        encoding="utf-8",
    )

    audit = audit_hermes_strategy_growth_autopilot_report(
        autopilot_report_path=report_path,
        output_path=tmp_path / "audit.json",
    )

    assert audit["state_hint"] == "HERMES_STRATEGY_GROWTH_AUTOPILOT_AUDITED"
    assert audit["decision"] == "passed"
    assert audit["issue_count"] == 0
    assert Path(audit["audit_path"]).exists()


def test_audit_hermes_strategy_growth_autopilot_cli_returns_payload(
    tmp_path: Path, monkeypatch
) -> None:
    import pde_agent.commands.audit_hermes_strategy_growth_autopilot as command_module
    from pde_agent.cli import app
    from typer.testing import CliRunner

    def fake_audit(**kwargs):
        return {
            "state_hint": "HERMES_STRATEGY_GROWTH_AUTOPILOT_AUDITED",
            "audit_path": str(tmp_path / "audit.json"),
            "decision": "passed",
            "issue_count": 0,
            "checks": [],
            "summary": {"step_count": 12},
            "next_command": "review-autopilot-results",
        }

    monkeypatch.setattr(
        command_module, "audit_hermes_strategy_growth_autopilot_report", fake_audit
    )
    result = CliRunner().invoke(
        app,
        [
            "audit-hermes-strategy-growth-autopilot",
            "--autopilot-report-path",
            str(tmp_path / "report.json"),
            "--output-path",
            str(tmp_path / "audit.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "audit-hermes-strategy-growth-autopilot"
    assert payload["state_hint"] == "HERMES_STRATEGY_GROWTH_AUTOPILOT_AUDITED"
    assert payload["metrics"]["decision"] == "passed"
