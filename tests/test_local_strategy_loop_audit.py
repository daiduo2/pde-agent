import json
from pathlib import Path


def test_audit_local_strategy_loop_runs_four_smoke_scenarios(tmp_path: Path) -> None:
    from pde_agent.core.local_strategy_loop_audit import audit_local_strategy_loop

    report = audit_local_strategy_loop(
        output_dir=tmp_path / "audit",
        report_path=tmp_path / "audit" / "audit_report.json",
        force=False,
    )

    assert report["state_hint"] == "LOCAL_STRATEGY_LOOP_AUDITED"
    assert report["decision"] == "passed"
    assert report["scenario_count"] == 4
    assert [scenario["scenario_id"] for scenario in report["scenarios"]] == [
        "positive_acceptance",
        "negative_continue",
        "failure_exhaustion",
        "competition_like_ingestion",
    ]
    assert all(scenario["passed"] for scenario in report["scenarios"])
    assert report["summary"]["accepted_count"] == 2
    assert report["summary"]["continued_count"] == 1
    assert report["summary"]["exhausted_count"] == 1
    assert Path(report["audit_report_path"]).exists()


def test_audit_local_strategy_loop_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_dir = tmp_path / "audit"
    report_path = output_dir / "audit_report.json"
    result = CliRunner().invoke(
        app,
        [
            "audit-local-strategy-loop",
            "--output-dir",
            str(output_dir),
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_STRATEGY_LOOP_AUDITED"
    assert payload["metrics"]["decision"] == "passed"
    assert payload["metrics"]["scenario_count"] == 4
    assert payload["next_hint"] == "use-local-strategy-loop"
