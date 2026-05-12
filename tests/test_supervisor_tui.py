import json
from pathlib import Path


def test_build_supervisor_runs_view_lists_recent_runs_with_state_and_metrics(
    tmp_path: Path,
) -> None:
    from pde_agent.core.supervisor_tui import build_supervisor_runs_view

    run_dir = tmp_path / "runs" / "fno_run"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "fno_selection_score": 75.75,
                    "config": {
                        "forecast_steps": 95,
                        "loss_type": "early_weighted_mse",
                    },
                    "early_rel_mse": 0.6294,
                    "early_score": 0.00034,
                    "middle_guardrail": {
                        "evaluated": True,
                        "rel_mse": 1.9201,
                        "score": 0.00000045,
                    },
                    "late_guardrail": {"evaluated": False},
                    "selection_reason_codes": [
                        "early_middle_objective_selected",
                        "middle_guardrail_available",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "claude_sdk_session.json").write_text(
        json.dumps(
            {
                "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
                "metrics": {"sdk_message_count": 94},
            }
        ),
        encoding="utf-8",
    )

    view = build_supervisor_runs_view(runs_root=tmp_path / "runs", limit=5, now=run_dir.stat().st_mtime + 125)

    assert view["state_hint"] == "SUPERVISOR_TUI_RUNS_VIEW_BUILT"
    assert view["read_only"] is True
    assert view["run_count"] == 1
    row = view["runs"][0]
    assert row["run_id"] == "fno_run"
    assert row["age"] == "2m"
    assert row["age_seconds"] == 125
    assert row["pipeline_state"] == "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED"
    assert row["agent_state"] == "CLAUDE_AGENT_SDK_RUNTIME_RAN"
    assert row["selected_candidate_id"] == "fno_train_candidate_002"
    assert row["early_rel_mse"] == 0.6294
    assert row["middle_rel_mse"] == 1.9201
    assert row["late_evaluated"] is False
    assert "early_middle_objective_selected" in row["selection_reasons"]
    assert "fno_run" in view["text"]
    assert "age" in view["text"]
    assert "early_rel" in view["text"]


def test_supervisor_runs_view_wide_mode_preserves_long_run_names(tmp_path: Path) -> None:
    from pde_agent.core.supervisor_tui import build_supervisor_runs_view

    long_name = "fno_early_middle_agent_sprint_20260506_iter001"
    run_dir = tmp_path / "runs" / long_name
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "middle_guardrail": {"evaluated": False},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )

    narrow = build_supervisor_runs_view(runs_root=tmp_path / "runs", limit=1)
    wide = build_supervisor_runs_view(runs_root=tmp_path / "runs", limit=1, wide=True)

    assert long_name not in narrow["text"]
    assert "…" in narrow["text"]
    assert long_name in wide["text"]
    assert wide["wide"] is True


def test_build_supervisor_run_view_reports_pipeline_artifacts_and_agent_summary(
    tmp_path: Path,
) -> None:
    from pde_agent.core.supervisor_tui import build_supervisor_run_view

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name, state_hint in [
        ("fno_training_plan.json", "FNO_TRAINING_CANDIDATES_PLANNED"),
        ("fno_training_optimization_evidence.json", "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED"),
        ("hypothesis_algorithm_design.json", "HYPOTHESIS_ALGORITHM_DESIGN_BUILT"),
        ("hypothesis_validation_result.json", "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"),
    ]:
        payload = {"state_hint": state_hint}
        if name == "fno_training_optimization_evidence.json":
            payload["selected_candidate"] = {
                "candidate_id": "fno_train_candidate_002",
                "fno_selection_score": 75.75,
                "config": {"forecast_steps": 95},
                "early_rel_mse": 0.6294,
                "middle_guardrail": {"evaluated": True, "rel_mse": 1.9201},
                "late_guardrail": {"evaluated": False},
            }
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "claude_sdk_session.json").write_text(
        json.dumps(
            {
                "state_hint": "CLAUDE_AGENT_SDK_RUNTIME_RAN",
                "metrics": {"sdk_message_count": 94, "dry_run": False},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    view = build_supervisor_run_view(run_dir=run_dir)

    assert view["state_hint"] == "SUPERVISOR_TUI_RUN_VIEW_BUILT"
    assert view["read_only"] is True
    assert view["run_id"] == "run"
    assert view["pipeline"][0]["artifact"] == "fno_training_plan.json"
    assert view["pipeline"][0]["exists"] is True
    by_artifact = {item["artifact"]: item for item in view["pipeline"]}
    assert by_artifact["hypothesis_validation_result.json"]["state_hint"] == (
        "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"
    )
    assert by_artifact["strategy_proposal.json"]["exists"] is False
    assert view["fno_evidence"]["selected_candidate_id"] == "fno_train_candidate_002"
    assert view["agent_session"]["sdk_message_count"] == 94
    assert "Pipeline" in view["text"]
    assert "Agent Session" in view["text"]


def test_supervisor_tui_cli_renders_text_and_json(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "early_rel_mse": 0.6,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.9},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    text_result = runner.invoke(
        app,
        ["supervisor-tui-runs", "--runs-root", str(tmp_path / "runs"), "--limit", "1"],
    )
    assert text_result.exit_code == 0
    assert "run_a" in text_result.stdout
    assert "early_rel" in text_result.stdout

    help_result = runner.invoke(app, ["supervisor-tui-runs", "--help"])
    assert help_result.exit_code == 0
    assert "--wide" in help_result.stdout
    assert "--watch" in help_result.stdout

    json_result = runner.invoke(
        app,
        ["supervisor-tui-run", "--run-dir", str(run_dir), "--json"],
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["state_hint"] == "SUPERVISOR_TUI_RUN_VIEW_BUILT"
    assert payload["read_only"] is True
    assert payload["fno_evidence"]["selected_candidate_id"] == "fno_train_candidate_002"


def test_supervisor_tui_prefers_latest_reselected_fno_evidence(tmp_path: Path) -> None:
    from pde_agent.core.supervisor_tui import build_supervisor_run_view

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fno_training_optimization_evidence.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "late_candidate",
                    "early_rel_mse": 0.95,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 2.28},
                    "late_guardrail": {"evaluated": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "fno_training_optimization_evidence_reselected.json").write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "selected_candidate": {
                    "candidate_id": "early_middle_candidate",
                    "early_rel_mse": 0.63,
                    "middle_guardrail": {"evaluated": True, "rel_mse": 1.92},
                    "late_guardrail": {"evaluated": False},
                },
            }
        ),
        encoding="utf-8",
    )

    view = build_supervisor_run_view(run_dir=run_dir)

    assert view["fno_evidence"]["artifact"] == (
        "fno_training_optimization_evidence_reselected.json"
    )
    assert view["fno_evidence"]["selected_candidate_id"] == "early_middle_candidate"
    assert "fno_training_optimization_evidence_reselected.json" in view["text"]
