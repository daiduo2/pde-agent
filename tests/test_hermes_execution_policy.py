from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_diagnose_heartbeat_report_routes_plateau_without_blueprint_to_hypothesis_lab(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_execution_policy import diagnose_heartbeat_report

    report_path = tmp_path / "heartbeat.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "stop_reason": "stagnation_detected",
                "parameters": {"candidate_generator": "postprocess"},
                "best_score": 85.6,
                "batches": [{"score_improved": False}],
            }
        ),
        encoding="utf-8",
    )

    diagnosis = diagnose_heartbeat_report(
        heartbeat_report_path=report_path,
        verified_blueprint_path=None,
        output_path=tmp_path / "diagnosis.json",
        force=False,
    )

    assert diagnosis["state_hint"] == "HERMES_EXECUTION_TRAJECTORY_DIAGNOSED"
    assert diagnosis["trajectory_state"] == "plateau"
    assert diagnosis["recommended_action"] == "run_hypothesis_lab"
    assert "stagnation_detected" in diagnosis["reason_codes"]
    assert Path(diagnosis["diagnosis_path"]).exists()


def test_diagnose_heartbeat_report_routes_plateau_with_blueprint_to_switch(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_execution_policy import diagnose_heartbeat_report

    report_path = tmp_path / "heartbeat.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "stop_reason": "stagnation_detected",
                "parameters": {"candidate_generator": "postprocess"},
                "best_score": 85.6,
                "batches": [{"score_improved": False}],
            }
        ),
        encoding="utf-8",
    )
    blueprint_path = tmp_path / "verified_blueprints.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED",
                "blueprints": [
                    {
                        "candidate_family": "linear_trend_spectral",
                        "seed_params": [{"params": {"slope_window": 5}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnosis = diagnose_heartbeat_report(
        heartbeat_report_path=report_path,
        verified_blueprint_path=blueprint_path,
        output_path=None,
        force=False,
    )

    assert diagnosis["recommended_action"] == "switch_to_verified_blueprint"
    assert diagnosis["recommended_candidate_generator"] == "linear_trend_spectral"
    assert diagnosis["verified_blueprint_path"] == str(blueprint_path)


def test_diagnose_heartbeat_report_does_not_repeat_same_blueprint_switch(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_execution_policy import diagnose_heartbeat_report

    blueprint_path = tmp_path / "verified_blueprints.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED",
                "blueprints": [{"candidate_family": "linear_trend_spectral"}],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "heartbeat.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "stop_reason": "stagnation_detected",
                "parameters": {
                    "candidate_generator": "linear_trend_spectral",
                    "hypothesis_blueprint_path": str(blueprint_path),
                },
                "best_score": 86.2,
                "batches": [{"score_improved": False}],
            }
        ),
        encoding="utf-8",
    )

    diagnosis = diagnose_heartbeat_report(
        heartbeat_report_path=report_path,
        verified_blueprint_path=blueprint_path,
        output_path=None,
        force=False,
    )

    assert diagnosis["recommended_action"] == "verified_blueprint_plateau"
    assert diagnosis["next_command"] == "run-hypothesis-lab"


def test_run_adaptive_heartbeat_switches_to_verified_blueprint_after_plateau(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_execution_policy import run_hermes_adaptive_heartbeat

    calls: list[dict[str, Any]] = []
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    blueprint_path = tmp_path / "verified_blueprints.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED",
                "blueprints": [{"candidate_family": "linear_trend_spectral"}],
            }
        ),
        encoding="utf-8",
    )

    def fake_heartbeat(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        index = len(calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / f"best_{index}"
        best_dir.mkdir()
        (best_dir / "task1_logs.log").write_text(
            "Hermes PDE Agent Scientific Log\n\nTrace:\n- prior\n\nExperiment Tracking:\n- prior\n",
            encoding="utf-8",
        )
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "candidate_generator": kwargs["candidate_generator"],
                "hypothesis_blueprint_path": str(kwargs.get("hypothesis_blueprint_path"))
                if kwargs.get("hypothesis_blueprint_path") is not None
                else None,
            },
            "batch_count": 1,
            "batches": [{"score_improved": index == 2}],
            "best_candidate": {
                "candidate_id": f"best_{index}",
                "score": 85.0 + index,
                "submission_dir": str(best_dir),
            },
            "best_score": 85.0 + index,
            "best_objective_score": 85.0 + index,
            "stop_reason": "stagnation_detected"
            if index == 1
            else "max_batches_exhausted",
            "decision": "review_before_next_heartbeat"
            if index == 1
            else "continue_next_heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    result = run_hermes_adaptive_heartbeat(
        output_dir=tmp_path / "adaptive",
        initial_submission_dir=initial_dir,
        verified_blueprint_path=blueprint_path,
        max_policy_stages=2,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        heartbeat_runner=fake_heartbeat,
        force=False,
    )

    assert len(calls) == 2
    assert calls[0]["candidate_generator"] == "postprocess"
    assert calls[1]["candidate_generator"] == "linear_trend_spectral"
    assert calls[1]["hypothesis_blueprint_path"] == blueprint_path
    assert result["best_candidate"]["candidate_id"] == "best_2"
    assert result["policy_events"][0]["action"] == "switch_to_verified_blueprint"
    log_text = (tmp_path / "best_2" / "task1_logs.log").read_text(encoding="utf-8")
    assert "Hermes Adaptive Execution Policy" in log_text
    assert "switch_to_verified_blueprint" in log_text


def test_run_hermes_adaptive_heartbeat_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_adaptive_heartbeat as command_module
    from typer.testing import CliRunner

    def fake_policy(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "adaptive.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ADAPTIVE_HEARTBEAT_RAN",
            "adaptive_report_path": str(report_path),
            "stage_count": 2,
            "policy_events": [{"action": "switch_to_verified_blueprint"}],
            "best_candidate": {"candidate_id": "best", "score": 86.2},
            "best_score": 86.2,
            "decision": "continue_next_adaptive_heartbeat",
            "next_command": "run-hermes-adaptive-heartbeat",
        }

    monkeypatch.setattr(command_module, "run_hermes_adaptive_heartbeat", fake_policy)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-adaptive-heartbeat",
            "--output-dir",
            str(tmp_path / "adaptive"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--verified-blueprint-path",
            str(tmp_path / "verified_blueprints.json"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_ADAPTIVE_HEARTBEAT_RAN"
    assert payload["metrics"]["best_score"] == 86.2
    assert payload["metrics"]["last_action"] == "switch_to_verified_blueprint"
