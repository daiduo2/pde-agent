import json
from pathlib import Path

import pde_agent.core.fno_evidence_sweep as sweep_module
from pde_agent.core.fno_evidence_sweep import run_fno_evidence_sweep


def test_run_fno_evidence_sweep_writes_summary_and_recommendation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "burgers.hdf5"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "evidence"
    source_path.write_bytes(b"hdf5-placeholder")
    checkpoint_path.write_bytes(b"checkpoint-placeholder")

    def fake_build_local_window_suite(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        window_path = output_dir / "windows_f003.npz"
        output_dir.mkdir(parents=True, exist_ok=True)
        window_path.write_bytes(b"npz-placeholder")
        return {"profiles": [{"windows_npz_path": str(window_path)}]}

    def fake_run_fno_iteration_dryrun(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "fno_score.json").write_text(
            json.dumps({"competition_like_score_short": 40.0}),
            encoding="utf-8",
        )
        return {
            "metrics": {
                "competition_like_score_short": 40.0,
                "selected_intervention": "use_scheduled_sampling_rollout_finetune",
            }
        }

    def fake_audit_experiment_config_from_file(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            json.dumps(
                {
                    "state_hint": "EXPERIMENT_CONFIG_AUDITED",
                    "decision": "approved_for_execution",
                    "config_path": str(output_path.parent / "experiment_config.json"),
                }
            ),
            encoding="utf-8",
        )
        return {}

    def fake_prepare_candidate_execution_from_audit(**kwargs):
        candidate_index = kwargs["candidate_index"]
        experiment_id = f"use_scheduled_sampling_rollout_finetune_{candidate_index + 1:03d}"
        workspace = Path(kwargs["output_dir"]) / experiment_id
        artifacts = workspace / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        plan_path = workspace / "candidate_execution_plan.json"
        strategy = (
            "scheduled_sampling"
            if candidate_index == 0
            else "late_weighted_scheduled_sampling"
        )
        return {
            "candidate": {
                "experiment_id": experiment_id,
                "parameters": {
                    "rollout_steps": 10,
                    "teacher_forcing_decay": 0.95,
                    "training_strategy": strategy,
                },
            },
            "workspace": {
                "workspace_dir": str(workspace),
                "artifacts_dir": str(artifacts),
                "candidate_execution_plan_path": str(plan_path),
            },
        }

    def fake_execute_fno_finetune_candidate(**kwargs):
        artifacts = Path(kwargs["candidate_plan_path"]).parent / "artifacts"
        report_path = artifacts / "candidate_execution_report.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "train": {
                "loss_trace": [0.1, 0.05],
            },
            "artifacts": {
                "candidate_execution_report_path": str(report_path),
            },
        }

    def fake_evaluate_fno_candidate(**kwargs):
        eval_dir = Path(kwargs["output_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_dir / "candidate_evaluation_report.json"
        is_second = "002" in str(kwargs["candidate_report_path"])
        score = 41.0 if is_second else 42.0
        report_path.write_text(json.dumps({"score": score}), encoding="utf-8")
        return {
            "baseline_comparison": {
                "baseline_score": 40.0,
                "score_delta": score - 40.0,
            },
            "score": {
                "competition_like_score_short": score,
                "rel_mse": 0.2,
                "rmse": 0.3,
            },
            "error_diagnosis": {
                "dominant_error_pattern": "long_horizon_drift",
            },
            "physics": {
                "physics_issue": "inconclusive",
            },
        }

    def fake_compare_fno_candidates(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {
            "winner": {
                "experiment_id": "use_scheduled_sampling_rollout_finetune_001",
                "score": 42.0,
                "score_delta": 2.0,
            }
        }

    monkeypatch.setattr(
        sweep_module, "build_local_window_suite", fake_build_local_window_suite
    )
    monkeypatch.setattr(
        sweep_module, "run_fno_iteration_dryrun", fake_run_fno_iteration_dryrun
    )
    monkeypatch.setattr(
        sweep_module,
        "audit_experiment_config_from_file",
        fake_audit_experiment_config_from_file,
    )
    monkeypatch.setattr(
        sweep_module,
        "prepare_candidate_execution_from_audit",
        fake_prepare_candidate_execution_from_audit,
    )
    monkeypatch.setattr(
        sweep_module,
        "execute_fno_finetune_candidate",
        fake_execute_fno_finetune_candidate,
    )
    monkeypatch.setattr(
        sweep_module,
        "evaluate_fno_candidate",
        fake_evaluate_fno_candidate,
    )
    monkeypatch.setattr(
        sweep_module,
        "compare_fno_candidates",
        fake_compare_fno_candidates,
    )

    summary = run_fno_evidence_sweep(
        source_path=source_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_key="tensor",
        start_indices=[0],
        sample_limit=2,
        context_steps=4,
        forecast_steps=3,
        time_reduction=1,
        space_reduction=1,
        train_steps=[1],
        learning_rate=1e-4,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert summary["state_hint"] == "FNO_EVIDENCE_SWEEP_COMPLETED"
    assert summary["record_count"] == 2
    assert summary["comparison_count"] == 1
    assert summary["recommendation"]["add_more_training_strategies"] is False
    assert summary["recommendation"]["recommended_training_strategy"] == (
        "scheduled_sampling"
    )
    assert Path(summary["artifacts"]["summary_json_path"]).exists()
    assert Path(summary["artifacts"]["summary_csv_path"]).exists()
    assert {
        record["training_strategy"] for record in summary["records"]
    } == {
        "scheduled_sampling",
        "late_weighted_scheduled_sampling",
    }
    written = json.loads(
        (output_dir / "evidence_summary.json").read_text(encoding="utf-8")
    )
    assert written == summary


def test_run_fno_evidence_sweep_cli_writes_payload(tmp_path: Path, monkeypatch) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    import pde_agent.commands.run_fno_evidence_sweep as command_module

    source_path = tmp_path / "burgers.hdf5"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "evidence"
    source_path.write_bytes(b"hdf5-placeholder")
    checkpoint_path.write_bytes(b"checkpoint-placeholder")

    def fake_run_fno_evidence_sweep(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "evidence_summary.json"
        csv_path = output / "evidence_summary.csv"
        summary_path.write_text("{}", encoding="utf-8")
        csv_path.write_text("header\n", encoding="utf-8")
        return {
            "state_hint": "FNO_EVIDENCE_SWEEP_COMPLETED",
            "artifacts": {
                "summary_json_path": str(summary_path),
                "summary_csv_path": str(csv_path),
            },
            "record_count": 2,
            "comparison_count": 1,
            "strategy_statistics": {
                "scheduled_sampling": {"best_score": 42.0},
            },
            "recommendation": {
                "add_more_training_strategies": False,
                "recommended_training_strategy": "scheduled_sampling",
                "next_command": "run-fno-iteration-dryrun",
            },
        }

    monkeypatch.setattr(
        command_module, "run_fno_evidence_sweep", fake_run_fno_evidence_sweep
    )

    result = CliRunner().invoke(
        app,
        [
            "run-fno-evidence-sweep",
            "--source-path",
            str(source_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--dataset-key",
            "tensor",
            "--start-index",
            "0",
            "--sample-limit",
            "2",
            "--context-steps",
            "4",
            "--forecast-steps",
            "3",
            "--time-reduction",
            "1",
            "--space-reduction",
            "1",
            "--train-steps",
            "1",
            "--learning-rate",
            "0.0001",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "run-fno-evidence-sweep"
    assert payload["state_hint"] == "FNO_EVIDENCE_SWEEP_COMPLETED"
    assert payload["artifacts"]["summary_json_path"] == str(
        output_dir / "evidence_summary.json"
    )
    assert payload["metrics"]["record_count"] == 2
    assert payload["metrics"]["comparison_count"] == 1
    assert payload["metrics"]["best_score"] == 42.0
    assert payload["metrics"]["add_more_training_strategies"] is False
    assert payload["next_hint"] == "run-fno-iteration-dryrun"
