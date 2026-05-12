import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_research_chronicle_summarizes_artifacts_and_updates_logs(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    audit = _write_json(
        tmp_path / "audit.json",
        {
            "status": "passed",
            "arm": "real_hdf5_deeponet_strategy_sweep",
            "session_id": "20260428_172142_d7472e",
            "tool_call_summary": {"read_file": 3, "terminal": 6},
            "terminal_artifact": {
                "state_hint": "STRATEGY_PROPOSAL_RECORDED",
                "strategy_id": "burgers_spectral_viscosity_real_hdf5_v0",
            },
        },
    )
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {
            "state_hint": "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED",
            "decision": {
                "decision": "retrieve_external_strategy",
                "reason_codes": ["negative_mean_score_delta"],
            },
            "metrics": {
                "best_strategy_id": "branch_trunk_spectral",
                "mean_score_delta": -11.8,
                "failure_rate": 1.0,
            },
        },
    )
    pipeline = _write_json(
        tmp_path / "pipeline.json",
        {
            "state_hint": "COMPETITION_TASK_PIPELINE_COMPLETED",
            "task_id": 1,
            "model_family": "fno",
            "audit": {"decision": "passed", "issue_count": 0},
            "materialized": {
                "submission": {
                    "metrics": {
                        "prediction_shape": [1000, 200, 256],
                        "inference_time": 51.2,
                    }
                }
            },
        },
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "task1_logs.log").write_text(
        "Hermes PDE Agent Scientific Log\nExperiment Tracking:\n",
        encoding="utf-8",
    )

    report = build_research_chronicle(
        artifact_paths=[audit, benchmark, pipeline],
        output_path=tmp_path / "research_chronicle.md",
        submission_dir=submission_dir,
        task_ids=[1],
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert report["state_hint"] == "RESEARCH_CHRONICLE_BUILT"
    assert "Hermes Research Chronicle" in chronicle
    assert "real_hdf5_deeponet_strategy_sweep" in chronicle
    assert "retrieve_external_strategy" in chronicle
    assert "negative_mean_score_delta" in chronicle
    assert "burgers_spectral_viscosity_real_hdf5_v0" in chronicle
    assert "model_family=fno" in chronicle
    assert "Arxiv Retrieval Policy" in chronicle

    updated_log = (submission_dir / "task1_logs.log").read_text(encoding="utf-8")
    assert "Hermes Research Chronicle" in updated_log
    assert "Evidence Sources" in updated_log


def test_build_research_chronicle_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    artifact = _write_json(
        tmp_path / "pipeline.json",
        {
            "state_hint": "COMPETITION_TASK_PIPELINE_COMPLETED",
            "task_id": 2,
            "model_family": "fno",
            "audit": {"decision": "passed", "issue_count": 0},
        },
    )
    output_path = tmp_path / "chronicle.md"

    result = CliRunner().invoke(
        app,
        [
            "build-research-chronicle",
            "--artifact-path",
            str(artifact),
            "--output-path",
            str(output_path),
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-research-chronicle"
    assert payload["state_hint"] == "RESEARCH_CHRONICLE_BUILT"
    assert payload["metrics"]["artifact_count"] == 1


def test_build_research_chronicle_summarizes_local_strategy_loop(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    loop_report = _write_json(
        tmp_path / "local_loop.json",
        {
            "state_hint": "LOCAL_STRATEGY_LOOP_RAN",
            "iteration_count": 2,
            "iterations": [
                {
                    "iteration": 1,
                    "strategy_id": "fno_spectral_viscosity_filter_v0",
                    "leaderboard_decision": "continue_scheduled_experiments",
                    "route_command": "claim-next-scheduled-experiment",
                },
                {
                    "iteration": 2,
                    "strategy_id": "fno_rollout_consistency_finetune_v0",
                    "leaderboard_decision": "accept_best_strategy",
                    "route_command": "record-strategy-proposal",
                },
            ],
            "decision": {
                "decision": "accept_best_strategy",
                "next_command": "record-strategy-proposal",
                "reason_codes": ["score_delta_meets_threshold"],
            },
        },
    )
    leaderboard = _write_json(
        tmp_path / "leaderboard.json",
        {
            "state_hint": "LOCAL_STRATEGY_LEADERBOARD_UPDATED",
            "result_count": 2,
            "best_result": {
                "strategy_id": "fno_rollout_consistency_finetune_v0",
                "score_delta": 3.0,
            },
            "decision": {
                "decision": "accept_best_strategy",
                "next_command": "record-strategy-proposal",
                "reason_codes": ["score_delta_meets_threshold"],
            },
        },
    )

    report = build_research_chronicle(
        artifact_paths=[loop_report, leaderboard],
        output_path=tmp_path / "chronicle.md",
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert "Local Strategy Loop:" in chronicle
    assert "iteration_count=2" in chronicle
    assert "fno_spectral_viscosity_filter_v0" in chronicle
    assert "fno_rollout_consistency_finetune_v0" in chronicle
    assert "best_strategy=fno_rollout_consistency_finetune_v0" in chronicle
    assert "score_delta=3.0" in chronicle


def test_build_research_chronicle_summarizes_operator_tuning_and_submission_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    tuning = _write_json(
        tmp_path / "operator_tuning.json",
        {
            "state_hint": "OPERATOR_TUNING_LOOP_RAN",
            "operator_family": "UNet",
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 1,
            },
            "best_projected_full_inference_seconds": 99.0,
            "decision": {
                "action": "accept_execution_config",
                "reason_codes": ["projected_inference_under_timeout"],
            },
        },
    )
    prediction = _write_json(
        tmp_path / "candidate_execution_report.json",
        {
            "state_hint": "UNET_COMPETITION_PREDICTION_READY",
            "candidate": {"model_family": "UNet", "task_id": 1},
            "inputs": {"sample_count": 1000, "forecast_steps": 190},
            "inference": {
                "elapsed_seconds": 94.7,
                "inference_batch_size": 8,
                "torch_num_threads": 1,
                "phase_timings": {"rollout_seconds": 94.0},
            },
        },
    )
    submission_candidate = _write_json(
        tmp_path / "submission_candidate.json",
        {
            "state_hint": "SUBMISSION_CANDIDATE_READY",
            "operator_family": "UNet",
            "selected_config": {
                "inference_batch_size": 8,
                "torch_num_threads": 1,
            },
            "materialized": {
                "submission": {
                    "metrics": {
                        "prediction_shape": [1000, 200, 256],
                        "inference_time": 94.7,
                    }
                }
            },
            "audit": {"decision": "passed", "issue_count": 0},
        },
    )

    report = build_research_chronicle(
        artifact_paths=[tuning, prediction, submission_candidate],
        output_path=tmp_path / "chronicle.md",
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert "Operator Runtime Tuning:" in chronicle
    assert "operator=UNet" in chronicle
    assert "inference_batch_size=8" in chronicle
    assert "torch_num_threads=1" in chronicle
    assert "UNet Competition Prediction:" in chronicle
    assert "rollout_seconds=94.0" in chronicle
    assert "Hermes Submission Candidate:" in chronicle
    assert "audit_decision=passed" in chronicle


def test_build_research_chronicle_records_repair_policy_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    repair_action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    runtime = _write_json(
        tmp_path / "runtime.json",
        {
            "state_hint": "RUNTIME_CLOSED_LOOP_RAN",
            "metrics": {
                "repair_policy_recommended_action_count": 1,
                "repair_policy_nonconverged_repair_error_count": 1,
                "repair_policy_converged_repair_error_count": 0,
            },
            "candidate_plan": {
                "repair_policy_recommended_actions": [repair_action],
                "repair_nonconvergence_errors": ["repair_convergence_consistency"],
                "repair_convergence_errors": [],
            },
            "heartbeat": {
                "best_score": 101.0,
                "decision": "continue_next_heartbeat",
            },
        },
    )
    heartbeat = _write_json(
        tmp_path / "heartbeat.json",
        {
            "state_hint": "HEARTBEAT_RAN",
            "parameters": {
                "repair_policy_summary_path": str(tmp_path / "repair_policy.json"),
                "repair_policy_recommended_action_count": 1,
            },
            "repair_policy_recommended_actions": [repair_action],
            "batches": [
                {
                    "batch_index": 1,
                    "candidate_generator": "physics_residual_nudge",
                    "repair_policy_recommended_actions": [repair_action],
                    "repair_policy_recommended_action_count": 1,
                }
            ],
            "best_candidate": {"candidate_id": "physics_repair_candidate"},
            "best_score": 101.0,
            "decision": "continue_next_heartbeat",
            "stop_reason": "max_batches_exhausted",
        },
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "task1_logs.log").write_text(
        "Hermes PDE Agent Scientific Log\n\nTrace:\n- prior\n",
        encoding="utf-8",
    )

    report = build_research_chronicle(
        artifact_paths=[runtime, heartbeat],
        output_path=tmp_path / "chronicle.md",
        submission_dir=submission_dir,
        task_ids=[1],
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert "Repair Policy Memory:" in chronicle
    assert "runtime_action_count=1" in chronicle
    assert "heartbeat_action_count=1" in chronicle
    assert "audit_error=repair_convergence_consistency" in chronicle
    assert "action=switch_repair_strategy_or_reload_missing_evidence" in chronicle
    assert "reason=repair_nonconverged" in chronicle
    assert "nonconverged_repair_errors=repair_convergence_consistency" in chronicle

    updated_log = (submission_dir / "task1_logs.log").read_text(encoding="utf-8")
    assert "Repair Policy Memory:" in updated_log
    assert "switch_repair_strategy_or_reload_missing_evidence" in updated_log


def test_build_research_chronicle_records_auto_repair_memory_bridge(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    audit_chain = _write_json(
        tmp_path / "audit_chain.json",
        {
            "state_hint": "RUNTIME_AUDIT_CHAIN_RAN",
            "decision": "repaired",
            "repair_convergence": {
                "decision": "converged",
                "input_repair_errors": ["runtime_memory_bridge_smoke_gate"],
                "repeated_audit_errors": [],
            },
            "metrics": {
                "auto_repair_attempted": True,
                "auto_repair_decision": "passed",
                "auto_repair_memory_bridge_passed": True,
                "auto_repair_repair_policy_recommended_action_count": 2,
                "auto_repair_batch_memory_constraint_action_count": 2,
            },
            "auto_repair": {
                "artifacts": {
                    "smoke_report_path": str(tmp_path / "smoke.json"),
                    "runtime_loop_report_path": str(tmp_path / "runtime.json"),
                    "audit_report_path": str(tmp_path / "audit.json"),
                },
                "metrics": {
                    "memory_bridge_passed": True,
                    "repair_policy_recommended_action_count": 2,
                    "batch_memory_constraint_action_count": 2,
                },
            },
            "chain_learning_signals": ["audit_repair_auto_repaired"],
        },
    )

    report = build_research_chronicle(
        artifact_paths=[audit_chain],
        output_path=tmp_path / "chronicle.md",
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert "auto_repair_decision=passed" in chronicle
    assert "memory_bridge_passed=True" in chronicle
    assert "auto_repair_action_count=2" in chronicle
    assert "converged_repair_errors=runtime_memory_bridge_smoke_gate" in chronicle
    assert (
        "repair_action source=auto_repair "
        "audit_error=runtime_memory_bridge_smoke_gate "
        "action=prefer_previously_converged_repair_path "
        "reason=repair_converged"
    ) in chronicle
