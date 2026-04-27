import json
from pathlib import Path


def _write_deeponet_eval(
    path: Path,
    *,
    experiment_id: str,
    start_index: int,
    train_steps: int,
    score: float,
    score_delta: float,
    strategy: str = "branch_trunk_rollout",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "DEEPONET_CANDIDATE_EVALUATED",
                "inputs": {"forecast_steps": 31},
                "candidate": {
                    "experiment_id": experiment_id,
                    "start_index": start_index,
                    "train_steps": train_steps,
                    "training_strategy": strategy,
                },
                "baseline_comparison": {
                    "baseline_score": score - score_delta,
                    "score_delta": score_delta,
                    "improved": score_delta > 0,
                },
                "score": {
                    "competition_like_score_short": score,
                    "rel_mse": 0.2,
                    "rmse": 0.3,
                },
                "error_diagnosis": {
                    "dominant_error_pattern": "long_horizon_drift"
                },
                "physics": {"physics_issue": "inconclusive"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_deeponet_evidence_sweep_groups_and_selects_winners(
    tmp_path: Path,
) -> None:
    from pde_agent.core.deeponet_evidence_sweep import build_deeponet_evidence_sweep

    reports = [
        _write_deeponet_eval(
            tmp_path / "evals" / "start0_a.json",
            experiment_id="deeponet_a",
            start_index=0,
            train_steps=1,
            score=41.0,
            score_delta=1.0,
            strategy="plain_supervised",
        ),
        _write_deeponet_eval(
            tmp_path / "evals" / "start0_b.json",
            experiment_id="deeponet_b",
            start_index=0,
            train_steps=1,
            score=43.0,
            score_delta=3.0,
            strategy="branch_trunk_rollout",
        ),
        _write_deeponet_eval(
            tmp_path / "evals" / "start1_a.json",
            experiment_id="deeponet_c",
            start_index=1,
            train_steps=1,
            score=40.5,
            score_delta=0.5,
            strategy="branch_trunk_rollout",
        ),
    ]
    output_dir = tmp_path / "deeponet_evidence"

    summary = build_deeponet_evidence_sweep(
        evaluation_report_paths=reports,
        output_dir=output_dir,
    )

    assert summary["schema_version"] == 1
    assert summary["state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert summary["model_family"] == "DeepONet"
    assert summary["record_count"] == 3
    assert summary["comparison_count"] == 2
    assert summary["inputs"]["forecast_steps"] == 31
    assert summary["comparison_winners"] == [
        {
            "start_index": 0,
            "train_steps": 1,
            "comparison_group_key": "start_0000_steps_0001",
            "winner_experiment_id": "deeponet_b",
            "winner_training_strategy": "branch_trunk_rollout",
            "winner_score": 43.0,
            "winner_score_delta": 3.0,
        },
        {
            "start_index": 1,
            "train_steps": 1,
            "comparison_group_key": "start_0001_steps_0001",
            "winner_experiment_id": "deeponet_c",
            "winner_training_strategy": "branch_trunk_rollout",
            "winner_score": 40.5,
            "winner_score_delta": 0.5,
        },
    ]
    assert summary["strategy_statistics"]["branch_trunk_rollout"]["win_count"] == 2
    assert Path(summary["artifacts"]["summary_json_path"]).exists()
    assert Path(summary["artifacts"]["summary_csv_path"]).exists()
    assert json.loads((output_dir / "evidence_summary.json").read_text()) == summary


def test_build_deeponet_evidence_sweep_cli_writes_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    report = _write_deeponet_eval(
        tmp_path / "evals" / "candidate.json",
        experiment_id="deeponet_a",
        start_index=0,
        train_steps=1,
        score=41.0,
        score_delta=1.0,
    )
    output_dir = tmp_path / "deeponet_evidence"

    result = CliRunner().invoke(
        app,
        [
            "build-deeponet-evidence-sweep",
            "--evaluation-report-path",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-deeponet-evidence-sweep"
    assert payload["state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert payload["artifacts"]["summary_json_path"] == str(
        output_dir / "evidence_summary.json"
    )
    assert payload["metrics"]["record_count"] == 1
    assert payload["metrics"]["comparison_count"] == 1
    assert payload["metrics"]["winner_strategy_count"] == 1
    assert payload["next_hint"] == "convert-operator-evidence-summary"


def test_deeponet_evidence_sweep_converts_to_strategy_growth_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.deeponet_evidence_sweep import build_deeponet_evidence_sweep
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    report = _write_deeponet_eval(
        tmp_path / "evals" / "candidate.json",
        experiment_id="deeponet_a",
        start_index=0,
        train_steps=1,
        score=41.0,
        score_delta=1.0,
    )
    summary = build_deeponet_evidence_sweep(
        evaluation_report_paths=[report],
        output_dir=tmp_path / "deeponet_evidence",
    )

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=summary["artifacts"]["summary_json_path"],
        output_path=tmp_path / "strategy_evidence_summary.json",
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["model_family"] == "DeepONet"
    assert evidence["source_state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert evidence["evaluation_window_count"] == 1
    assert evidence["mean_score_delta"] == 1.0
    assert evidence["long_window_score_delta"] == 1.0
    assert evidence["failure_rate"] == 0.0
