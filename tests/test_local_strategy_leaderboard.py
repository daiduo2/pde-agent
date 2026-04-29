import json
from pathlib import Path


def _write_schedule(tmp_path: Path) -> Path:
    from pde_agent.core.local_strategy_library import (
        build_local_strategy_library,
        lookup_local_strategies,
    )
    from pde_agent.core.local_strategy_scheduler import (
        schedule_local_strategy_evaluation,
    )

    library_path = tmp_path / "library.json"
    lookup_path = tmp_path / "lookup.json"
    schedule_path = tmp_path / "schedule.json"
    build_local_strategy_library(output_path=library_path, force=False)
    lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="FNO",
        failure_modes=["long_horizon_drift"],
        max_results=2,
        output_path=lookup_path,
    )
    schedule_local_strategy_evaluation(
        lookup_report_path=lookup_path,
        output_path=schedule_path,
        workspace_root=tmp_path / "workspace",
        max_experiments=2,
        max_train_seconds=3600.0,
        max_inference_seconds=120.0,
        force=False,
    )
    return schedule_path


def _write_result(path: Path, *, score: float, baseline_score: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "LOCAL_EXPERIMENT_RESULT",
                "score": score,
                "baseline_score": baseline_score,
                "inference_time_seconds": 50.0,
                "result_summary": "synthetic scheduler result",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_record_local_strategy_result_accepts_positive_best_experiment(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_leaderboard import (
        record_local_strategy_result,
    )

    schedule_path = _write_schedule(tmp_path)
    result_path = _write_result(tmp_path / "result.json", score=76.0, baseline_score=72.0)
    leaderboard_path = tmp_path / "leaderboard.json"

    leaderboard = record_local_strategy_result(
        schedule_path=schedule_path,
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=result_path,
        leaderboard_path=leaderboard_path,
        result_status="completed",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )

    assert leaderboard["state_hint"] == "LOCAL_STRATEGY_LEADERBOARD_UPDATED"
    assert leaderboard["result_count"] == 1
    assert leaderboard["best_result"]["strategy_id"] == (
        "fno_spectral_viscosity_filter_v0"
    )
    assert leaderboard["best_result"]["score_delta"] == 4.0
    assert leaderboard["decision"]["decision"] == "accept_best_strategy"
    assert leaderboard["decision"]["next_command"] == "record-strategy-proposal"
    assert leaderboard_path.exists()


def test_record_local_strategy_result_continues_after_negative_result(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_leaderboard import (
        record_local_strategy_result,
    )

    schedule_path = _write_schedule(tmp_path)
    result_path = _write_result(tmp_path / "result.json", score=70.0, baseline_score=72.0)

    leaderboard = record_local_strategy_result(
        schedule_path=schedule_path,
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=result_path,
        leaderboard_path=tmp_path / "leaderboard.json",
        result_status="completed",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )

    assert leaderboard["decision"]["decision"] == "continue_scheduled_experiments"
    assert leaderboard["decision"]["next_command"] == "claim-next-scheduled-experiment"
    assert leaderboard["completed_experiment_ids"] == [
        "fno_spectral_viscosity_filter_v0_001"
    ]
    assert leaderboard["pending_experiment_ids"] == [
        "fno_rollout_consistency_finetune_v0_002"
    ]


def test_local_strategy_leaderboard_cli_records_failed_result(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    schedule_path = _write_schedule(tmp_path)
    result_path = tmp_path / "failed_result.json"
    result_path.write_text(
        json.dumps(
            {
                "state_hint": "LOCAL_EXPERIMENT_RESULT",
                "error": "checkpoint incompatible",
            }
        ),
        encoding="utf-8",
    )
    leaderboard_path = tmp_path / "leaderboard.json"

    result = CliRunner().invoke(
        app,
        [
            "record-local-strategy-result",
            "--schedule-path",
            str(schedule_path),
            "--experiment-id",
            "fno_spectral_viscosity_filter_v0_001",
            "--result-summary-path",
            str(result_path),
            "--leaderboard-path",
            str(leaderboard_path),
            "--result-status",
            "failed",
            "--failure-reason",
            "checkpoint incompatible",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_STRATEGY_LEADERBOARD_UPDATED"
    assert payload["metrics"]["decision"] == "continue_scheduled_experiments"
    assert payload["metrics"]["result_count"] == 1
