import json
from pathlib import Path


def _write_lookup_report(tmp_path: Path) -> Path:
    from pde_agent.core.local_strategy_library import (
        build_local_strategy_library,
        lookup_local_strategies,
    )

    library_path = tmp_path / "library.json"
    lookup_path = tmp_path / "lookup.json"
    build_local_strategy_library(output_path=library_path, force=False)
    lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="FNO",
        failure_modes=["long_horizon_drift", "spectral_dissipation_mismatch"],
        max_results=3,
        output_path=lookup_path,
    )
    return lookup_path


def test_schedule_local_strategy_evaluation_builds_ranked_experiment_queue(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_scheduler import (
        schedule_local_strategy_evaluation,
    )

    lookup_path = _write_lookup_report(tmp_path)
    output_path = tmp_path / "schedule.json"

    schedule = schedule_local_strategy_evaluation(
        lookup_report_path=lookup_path,
        output_path=output_path,
        workspace_root=tmp_path / "workspace",
        max_experiments=2,
        max_train_seconds=3600.0,
        max_inference_seconds=120.0,
        force=False,
    )

    assert schedule["state_hint"] == "LOCAL_STRATEGY_EVALUATION_SCHEDULED"
    assert schedule["experiment_count"] == 2
    assert schedule["next_command"] == "claim-next-scheduled-experiment"
    assert output_path.exists()

    first = schedule["experiments"][0]
    assert first["strategy_id"] == "fno_spectral_viscosity_filter_v0"
    assert first["status"] == "queued"
    assert first["priority_rank"] == 1
    assert first["workspace_dir"].endswith("fno_spectral_viscosity_filter_v0_001")
    assert first["execution_plan"]["planned_commands"][0]["command"] == (
        "predict-fno-competition"
    )
    assert first["execution_plan"]["budget"]["max_inference_seconds"] == 120.0
    assert first["evaluation_protocol"]["primary_metric"] == (
        "competition_segment_score"
    )


def test_claim_next_scheduled_experiment_skips_completed_ids(tmp_path: Path) -> None:
    from pde_agent.core.local_strategy_scheduler import (
        claim_next_scheduled_experiment,
        schedule_local_strategy_evaluation,
    )

    lookup_path = _write_lookup_report(tmp_path)
    schedule_path = tmp_path / "schedule.json"
    claim_path = tmp_path / "claim.json"
    schedule = schedule_local_strategy_evaluation(
        lookup_report_path=lookup_path,
        output_path=schedule_path,
        workspace_root=tmp_path / "workspace",
        max_experiments=3,
        max_train_seconds=3600.0,
        max_inference_seconds=120.0,
        force=False,
    )

    claim = claim_next_scheduled_experiment(
        schedule_path=schedule_path,
        completed_experiment_ids=[schedule["experiments"][0]["experiment_id"]],
        output_path=claim_path,
    )

    assert claim["state_hint"] == "SCHEDULED_EXPERIMENT_CLAIMED"
    assert claim["selected_experiment"]["priority_rank"] == 2
    assert claim["selected_experiment"]["status"] == "claimed"
    assert claim["next_command"] == claim["selected_experiment"]["next_command"]
    assert claim_path.exists()


def test_local_strategy_scheduler_cli_schedule_and_claim(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    lookup_path = _write_lookup_report(tmp_path)
    schedule_path = tmp_path / "schedule.json"
    claim_path = tmp_path / "claim.json"
    runner = CliRunner()

    scheduled = runner.invoke(
        app,
        [
            "schedule-local-strategy-evaluation",
            "--lookup-report-path",
            str(lookup_path),
            "--output-path",
            str(schedule_path),
            "--workspace-root",
            str(tmp_path / "workspace"),
            "--max-experiments",
            "2",
        ],
    )

    assert scheduled.exit_code == 0
    scheduled_payload = json.loads(scheduled.stdout)
    assert scheduled_payload["status"] == "ok"
    assert scheduled_payload["state_hint"] == "LOCAL_STRATEGY_EVALUATION_SCHEDULED"
    assert scheduled_payload["metrics"]["experiment_count"] == 2

    claimed = runner.invoke(
        app,
        [
            "claim-next-scheduled-experiment",
            "--schedule-path",
            str(schedule_path),
            "--output-path",
            str(claim_path),
        ],
    )

    assert claimed.exit_code == 0
    claim_payload = json.loads(claimed.stdout)
    assert claim_payload["status"] == "ok"
    assert claim_payload["state_hint"] == "SCHEDULED_EXPERIMENT_CLAIMED"
    assert claim_payload["metrics"]["strategy_id"] == (
        "fno_spectral_viscosity_filter_v0"
    )
