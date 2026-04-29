import json
from pathlib import Path


def _write_artifact(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_route_local_strategy_next_action_routes_accepted_leaderboard(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_route import route_local_strategy_next_action

    artifact_path = _write_artifact(
        tmp_path / "leaderboard.json",
        {
            "state_hint": "LOCAL_STRATEGY_LEADERBOARD_UPDATED",
            "decision": {
                "decision": "accept_best_strategy",
                "next_command": "record-strategy-proposal",
            },
            "best_result": {
                "experiment_id": "fno_spectral_viscosity_filter_v0_001",
                "strategy_id": "fno_spectral_viscosity_filter_v0",
            },
        },
    )

    route = route_local_strategy_next_action(
        artifact_path=artifact_path,
        output_path=tmp_path / "route.json",
    )

    assert route["state_hint"] == "LOCAL_STRATEGY_NEXT_ACTION_ROUTED"
    assert route["route"]["action"] == "record_strategy_proposal"
    assert route["route"]["command"] == "record-strategy-proposal"
    assert route["route"]["requires_human_review"] is False
    assert route["route"]["reason_codes"] == ["accepted_local_strategy"]


def test_route_local_strategy_next_action_routes_continue_with_completed_ids(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_route import route_local_strategy_next_action

    artifact_path = _write_artifact(
        tmp_path / "leaderboard.json",
        {
            "state_hint": "LOCAL_STRATEGY_LEADERBOARD_UPDATED",
            "leaderboard_path": str(tmp_path / "leaderboard.json"),
            "schedule_path": str(tmp_path / "schedule.json"),
            "completed_experiment_ids": ["exp_001"],
            "decision": {
                "decision": "continue_scheduled_experiments",
                "next_command": "claim-next-scheduled-experiment",
            },
        },
    )

    route = route_local_strategy_next_action(
        artifact_path=artifact_path,
        output_path=tmp_path / "route.json",
    )

    assert route["route"]["action"] == "claim_next_scheduled_experiment"
    assert route["route"]["command"] == "claim-next-scheduled-experiment"
    assert route["route"]["arguments"]["completed_experiment_ids"] == ["exp_001"]
    assert route["route"]["arguments"]["schedule_path"] == str(tmp_path / "schedule.json")


def test_route_local_strategy_next_action_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    artifact_path = _write_artifact(
        tmp_path / "audit.json",
        {
            "state_hint": "LOCAL_STRATEGY_LOOP_AUDITED",
            "decision": "passed",
            "next_command": "use-local-strategy-loop",
        },
    )
    output_path = tmp_path / "route.json"
    result = CliRunner().invoke(
        app,
        [
            "route-local-strategy-next-action",
            "--artifact-path",
            str(artifact_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_STRATEGY_NEXT_ACTION_ROUTED"
    assert payload["metrics"]["action"] == "use_local_strategy_loop"
