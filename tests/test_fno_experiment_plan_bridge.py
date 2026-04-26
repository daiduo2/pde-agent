import json
from pathlib import Path

import pytest

from pde_agent.core.next_experiment_plan import (
    NextExperimentPlanError,
    plan_next_experiment_from_fno_summary,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fno_planning_bundle(
    root: Path,
    *,
    pattern: str = "long_horizon_drift",
    issue: str = "mass_drift",
    hypothesis: str = "use_scheduled_sampling_rollout_finetune",
) -> tuple[Path, Path]:
    score_path = root / "fno_score.json"
    error_path = root / "fno_rollout_error.json"
    physics_path = root / "fno_burgers_physics.json"
    summary_path = root / "fno_rollout_summary.json"
    hypothesis_path = root / "selected_pde_hypothesis.json"

    _write_json(
        score_path,
        {
            "competition_like_score_short": 49.0,
            "rel_mse": 0.07,
            "rmse": 0.16,
        },
    )
    _write_json(
        error_path,
        {
            "dominant_error_pattern": pattern,
            "late_rel_mse": 0.12,
            "time_error_slope": 0.01,
            "shock_region_error_ratio": 1.3,
        },
    )
    _write_json(
        physics_path,
        {
            "physics_issue": issue,
            "residual_ratio": 1.1,
            "total_variation_ratio": 1.2,
            "gradient_error_ratio": 1.4,
        },
    )
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "state_hint": "FNO_ROLLOUT_DIAGNOSED",
            "artifacts": {
                "score_json_path": str(score_path),
                "rollout_error_json_path": str(error_path),
                "burgers_physics_json_path": str(physics_path),
            },
        },
    )
    _write_json(
        hypothesis_path,
        {
            "selected_hypothesis": hypothesis,
            "state_hint": "PDE_HYPOTHESIS_SELECTED",
        },
    )
    return summary_path, hypothesis_path


def test_plan_next_experiment_from_fno_summary_delegates_to_existing_planner(
    tmp_path: Path,
) -> None:
    summary_path, hypothesis_path = _write_fno_planning_bundle(tmp_path)
    output_path = tmp_path / "next_experiment_plan.json"

    plan = plan_next_experiment_from_fno_summary(
        summary_path=summary_path,
        hypothesis_path=hypothesis_path,
        output_path=output_path,
    )

    assert plan["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert plan["selected_intervention"] == "use_scheduled_sampling_rollout_finetune"
    assert plan["change_scope"] == "training_schedule_only"
    assert plan["summary_path"] == str(summary_path)
    assert plan["hypothesis_path"] == str(hypothesis_path)
    assert plan["score_path"] == str(tmp_path / "fno_score.json")
    assert plan["next_experiment_plan_path"] == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == plan


def test_plan_next_experiment_from_fno_summary_rejects_wrong_state_hint(
    tmp_path: Path,
) -> None:
    summary_path, hypothesis_path = _write_fno_planning_bundle(tmp_path)
    output_path = tmp_path / "next_experiment_plan.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["state_hint"] = "PDE_HYPOTHESIS_SELECTED"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NextExperimentPlanError, match="FNO_ROLLOUT_DIAGNOSED"):
        plan_next_experiment_from_fno_summary(
            summary_path=summary_path,
            hypothesis_path=hypothesis_path,
            output_path=output_path,
        )

    assert not output_path.exists()


def test_plan_next_experiment_from_fno_summary_cli_writes_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    summary_path, hypothesis_path = _write_fno_planning_bundle(tmp_path)
    output_path = tmp_path / "next_experiment_plan.json"

    result = CliRunner().invoke(
        app,
        [
            "plan-next-experiment-from-fno-summary",
            "--summary-path",
            str(summary_path),
            "--hypothesis-path",
            str(hypothesis_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "plan-next-experiment-from-fno-summary"
    assert payload["state_hint"] == "NEXT_EXPERIMENT_PLANNED"
    assert payload["artifacts"]["next_experiment_plan_path"] == str(output_path)
    assert (
        payload["metrics"]["selected_intervention"]
        == "use_scheduled_sampling_rollout_finetune"
    )
    assert payload["metrics"]["change_scope"] == "training_schedule_only"
    assert payload["metrics"]["next_command"] == "generate-experiment-config"
    assert payload["next_hint"] == "generate-experiment-config"
