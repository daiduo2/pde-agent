import json
from pathlib import Path

import pytest

from pde_agent.core.pde_hypothesis import (
    PDEHypothesisError,
    select_pde_hypothesis_from_fno_summary,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fno_diagnosis_bundle(
    root: Path,
    *,
    pattern: str = "long_horizon_drift",
    issue: str = "mass_drift",
) -> Path:
    score_path = root / "fno_score.json"
    error_path = root / "fno_rollout_error.json"
    physics_path = root / "fno_burgers_physics.json"
    summary_path = root / "fno_rollout_summary.json"

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
    return summary_path


def test_select_pde_hypothesis_from_fno_summary_delegates_to_existing_selector(
    tmp_path: Path,
) -> None:
    summary_path = _write_fno_diagnosis_bundle(tmp_path)
    output_path = tmp_path / "selected_pde_hypothesis.json"

    result = select_pde_hypothesis_from_fno_summary(
        summary_path=summary_path,
        output_path=output_path,
    )

    assert result["selected_hypothesis"] == "use_scheduled_sampling_rollout_finetune"
    assert result["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert result["summary_path"] == str(summary_path)
    assert result["score_path"] == str(tmp_path / "fno_score.json")
    assert result["hypothesis_path"] == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == result


def test_select_pde_hypothesis_from_fno_summary_rejects_wrong_state_hint(
    tmp_path: Path,
) -> None:
    summary_path = _write_fno_diagnosis_bundle(tmp_path)
    output_path = tmp_path / "selected_pde_hypothesis.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["state_hint"] = "LOCAL_ROLLOUT_SCORED"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PDEHypothesisError, match="FNO_ROLLOUT_DIAGNOSED"):
        select_pde_hypothesis_from_fno_summary(
            summary_path=summary_path,
            output_path=output_path,
        )

    assert not output_path.exists()
