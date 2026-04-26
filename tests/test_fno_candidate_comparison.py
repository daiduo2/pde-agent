import json
from pathlib import Path

import pytest

from pde_agent.core.fno_candidate_comparison import (
    FNOCandidateComparisonError,
    compare_fno_candidates,
)


def _write_evaluation_report(
    path: Path,
    *,
    experiment_id: str,
    intervention: str,
    score: float,
    rel_mse: float,
    rmse: float,
    baseline_score: float | None = 50.0,
) -> None:
    baseline_comparison = (
        {
            "baseline_score_path": str(path.parent / "baseline_score.json"),
            "baseline_score": baseline_score,
            "candidate_score": score,
            "score_delta": score - baseline_score,
            "improved": score > baseline_score,
        }
        if baseline_score is not None
        else None
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_CANDIDATE_EVALUATED",
                "candidate_report_path": str(path.parent / "candidate_report.json"),
                "target_path": str(path.parent / "windows.npz"),
                "prediction_path": str(path.parent / "candidate_prediction.npz"),
                "artifacts": {
                    "evaluation_report_path": str(path),
                },
                "candidate": {
                    "experiment_id": experiment_id,
                    "intervention": intervention,
                },
                "score": {
                    "competition_like_score_short": score,
                    "rel_mse": rel_mse,
                    "rmse": rmse,
                },
                "error_diagnosis": {
                    "dominant_error_pattern": "long_horizon_drift",
                    "late_rel_mse": rel_mse * 1.2,
                    "time_error_slope": 0.01,
                    "shock_region_error_ratio": 1.5,
                },
                "physics": {
                    "physics_issue": "mass_drift",
                    "residual_ratio": 0.2,
                    "total_variation_ratio": 1.1,
                    "gradient_error_ratio": 0.9,
                },
                "selected_hypothesis": {
                    "state_hint": "PDE_HYPOTHESIS_SELECTED",
                    "selected_hypothesis": intervention,
                    "next_experiment_scope": "training_schedule_only",
                },
                "baseline_comparison": baseline_comparison,
                "next_command": "compare-fno-candidates",
            }
        ),
        encoding="utf-8",
    )


def test_compare_fno_candidates_ranks_by_score_and_writes_report(
    tmp_path: Path,
) -> None:
    weak_path = tmp_path / "weak" / "candidate_evaluation_report.json"
    strong_path = tmp_path / "strong" / "candidate_evaluation_report.json"
    output_path = tmp_path / "comparison.json"
    _write_evaluation_report(
        weak_path,
        experiment_id="candidate_weak",
        intervention="use_scheduled_sampling_rollout_finetune",
        score=48.0,
        rel_mse=0.3,
        rmse=0.4,
    )
    _write_evaluation_report(
        strong_path,
        experiment_id="candidate_strong",
        intervention="use_scheduled_sampling_rollout_finetune",
        score=52.0,
        rel_mse=0.2,
        rmse=0.3,
    )

    report = compare_fno_candidates(
        evaluation_report_paths=[weak_path, strong_path],
        output_path=output_path,
        force=False,
    )

    assert report["state_hint"] == "FNO_CANDIDATES_COMPARED"
    assert report["candidate_count"] == 2
    assert report["winner"]["experiment_id"] == "candidate_strong"
    assert [candidate["experiment_id"] for candidate in report["candidates"]] == [
        "candidate_strong",
        "candidate_weak",
    ]
    assert report["candidates"][0]["rank"] == 1
    assert report["candidates"][0]["rejection_reason"] is None
    assert report["candidates"][1]["rank"] == 2
    assert report["candidates"][1]["rejection_reason"] == "lower_score"
    assert report["rejected_candidates"][0]["experiment_id"] == "candidate_weak"
    assert report["next_command"] == "prepare-next-fno-iteration"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == report


def test_compare_fno_candidates_supports_single_candidate(tmp_path: Path) -> None:
    report_path = tmp_path / "candidate" / "candidate_evaluation_report.json"
    _write_evaluation_report(
        report_path,
        experiment_id="candidate_only",
        intervention="use_scheduled_sampling_rollout_finetune",
        score=51.0,
        rel_mse=0.25,
        rmse=0.35,
        baseline_score=None,
    )

    report = compare_fno_candidates(
        evaluation_report_paths=[report_path],
        output_path=tmp_path / "comparison.json",
        force=False,
    )

    assert report["candidate_count"] == 1
    assert report["winner"]["experiment_id"] == "candidate_only"
    assert report["winner"]["baseline_score"] is None
    assert report["winner"]["score_delta"] is None
    assert report["rejected_candidates"] == []


def test_compare_fno_candidates_rejects_duplicate_experiment_ids(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first" / "candidate_evaluation_report.json"
    second_path = tmp_path / "second" / "candidate_evaluation_report.json"
    _write_evaluation_report(
        first_path,
        experiment_id="duplicate_candidate",
        intervention="use_scheduled_sampling_rollout_finetune",
        score=51.0,
        rel_mse=0.25,
        rmse=0.35,
    )
    _write_evaluation_report(
        second_path,
        experiment_id="duplicate_candidate",
        intervention="use_scheduled_sampling_rollout_finetune",
        score=52.0,
        rel_mse=0.2,
        rmse=0.3,
    )

    with pytest.raises(FNOCandidateComparisonError, match="duplicate experiment_id"):
        compare_fno_candidates(
            evaluation_report_paths=[first_path, second_path],
            output_path=tmp_path / "comparison.json",
            force=False,
        )
