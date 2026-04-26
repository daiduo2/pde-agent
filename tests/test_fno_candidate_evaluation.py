import json
from pathlib import Path

from pde_agent.core.fno_candidate_evaluation import evaluate_fno_candidate


def _write_prediction_and_target(root: Path) -> tuple[Path, Path]:
    import numpy as np

    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    target = np.zeros((2, 3, 8), dtype=np.float32)
    prediction = np.zeros((2, 3, 8), dtype=np.float32)
    for sample in range(2):
        base = np.sin(x + sample * 0.1)
        delta = 0.04 * np.cos(x) + 0.01 * sample
        for time_index in range(3):
            target[sample, time_index, :] = base + time_index * delta
            prediction[sample, time_index, :] = target[sample, time_index, :] + 0.01

    prediction_path = root / "candidate_prediction.npz"
    target_path = root / "windows.npz"
    np.savez(prediction_path, prediction=prediction)
    np.savez(target_path, context=target[:, :2, :], target=target)
    return prediction_path, target_path


def _write_candidate_report(root: Path, prediction_path: Path) -> Path:
    report_path = root / "candidate_execution_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "FNO_FINETUNE_CANDIDATE_EXECUTED",
                "candidate": {
                    "experiment_id": "use_scheduled_sampling_rollout_finetune_001",
                    "intervention": "use_scheduled_sampling_rollout_finetune",
                },
                "artifacts": {
                    "candidate_prediction_path": str(prediction_path),
                },
                "next_command": "score-local-rollout",
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _write_baseline_score(path: Path, score: float) -> None:
    path.write_text(
        json.dumps(
            {
                "competition_like_score_short": score,
                "rel_mse": 0.5,
                "rmse": 0.5,
            }
        ),
        encoding="utf-8",
    )


def test_evaluate_fno_candidate_writes_diagnostic_bundle(tmp_path: Path) -> None:
    prediction_path, target_path = _write_prediction_and_target(tmp_path)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)
    baseline_score_path = tmp_path / "baseline_score.json"
    output_dir = tmp_path / "evaluation"
    _write_baseline_score(baseline_score_path, score=40.0)

    report = evaluate_fno_candidate(
        candidate_report_path=candidate_report_path,
        target_path=target_path,
        output_dir=output_dir,
        baseline_score_path=baseline_score_path,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    expected_files = {
        "candidate_score.json",
        "candidate_rollout_error.json",
        "candidate_burgers_physics.json",
        "candidate_selected_pde_hypothesis.json",
        "candidate_evaluation_report.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert report["state_hint"] == "FNO_CANDIDATE_EVALUATED"
    assert report["candidate"]["experiment_id"] == (
        "use_scheduled_sampling_rollout_finetune_001"
    )
    assert report["score"]["competition_like_score_short"] > 0
    assert report["baseline_comparison"]["baseline_score"] == 40.0
    assert report["baseline_comparison"]["score_delta"] == (
        report["score"]["competition_like_score_short"] - 40.0
    )
    assert report["selected_hypothesis"]["state_hint"] == "PDE_HYPOTHESIS_SELECTED"
    assert report["next_command"] == "compare-fno-candidates"
    written = json.loads(
        (output_dir / "candidate_evaluation_report.json").read_text(encoding="utf-8")
    )
    assert written == report


def test_evaluate_fno_candidate_without_baseline_uses_null_comparison(
    tmp_path: Path,
) -> None:
    prediction_path, target_path = _write_prediction_and_target(tmp_path)
    candidate_report_path = _write_candidate_report(tmp_path, prediction_path)

    report = evaluate_fno_candidate(
        candidate_report_path=candidate_report_path,
        target_path=target_path,
        output_dir=tmp_path / "evaluation",
        baseline_score_path=None,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["baseline_comparison"] is None
