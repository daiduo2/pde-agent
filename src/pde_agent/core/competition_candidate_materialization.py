"""Materialize an executed candidate into competition submission artifacts."""


import json
from pathlib import Path
from typing import Any

from pde_agent.core.competition_scoring import score_competition_prediction
from pde_agent.core.competition_submission import build_competition_submission


STATE_HINT = "COMPETITION_CANDIDATE_MATERIALIZED"


class CompetitionCandidateMaterializationError(RuntimeError):
    pass


def materialize_competition_candidate(
    *,
    task_id: int,
    candidate_report_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    source_prediction_key: str = "prediction",
    train_time: float | None = None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    candidate_file = Path(candidate_report_path)
    candidate_report = _load_json_object(candidate_file)
    candidate_prediction_path = _candidate_prediction_path(candidate_report)
    elapsed = _train_time(candidate_report) if train_time is None else train_time
    inference_elapsed = _inference_time(candidate_report)
    output_root = Path(output_dir)
    submission = build_competition_submission(
        task_id=task_id,
        test_path=Path(test_path),
        output_dir=output_root / "submission_package",
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method="external_prediction",
        source_prediction_path=candidate_prediction_path,
        source_prediction_key=source_prediction_key,
        train_time=elapsed,
        inference_time=inference_elapsed,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    score = None
    score_path = None
    if target_path is not None:
        score_path = output_root / f"task{task_id}_competition_score.json"
        score = score_competition_prediction(
            task_id=task_id,
            prediction_path=Path(submission["artifacts"]["prediction_path"]),
            target_path=Path(target_path),
            output_path=score_path,
            prediction_dataset_key=prediction_dataset_key,
            target_dataset_key=dataset_key,
        )

    report_path = output_root / "competition_candidate_report.json"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "candidate_report_path": str(candidate_file),
        "candidate": candidate_report.get("candidate", {}),
        "submission": submission,
        "score": score,
        "artifacts": {
            "competition_candidate_report_path": str(report_path),
            "submission_zip_path": submission["artifacts"]["submission_zip_path"],
            "submission_dir": submission["artifacts"]["submission_dir"],
            "prediction_path": submission["artifacts"]["prediction_path"],
            "score_json_path": str(score_path) if score_path is not None else None,
        },
        "next_command": "review-or-submit-competition-package",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CompetitionCandidateMaterializationError(
            f"candidate report not found: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompetitionCandidateMaterializationError(
            f"failed to parse candidate report {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CompetitionCandidateMaterializationError(
            "candidate report must be a JSON object"
        )
    return value


def _candidate_prediction_path(report: dict[str, Any]) -> Path:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CompetitionCandidateMaterializationError(
            "candidate report must include artifacts"
        )
    value = artifacts.get("candidate_prediction_path")
    if not isinstance(value, str) or not value:
        raise CompetitionCandidateMaterializationError(
            "candidate report must include artifacts.candidate_prediction_path"
        )
    return Path(value)


def _train_time(report: dict[str, Any]) -> float:
    train = report.get("train")
    if isinstance(train, dict):
        value = train.get("elapsed_seconds")
        if isinstance(value, int | float):
            return max(0.0, float(value))
    return 0.0


def _inference_time(report: dict[str, Any]) -> float | None:
    inference = report.get("inference")
    if isinstance(inference, dict):
        value = inference.get("elapsed_seconds")
        if isinstance(value, int | float):
            return max(0.0, float(value))
    return None
