"""End-to-end competition task pipeline for official operator checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_submission_audit import audit_competition_submission
from pde_agent.core.fno_competition_prediction import predict_fno_competition
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "COMPETITION_TASK_PIPELINE_COMPLETED"


class CompetitionTaskPipelineError(RuntimeError):
    pass


def run_competition_task_pipeline(
    *,
    task_id: int,
    model_family: str,
    test_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    submission_id: str,
    target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    force: bool = False,
) -> dict[str, Any]:
    family = _model_family(model_family)
    output_root = Path(output_dir)
    pipeline_report_path = output_root / "competition_task_pipeline_report.json"
    if pipeline_report_path.exists() and not force:
        raise CompetitionTaskPipelineError(
            f"competition task pipeline report already exists: {pipeline_report_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    prediction = _predict(
        family=family,
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=output_root / f"{family}_prediction",
        dataset_key=dataset_key,
        force=force,
    )
    materialized = materialize_competition_candidate(
        task_id=task_id,
        candidate_report_path=prediction["artifacts"]["candidate_execution_report_path"],
        test_path=Path(test_path),
        output_dir=output_root / "materialized",
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        prediction_dataset_key=dataset_key,
        source_prediction_key="prediction",
        train_time=None,
        force=force,
    )
    audit = audit_competition_submission(
        task_id=task_id,
        submission_dir=materialized["artifacts"]["submission_dir"],
        test_path=Path(test_path),
        output_path=output_root / "competition_submission_audit.json",
        dataset_key=dataset_key,
    )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "task_id": task_id,
        "model_family": family,
        "test_path": str(test_path),
        "checkpoint_path": str(checkpoint_path),
        "prediction": prediction,
        "materialized": materialized,
        "audit": audit,
        "artifacts": {
            "pipeline_report_path": str(pipeline_report_path),
            "candidate_execution_report_path": prediction["artifacts"][
                "candidate_execution_report_path"
            ],
            "candidate_prediction_path": prediction["artifacts"][
                "candidate_prediction_path"
            ],
            "submission_zip_path": materialized["artifacts"]["submission_zip_path"],
            "submission_dir": materialized["artifacts"]["submission_dir"],
            "audit_path": audit["audit_path"],
            "score_json_path": materialized["artifacts"].get("score_json_path"),
        },
        "next_command": "submit-if-audit-passed",
    }
    pipeline_report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _predict(
    *,
    family: str,
    task_id: int,
    test_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_key: str,
    force: bool,
) -> dict[str, Any]:
    if family == "fno":
        return predict_fno_competition(
            task_id=task_id,
            test_path=test_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            dataset_key=dataset_key,
            force=force,
        )
    if family == "unet":
        return predict_unet_competition(
            task_id=task_id,
            test_path=test_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            dataset_key=dataset_key,
            force=force,
        )
    raise CompetitionTaskPipelineError(f"unsupported model family: {family}")


def _model_family(value: str) -> str:
    family = value.strip().lower()
    if family in {"fno", "unet"}:
        return family
    raise CompetitionTaskPipelineError("model_family must be fno or unet")
