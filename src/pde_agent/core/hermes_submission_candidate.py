"""Build a competition submission candidate from Hermes-selected operator config."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.competition_candidate_materialization import (
    materialize_competition_candidate,
)
from pde_agent.core.competition_submission_audit import audit_competition_submission
from pde_agent.core.research_chronicle import build_research_chronicle
from pde_agent.core.unet_competition_prediction import predict_unet_competition


STATE_HINT = "HERMES_SUBMISSION_CANDIDATE_READY"
Predictor = Callable[..., dict[str, Any]]
Materializer = Callable[..., dict[str, Any]]
ChronicleBuilder = Callable[..., dict[str, Any]]
Auditor = Callable[..., dict[str, Any]]


class HermesSubmissionCandidateError(RuntimeError):
    pass


def run_hermes_submission_candidate(
    *,
    tuning_report_path: str | Path,
    output_dir: str | Path,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    submission_id: str,
    target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
    force: bool = False,
    predictor: Predictor | None = None,
    materializer: Materializer | None = None,
    chronicle_builder: ChronicleBuilder | None = None,
    auditor: Auditor | None = None,
) -> dict[str, Any]:
    tuning_file = Path(tuning_report_path)
    tuning = _read_json_object(tuning_file)
    if tuning.get("state_hint") != "HERMES_OPERATOR_TUNING_LOOP_RAN":
        raise HermesSubmissionCandidateError("tuning report has invalid state_hint")
    operator_family = tuning.get("operator_family")
    if operator_family != "UNet":
        raise HermesSubmissionCandidateError("submission candidate currently supports UNet")
    selected_config = tuning.get("selected_config")
    if not isinstance(selected_config, dict):
        raise HermesSubmissionCandidateError("tuning report missing selected_config")
    batch_size = _positive_int(selected_config.get("inference_batch_size"), "batch")
    thread_count = _positive_int(selected_config.get("torch_num_threads"), "threads")

    root = Path(output_dir)
    report_path = root / "hermes_submission_candidate.json"
    if report_path.exists() and not force:
        raise HermesSubmissionCandidateError(
            f"Hermes submission candidate already exists: {report_path}"
        )
    root.mkdir(parents=True, exist_ok=True)
    run_predictor = predictor or predict_unet_competition
    run_materializer = materializer or materialize_competition_candidate
    run_chronicle = chronicle_builder or build_research_chronicle
    run_auditor = auditor or audit_competition_submission

    prediction = run_predictor(
        task_id=task_id,
        test_path=Path(test_path),
        checkpoint_path=Path(checkpoint_path),
        output_dir=root / "prediction",
        dataset_key=dataset_key,
        inference_batch_size=batch_size,
        torch_num_threads=thread_count,
        force=True,
    )
    candidate_report_path = Path(
        prediction["artifacts"]["candidate_execution_report_path"]
    )
    materialized = run_materializer(
        task_id=task_id,
        candidate_report_path=candidate_report_path,
        test_path=Path(test_path),
        output_dir=root / "materialized",
        submission_id=submission_id,
        target_path=Path(target_path) if target_path is not None else None,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        source_prediction_key="prediction",
        train_time=None,
        force=True,
    )
    submission_dir = Path(materialized["artifacts"]["submission_dir"])
    chronicle = run_chronicle(
        artifact_paths=[str(tuning_file), str(candidate_report_path)],
        output_path=root / "research_chronicle.md",
        submission_dir=submission_dir,
        task_ids=[task_id],
        force=True,
    )
    audit = run_auditor(
        task_id=task_id,
        submission_dir=str(submission_dir),
        test_path=str(test_path),
        output_path=str(root / "submission_audit.json"),
        dataset_key=dataset_key,
    )

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "candidate_report_path": str(report_path),
        "operator_family": operator_family,
        "task_id": task_id,
        "selected_config": {
            "inference_batch_size": batch_size,
            "torch_num_threads": thread_count,
        },
        "tuning_report_path": str(tuning_file),
        "prediction": prediction,
        "materialized": materialized,
        "chronicle": chronicle,
        "audit": audit,
        "artifacts": {
            "hermes_submission_candidate_path": str(report_path),
            "submission_dir": str(submission_dir),
            "submission_zip_path": materialized["artifacts"]["submission_zip_path"],
            "chronicle_path": chronicle["artifacts"]["chronicle_path"],
            "audit_path": audit["audit_path"],
        },
        "next_command": (
            "submit-if-passed-or-review"
            if audit["decision"] == "passed"
            else "repair-submission-candidate"
        ),
    }
    _write_json(report_path, report)
    return report


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HermesSubmissionCandidateError(f"selected_config has invalid {label}")
    return int(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesSubmissionCandidateError(f"missing tuning report: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesSubmissionCandidateError(f"invalid tuning report JSON: {path}") from exc
    if not isinstance(data, dict):
        raise HermesSubmissionCandidateError("tuning report JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
