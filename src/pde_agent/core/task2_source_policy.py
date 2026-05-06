"""Task 2 source-policy audit for avoiding Task 1 data and checkpoint leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


STATE_HINT = "TASK2_SOURCE_POLICY_AUDITED"


TASK1_MARKERS = (
    "task1",
    "task_1",
    "task-1",
    "1d_burgers_sols_nu0.001",
    "nu0.001_fno",
    "nu0.001_unet",
    "task1_current_best",
    "task1_scalar_source_gate",
)
class Task2SourcePolicyError(RuntimeError):
    pass


def audit_task2_source_policy(
    *,
    train_data_paths: Iterable[str | Path],
    test_data_paths: Iterable[str | Path],
    checkpoint_paths: Iterable[str | Path],
    candidate_source_paths: Iterable[str | Path],
    output_path: str | Path | None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    train_paths = [Path(path) for path in train_data_paths]
    test_paths = [Path(path) for path in test_data_paths]
    checkpoints = [Path(path) for path in checkpoint_paths]
    candidate_paths = [Path(path) for path in candidate_source_paths]

    for path in train_paths:
        if _looks_task1(path):
            issues.append(
                {
                    "code": "task1_data_path_forbidden",
                    "detail": str(path),
                }
            )
    for path in checkpoints:
        if _looks_task1(path):
            issues.append(
                {
                    "code": "task1_checkpoint_forbidden",
                    "detail": str(path),
                }
            )
    for path in candidate_paths:
        if _looks_task1(path):
            issues.append(
                {
                    "code": "task1_candidate_artifact_forbidden",
                    "detail": str(path),
                }
            )

    decision = "passed" if not issues else "failed"
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": decision,
        "issue_count": len(issues),
        "issue_codes": [issue["code"] for issue in issues],
        "issues": issues,
        "inputs": {
            "train_data_paths": [str(path) for path in train_paths],
            "test_data_paths": [str(path) for path in test_paths],
            "checkpoint_paths": [str(path) for path in checkpoints],
            "candidate_source_paths": [str(path) for path in candidate_paths],
        },
        "allowed_reuse": {
            "agent_code_and_search_policy": True,
            "physics_priors_without_task1_fitted_parameters": True,
            "model_architecture_templates": True,
        },
        "forbidden_reuse": {
            "task1_data_or_checkpoints": True,
            "task1_validation_tuned_thresholds": True,
            "task1_candidate_predictions": True,
        },
        "next_command": "proceed_to_task2_training"
        if decision == "passed"
        else "replace_task1_sources_before_task2_training",
    }
    if output_path is not None:
        audit_path = Path(output_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        report["audit_path"] = str(audit_path)
        audit_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _looks_task1(path: Path) -> bool:
    normalized = str(path).lower().replace("\\", "/")
    return any(marker in normalized for marker in TASK1_MARKERS)
