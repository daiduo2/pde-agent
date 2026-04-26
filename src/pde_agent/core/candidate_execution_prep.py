"""Prepare an audited candidate experiment workspace without executing it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CandidateExecutionPrepError(RuntimeError):
    pass


def prepare_candidate_execution_from_audit(
    *,
    audit_path: str | Path,
    output_dir: str | Path,
    candidate_index: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    audit_file = _require_file(Path(audit_path), "experiment config audit")
    audit = _load_json_object(audit_file, "experiment config audit")
    _validate_audit(audit)

    config_file = _require_file(Path(audit["config_path"]), "experiment config")
    config = _load_json_object(config_file, "experiment config")
    _validate_config(config)
    candidates = config["candidate_experiments"]
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
        or candidate_index >= len(candidates)
    ):
        raise CandidateExecutionPrepError(
            "candidate_index must select an existing candidate experiment"
        )

    candidate = candidates[candidate_index]
    if not isinstance(candidate, dict):
        raise CandidateExecutionPrepError("selected candidate experiment must be an object")
    experiment_id = candidate.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CandidateExecutionPrepError(
            "selected candidate experiment must include string experiment_id"
        )

    workspace_root = Path(output_dir) / experiment_id
    artifacts_dir = workspace_root / "artifacts"
    logs_dir = workspace_root / "logs"
    plan_path = workspace_root / "candidate_execution_plan.json"
    if plan_path.exists() and not force:
        raise CandidateExecutionPrepError(
            f"candidate execution plan already exists: {plan_path}"
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(
        audit_file=audit_file,
        config_file=config_file,
        candidate_index=candidate_index,
        candidate=candidate,
        workspace_root=workspace_root,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        plan_path=plan_path,
    )
    plan_path.write_text(
        json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def _build_plan(
    *,
    audit_file: Path,
    config_file: Path,
    candidate_index: int,
    candidate: dict[str, Any],
    workspace_root: Path,
    artifacts_dir: Path,
    logs_dir: Path,
    plan_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": "CANDIDATE_EXECUTION_PREPARED",
        "audit_path": str(audit_file),
        "config_path": str(config_file),
        "candidate_index": candidate_index,
        "candidate": candidate,
        "workspace": {
            "workspace_dir": str(workspace_root),
            "artifacts_dir": str(artifacts_dir),
            "logs_dir": str(logs_dir),
            "candidate_execution_plan_path": str(plan_path),
        },
        "execution_status": "not_started",
        "next_command": "execute-candidate-experiment",
    }


def _validate_audit(audit: dict[str, Any]) -> None:
    if audit.get("state_hint") != "EXPERIMENT_CONFIG_AUDITED":
        raise CandidateExecutionPrepError(
            "audit must include state_hint EXPERIMENT_CONFIG_AUDITED"
        )
    if audit.get("decision") != "approved_for_execution":
        raise CandidateExecutionPrepError(
            "audit decision must be approved_for_execution before preparing "
            "candidate execution"
        )
    config_path = audit.get("config_path")
    if not isinstance(config_path, str):
        raise CandidateExecutionPrepError("audit must include string config_path")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("state_hint") != "EXPERIMENT_CONFIG_GENERATED":
        raise CandidateExecutionPrepError(
            "config must include state_hint EXPERIMENT_CONFIG_GENERATED"
        )
    candidates = config.get("candidate_experiments")
    if not isinstance(candidates, list) or not candidates:
        raise CandidateExecutionPrepError(
            "config must include non-empty candidate_experiments list"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateExecutionPrepError(
            f"failed to parse {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CandidateExecutionPrepError(f"{label} JSON must be an object")
    return data


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise CandidateExecutionPrepError(f"missing {label} path: {path}")
    if not path.is_file():
        raise CandidateExecutionPrepError(f"{label} path is not a file: {path}")
    return path
