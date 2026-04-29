"""Plan Hermes next actions from structured convergence diagnoses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_NEXT_ACTION_PLANNED"


class HermesNextActionPolicyError(RuntimeError):
    pass


def plan_hermes_next_action(
    *,
    diagnosis_path: str | Path,
    output_path: str | Path | None,
) -> dict[str, Any]:
    diagnosis_file = Path(diagnosis_path)
    diagnosis = _read_json_object(diagnosis_file)
    if diagnosis.get("state_hint") != "HERMES_RESULT_DIAGNOSED":
        raise HermesNextActionPolicyError("diagnosis has invalid state_hint")
    plan_core = _plan(diagnosis)
    plan = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "diagnosis_path": str(diagnosis_file),
        "source_artifact_path": diagnosis.get("source_artifact_path"),
        "diagnosis": {
            "primary_bottleneck": diagnosis.get("primary_bottleneck"),
            "recommended_action_family": diagnosis.get("recommended_action_family"),
            "confidence": diagnosis.get("confidence"),
        },
        **plan_core,
    }
    if output_path is not None:
        output_file = Path(output_path)
        plan["plan_path"] = str(output_file)
        _write_json(output_file, plan)
    return plan


def _plan(diagnosis: dict[str, Any]) -> dict[str, Any]:
    family = diagnosis.get("recommended_action_family")
    operator_family = diagnosis.get("operator_family") or "UNet"
    if family == "runtime_tuning":
        return {
            "action": "run_operator_runtime_tuning",
            "command": "run-hermes-operator-tuning-loop",
            "arguments": {
                "operator_family": operator_family,
                "source_artifact_path": diagnosis.get("source_artifact_path"),
            },
            "requires_human_review": False,
            "reason_codes": ["diagnosed_runtime_failure"],
            "terminal": False,
        }
    if family == "operator_strategy":
        return {
            "action": "run_local_operator_strategy_loop",
            "command": "run-hermes-local-strategy-loop",
            "arguments": {
                "model_family": operator_family,
                "source_artifact_path": diagnosis.get("source_artifact_path"),
            },
            "requires_human_review": False,
            "reason_codes": ["diagnosed_accuracy_failure"],
            "terminal": False,
        }
    if family == "continue_local_schedule":
        return {
            "action": "claim_next_scheduled_experiment",
            "command": "claim-next-scheduled-experiment",
            "arguments": {"source_artifact_path": diagnosis.get("source_artifact_path")},
            "requires_human_review": False,
            "reason_codes": ["pending_local_schedule"],
            "terminal": False,
        }
    if family == "research_log":
        return {
            "action": "build_research_chronicle",
            "command": "build-research-chronicle",
            "arguments": {"source_artifact_path": diagnosis.get("source_artifact_path")},
            "requires_human_review": False,
            "reason_codes": ["local_strategy_exhausted"],
            "terminal": True,
        }
    return {
        "action": "ready_for_submission_prediction",
        "command": _submission_command(operator_family),
        "arguments": {
            "operator_family": operator_family,
            "source_artifact_path": diagnosis.get("source_artifact_path"),
        },
        "requires_human_review": False,
        "reason_codes": ["no_blocking_bottleneck"],
        "terminal": True,
    }


def _submission_command(operator_family: str) -> str:
    if operator_family == "UNet":
        return "predict-unet-competition"
    if operator_family == "FNO":
        return "predict-fno-competition"
    return "build-competition-submission"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HermesNextActionPolicyError(f"missing diagnosis path: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesNextActionPolicyError(f"invalid diagnosis JSON: {path}") from exc
    if not isinstance(data, dict):
        raise HermesNextActionPolicyError("diagnosis JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
