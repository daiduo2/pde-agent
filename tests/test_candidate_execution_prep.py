import json
from pathlib import Path

import pytest

from pde_agent.core.candidate_execution_prep import (
    CandidateExecutionPrepError,
    prepare_candidate_execution_from_audit,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config_and_audit(root: Path, *, decision: str = "approved_for_execution") -> Path:
    config_path = root / "experiment_config.json"
    audit_path = root / "experiment_config_audit.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "state_hint": "EXPERIMENT_CONFIG_GENERATED",
            "selected_intervention": "use_scheduled_sampling_rollout_finetune",
            "change_scope": "training_schedule_only",
            "requires_training": True,
            "budget": {
                "max_experiment_count": 2,
                "max_train_time_seconds": 3600,
                "max_inference_time_seconds": 120,
                "requires_training": True,
            },
            "candidate_experiments": [
                {
                    "experiment_id": "use_scheduled_sampling_rollout_finetune_001",
                    "intervention": "use_scheduled_sampling_rollout_finetune",
                    "change_scope": "training_schedule_only",
                    "requires_training": True,
                    "parameters": {
                        "rollout_steps": 10,
                        "teacher_forcing_ratio": 0.75,
                    },
                    "expected_artifacts": ["candidate prediction NPZ"],
                    "forbidden_actions": ["overwrite previous artifacts"],
                }
            ],
            "next_command": "execute-candidate-experiment",
        },
    )
    _write_json(
        audit_path,
        {
            "schema_version": 1,
            "state_hint": "EXPERIMENT_CONFIG_AUDITED",
            "decision": decision,
            "issue_count": 0 if decision == "approved_for_execution" else 1,
            "issues": [],
            "config_path": str(config_path),
            "plan_path": str(root / "next_experiment_plan.json"),
        },
    )
    return audit_path


def test_prepare_candidate_execution_creates_workspace_and_plan(
    tmp_path: Path,
) -> None:
    audit_path = _write_config_and_audit(tmp_path)
    output_dir = tmp_path / "candidate_workspaces"

    plan = prepare_candidate_execution_from_audit(
        audit_path=audit_path,
        output_dir=output_dir,
        candidate_index=0,
        force=False,
    )

    candidate_dir = output_dir / "use_scheduled_sampling_rollout_finetune_001"
    assert plan["state_hint"] == "CANDIDATE_EXECUTION_PREPARED"
    assert plan["candidate"]["experiment_id"] == (
        "use_scheduled_sampling_rollout_finetune_001"
    )
    assert plan["execution_status"] == "not_started"
    assert plan["next_command"] == "execute-candidate-experiment"
    assert Path(plan["workspace"]["workspace_dir"]) == candidate_dir
    assert (candidate_dir / "artifacts").is_dir()
    assert (candidate_dir / "logs").is_dir()
    assert (candidate_dir / "candidate_execution_plan.json").exists()
    written = json.loads(
        (candidate_dir / "candidate_execution_plan.json").read_text(encoding="utf-8")
    )
    assert written == plan


def test_prepare_candidate_execution_rejects_unapproved_audit(
    tmp_path: Path,
) -> None:
    audit_path = _write_config_and_audit(tmp_path, decision="rejected")
    output_dir = tmp_path / "candidate_workspaces"

    with pytest.raises(CandidateExecutionPrepError, match="approved_for_execution"):
        prepare_candidate_execution_from_audit(
            audit_path=audit_path,
            output_dir=output_dir,
            candidate_index=0,
            force=False,
        )

    assert not output_dir.exists()
