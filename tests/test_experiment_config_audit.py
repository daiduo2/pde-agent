import json
from pathlib import Path

from pde_agent.core.experiment_config import generate_experiment_config_from_file
from pde_agent.core.experiment_config_audit import audit_experiment_config_from_file


def _write_next_experiment_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "NEXT_EXPERIMENT_PLANNED",
                "selected_intervention": "apply_low_pass_stabilization",
                "source_hypothesis": "apply_low_pass_stabilization",
                "change_scope": "postprocess_or_inference_only",
                "intervention_plan": {
                    "description": "Sweep low-pass thresholds.",
                    "allowed_actions": ["sweep spectral cutoff"],
                    "forbidden_actions": ["overwrite previous artifacts"],
                    "candidate_parameters": {
                        "low_pass_cutoff_ratio": [0.5, 0.625, 0.75],
                        "preserve_mean": [True],
                    },
                },
                "budget": {
                    "max_experiment_count": 3,
                    "max_train_time_seconds": 0,
                    "max_inference_time_seconds": 120,
                    "requires_training": False,
                },
                "controls": {
                    "baseline_required": True,
                    "baseline_artifacts": ["previous prediction NPZ"],
                },
                "acceptance_criteria": ["score improves"],
                "rejection_criteria": ["score decreases"],
                "required_next_artifacts": [
                    "candidate prediction NPZ",
                    "candidate score JSON",
                ],
                "next_command": "generate-experiment-config",
                "rationale": "Fixture plan.",
            }
        ),
        encoding="utf-8",
    )


def test_audit_experiment_config_approves_generated_config(tmp_path: Path) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    config_path = tmp_path / "experiment_config.json"
    audit_path = tmp_path / "experiment_config_audit.json"
    _write_next_experiment_plan(plan_path)
    generate_experiment_config_from_file(plan_path=plan_path, output_path=config_path)

    audit = audit_experiment_config_from_file(
        config_path=config_path,
        output_path=audit_path,
        plan_path=None,
    )

    assert audit["state_hint"] == "EXPERIMENT_CONFIG_AUDITED"
    assert audit["decision"] == "approved_for_execution"
    assert audit["issue_count"] == 0
    assert audit["issues"] == []
    assert audit["summary"]["candidate_count"] == 3
    assert audit["summary"]["next_command"] == "execute-candidate-experiment"
    written = json.loads(audit_path.read_text(encoding="utf-8"))
    assert written == audit


def test_audit_experiment_config_rejects_tampered_candidate_parameter(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "next_experiment_plan.json"
    config_path = tmp_path / "experiment_config.json"
    audit_path = tmp_path / "experiment_config_audit.json"
    _write_next_experiment_plan(plan_path)
    generate_experiment_config_from_file(plan_path=plan_path, output_path=config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["candidate_experiments"][0]["parameters"]["low_pass_cutoff_ratio"] = 0.95
    config_path.write_text(json.dumps(config), encoding="utf-8")

    audit = audit_experiment_config_from_file(
        config_path=config_path,
        output_path=audit_path,
        plan_path=plan_path,
    )

    assert audit["decision"] == "rejected"
    assert audit["issue_count"] == 1
    assert audit["issues"][0]["code"] == "candidate_parameter_out_of_grid"
    assert audit["issues"][0]["severity"] == "error"


def test_audit_experiment_config_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    plan_path = tmp_path / "next_experiment_plan.json"
    config_path = tmp_path / "experiment_config.json"
    audit_path = tmp_path / "experiment_config_audit.json"
    _write_next_experiment_plan(plan_path)
    generate_experiment_config_from_file(plan_path=plan_path, output_path=config_path)

    result = CliRunner().invoke(
        app,
        [
            "audit-experiment-config",
            "--config-path",
            str(config_path),
            "--output-path",
            str(audit_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "audit-experiment-config"
    assert payload["state_hint"] == "EXPERIMENT_CONFIG_AUDITED"
    assert payload["artifacts"]["experiment_config_audit_path"] == str(audit_path)
    assert payload["metrics"]["decision"] == "approved_for_execution"
    assert payload["metrics"]["issue_count"] == 0
    assert payload["next_hint"] == "execute-candidate-experiment"
