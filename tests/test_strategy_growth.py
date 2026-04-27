import json
from pathlib import Path

import pytest


def _proposal_kwargs(output_path: Path) -> dict[str, object]:
    return {
        "strategy_id": "fno_spectral_filter_v0",
        "title": "Spectral filtering before rollout fine-tuning",
        "source_type": "agent_hypothesis",
        "problem_context": "1D Burgers fixed-nu short-window FNO rollout",
        "hypothesis": (
            "Removing unstable high-frequency residual modes improves "
            "long-window stability."
        ),
        "mechanism": (
            "Apply a low-pass residual filter before fine-tuning rollout "
            "predictions."
        ),
        "implementation_plan": {
            "model_family": "FNO",
            "allowed_commands": ["run-fno-iteration", "run-fno-evidence-sweep"],
            "expected_artifacts": ["candidate_comparison.json"],
        },
        "expected_observables": [
            "positive mean score delta across windows",
            "lower high-frequency residual energy",
        ],
        "known_risks": [
            "over-smoothing shocks",
            "single-window improvement may not generalize",
        ],
        "output_path": output_path,
    }


def _write_proposal_and_rubric(tmp_path: Path) -> tuple[Path, Path]:
    from pde_agent.core.strategy_growth import (
        build_strategy_rubric,
        record_strategy_proposal,
    )

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "experiment_rubric.json"
    record_strategy_proposal(**_proposal_kwargs(proposal_path))
    build_strategy_rubric(
        proposal_path=proposal_path,
        output_path=rubric_path,
        min_evaluation_windows=3,
        max_inference_time_regression=0.10,
    )
    return proposal_path, rubric_path


def _write_strong_evidence(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluation_window_count": 4,
                "mean_score_delta": 2.5,
                "score_delta_std": 0.8,
                "long_window_score_delta": 1.1,
                "physics_residual_delta": -0.05,
                "inference_time_regression": 0.02,
                "failure_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_promoted_strategy_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from pde_agent.core.strategy_growth import evaluate_strategy_promotion

    proposal_path, rubric_path = _write_proposal_and_rubric(tmp_path)
    evidence_path = _write_strong_evidence(tmp_path / "evidence_summary.json")
    promotion_path = tmp_path / "promotion_report.json"
    evaluate_strategy_promotion(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        output_path=promotion_path,
    )
    return proposal_path, rubric_path, evidence_path, promotion_path


def test_record_strategy_proposal_writes_valid_artifact(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import record_strategy_proposal

    output_path = tmp_path / "strategy_proposal.json"

    report = record_strategy_proposal(**_proposal_kwargs(output_path))

    assert output_path.exists()
    assert report == json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert report["strategy_id"] == "fno_spectral_filter_v0"
    assert report["source_type"] == "agent_hypothesis"
    assert report["strategy_proposal_path"] == str(output_path)
    assert report["next_command"] == "build-strategy-rubric"


def test_record_strategy_proposal_rejects_invalid_source_type(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_growth import (
        StrategyGrowthError,
        record_strategy_proposal,
    )

    kwargs = _proposal_kwargs(tmp_path / "strategy_proposal.json")
    kwargs["source_type"] = "random_blog_copy"

    with pytest.raises(StrategyGrowthError, match="unsupported source_type"):
        record_strategy_proposal(**kwargs)


def test_record_strategy_proposal_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_path = tmp_path / "strategy_proposal.json"
    result = CliRunner().invoke(
        app,
        [
            "record-strategy-proposal",
            "--strategy-id",
            "fno_spectral_filter_v0",
            "--title",
            "Spectral filtering before rollout fine-tuning",
            "--source-type",
            "agent_hypothesis",
            "--problem-context",
            "1D Burgers fixed-nu short-window FNO rollout",
            "--hypothesis",
            "Removing unstable high-frequency residual modes improves long-window stability.",
            "--mechanism",
            "Apply a low-pass residual filter before fine-tuning rollout predictions.",
            "--implementation-plan-json",
            '{"model_family":"FNO","allowed_commands":["run-fno-iteration"]}',
            "--expected-observable",
            "positive mean score delta across windows",
            "--known-risk",
            "over-smoothing shocks",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "record-strategy-proposal"
    assert payload["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert payload["artifacts"]["strategy_proposal_path"] == str(output_path)
    assert payload["metrics"]["strategy_id"] == "fno_spectral_filter_v0"
    assert payload["metrics"]["source_type"] == "agent_hypothesis"
    assert payload["metrics"]["observable_count"] == 1
    assert payload["metrics"]["risk_count"] == 1
    assert payload["next_hint"] == "build-strategy-rubric"


def test_build_strategy_rubric_requires_multi_window_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_growth import (
        build_strategy_rubric,
        record_strategy_proposal,
    )

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "experiment_rubric.json"
    record_strategy_proposal(**_proposal_kwargs(proposal_path))

    rubric = build_strategy_rubric(
        proposal_path=proposal_path,
        output_path=rubric_path,
        min_evaluation_windows=3,
        max_inference_time_regression=0.10,
    )

    assert rubric_path.exists()
    assert rubric == json.loads(rubric_path.read_text(encoding="utf-8"))
    assert rubric["state_hint"] == "STRATEGY_RUBRIC_BUILT"
    assert rubric["strategy_id"] == "fno_spectral_filter_v0"
    assert rubric["strategy_proposal_path"] == str(proposal_path)
    assert rubric["experiment_rubric_path"] == str(rubric_path)
    assert rubric["promotion_thresholds"]["min_evaluation_windows"] == 3
    assert rubric["promotion_thresholds"]["max_inference_time_regression"] == 0.10
    assert "mean_score_delta" in rubric["required_metrics"]
    assert "long_window_score_delta" in rubric["required_metrics"]
    assert "physics_residual_delta" in rubric["required_metrics"]
    assert rubric["next_command"] == "evaluate-strategy-promotion"


def test_build_strategy_rubric_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from pde_agent.core.strategy_growth import record_strategy_proposal
    from typer.testing import CliRunner

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "experiment_rubric.json"
    record_strategy_proposal(**_proposal_kwargs(proposal_path))

    result = CliRunner().invoke(
        app,
        [
            "build-strategy-rubric",
            "--proposal-path",
            str(proposal_path),
            "--output-path",
            str(rubric_path),
            "--min-evaluation-windows",
            "4",
            "--max-inference-time-regression",
            "0.05",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-strategy-rubric"
    assert payload["state_hint"] == "STRATEGY_RUBRIC_BUILT"
    assert payload["artifacts"]["experiment_rubric_path"] == str(rubric_path)
    assert payload["artifacts"]["strategy_proposal_path"] == str(proposal_path)
    assert payload["metrics"]["strategy_id"] == "fno_spectral_filter_v0"
    assert payload["metrics"]["min_evaluation_windows"] == 4
    assert payload["metrics"]["required_metric_count"] >= 7
    assert payload["next_hint"] == "evaluate-strategy-promotion"


def test_evaluate_strategy_promotion_promotes_strong_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_growth import evaluate_strategy_promotion

    proposal_path, rubric_path = _write_proposal_and_rubric(tmp_path)
    evidence_path = _write_strong_evidence(tmp_path / "evidence_summary.json")
    promotion_path = tmp_path / "promotion_report.json"

    report = evaluate_strategy_promotion(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        output_path=promotion_path,
    )

    assert promotion_path.exists()
    assert report == json.loads(promotion_path.read_text(encoding="utf-8"))
    assert report["state_hint"] == "STRATEGY_PROMOTION_EVALUATED"
    assert report["strategy_id"] == "fno_spectral_filter_v0"
    assert report["decision"] == "promote"
    assert report["promote_to_skill_candidate"] is True
    assert report["reason_codes"] == ["all_thresholds_passed"]
    assert report["next_command"] == "append-strategy-ledger"


def test_evaluate_strategy_promotion_rejects_single_window_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_growth import evaluate_strategy_promotion

    proposal_path, rubric_path = _write_proposal_and_rubric(tmp_path)
    evidence_path = tmp_path / "evidence_summary.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evaluation_window_count": 1,
                "mean_score_delta": 5.0,
                "score_delta_std": 0.0,
                "long_window_score_delta": 0.0,
                "physics_residual_delta": 0.0,
                "inference_time_regression": 0.0,
                "failure_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_strategy_promotion(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        output_path=tmp_path / "promotion_report.json",
    )

    assert report["decision"] == "reject"
    assert report["promote_to_skill_candidate"] is False
    assert "insufficient_evaluation_windows" in report["reason_codes"]


def test_evaluate_strategy_promotion_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    proposal_path, rubric_path = _write_proposal_and_rubric(tmp_path)
    evidence_path = _write_strong_evidence(tmp_path / "evidence_summary.json")
    promotion_path = tmp_path / "promotion_report.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-strategy-promotion",
            "--proposal-path",
            str(proposal_path),
            "--rubric-path",
            str(rubric_path),
            "--evidence-summary-path",
            str(evidence_path),
            "--output-path",
            str(promotion_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "evaluate-strategy-promotion"
    assert payload["state_hint"] == "STRATEGY_PROMOTION_EVALUATED"
    assert payload["artifacts"]["promotion_report_path"] == str(promotion_path)
    assert payload["metrics"]["strategy_id"] == "fno_spectral_filter_v0"
    assert payload["metrics"]["decision"] == "promote"
    assert payload["metrics"]["promote_to_skill_candidate"] is True
    assert payload["metrics"]["reason_count"] == 1
    assert payload["next_hint"] == "append-strategy-ledger"


def test_append_strategy_ledger_records_attempt(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import append_strategy_ledger

    proposal_path, rubric_path, evidence_path, promotion_path = (
        _write_promoted_strategy_artifacts(tmp_path)
    )
    ledger_path = tmp_path / "strategy_ledger.jsonl"

    row = append_strategy_ledger(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        ledger_path=ledger_path,
    )

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row
    assert row["schema_version"] == 1
    assert row["state_hint"] == "STRATEGY_LEDGER_APPENDED"
    assert row["strategy_id"] == "fno_spectral_filter_v0"
    assert row["decision"] == "promote"
    assert row["promote_to_skill_candidate"] is True
    assert row["reason_codes"] == ["all_thresholds_passed"]
    assert row["next_command"] == "create-skill-candidate"


def test_append_strategy_ledger_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    proposal_path, rubric_path, evidence_path, promotion_path = (
        _write_promoted_strategy_artifacts(tmp_path)
    )
    ledger_path = tmp_path / "strategy_ledger.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "append-strategy-ledger",
            "--proposal-path",
            str(proposal_path),
            "--rubric-path",
            str(rubric_path),
            "--evidence-summary-path",
            str(evidence_path),
            "--promotion-report-path",
            str(promotion_path),
            "--ledger-path",
            str(ledger_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "append-strategy-ledger"
    assert payload["state_hint"] == "STRATEGY_LEDGER_APPENDED"
    assert payload["artifacts"]["strategy_ledger_path"] == str(ledger_path)
    assert payload["metrics"]["strategy_id"] == "fno_spectral_filter_v0"
    assert payload["metrics"]["decision"] == "promote"
    assert payload["metrics"]["promote_to_skill_candidate"] is True
    assert payload["next_hint"] == "create-skill-candidate"


def test_create_skill_candidate_writes_auditable_artifact(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import create_skill_candidate

    proposal_path, rubric_path, evidence_path, promotion_path = (
        _write_promoted_strategy_artifacts(tmp_path)
    )
    output_path = tmp_path / "skill_candidate.json"

    candidate = create_skill_candidate(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        output_path=output_path,
        skill_name="pde-fno-spectral-filter",
        skill_dir="skills/pde-fno-spectral-filter",
    )

    assert output_path.exists()
    assert candidate == json.loads(output_path.read_text(encoding="utf-8"))
    assert candidate["schema_version"] == 1
    assert candidate["state_hint"] == "SKILL_CANDIDATE_CREATED"
    assert candidate["strategy_id"] == "fno_spectral_filter_v0"
    assert candidate["skill_name"] == "pde-fno-spectral-filter"
    assert candidate["suggested_skill_dir"] == "skills/pde-fno-spectral-filter"
    assert candidate["evidence_sources"]["promotion_report_path"] == str(promotion_path)
    assert candidate["promotion_reason_codes"] == ["all_thresholds_passed"]
    assert "positive mean score delta across windows" in candidate["trigger_conditions"]
    assert "Apply a low-pass residual filter" in candidate["procedure_steps"][0]
    assert "over-smoothing shocks" in candidate["failure_boundaries"]
    assert candidate["next_command"] == "review-skill-candidate"


def test_create_skill_candidate_rejects_unpromoted_report(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import (
        StrategyGrowthError,
        build_strategy_rubric,
        create_skill_candidate,
        evaluate_strategy_promotion,
        record_strategy_proposal,
    )

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "experiment_rubric.json"
    evidence_path = tmp_path / "evidence_summary.json"
    promotion_path = tmp_path / "promotion_report.json"
    record_strategy_proposal(**_proposal_kwargs(proposal_path))
    build_strategy_rubric(
        proposal_path=proposal_path,
        output_path=rubric_path,
        min_evaluation_windows=3,
        max_inference_time_regression=0.10,
    )
    evidence_path.write_text(
        json.dumps(
            {
                "evaluation_window_count": 1,
                "mean_score_delta": 2.5,
                "score_delta_std": 0.8,
                "long_window_score_delta": 1.1,
                "physics_residual_delta": -0.05,
                "inference_time_regression": 0.02,
                "failure_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    evaluate_strategy_promotion(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        output_path=promotion_path,
    )

    with pytest.raises(StrategyGrowthError, match="promotion report did not promote"):
        create_skill_candidate(
            proposal_path=proposal_path,
            rubric_path=rubric_path,
            evidence_summary_path=evidence_path,
            promotion_report_path=promotion_path,
            output_path=tmp_path / "skill_candidate.json",
            skill_name="pde-fno-spectral-filter",
            skill_dir="skills/pde-fno-spectral-filter",
        )


def test_create_skill_candidate_cli_returns_protocol_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    proposal_path, rubric_path, evidence_path, promotion_path = (
        _write_promoted_strategy_artifacts(tmp_path)
    )
    output_path = tmp_path / "skill_candidate.json"

    result = CliRunner().invoke(
        app,
        [
            "create-skill-candidate",
            "--proposal-path",
            str(proposal_path),
            "--rubric-path",
            str(rubric_path),
            "--evidence-summary-path",
            str(evidence_path),
            "--promotion-report-path",
            str(promotion_path),
            "--output-path",
            str(output_path),
            "--skill-name",
            "pde-fno-spectral-filter",
            "--skill-dir",
            "skills/pde-fno-spectral-filter",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "create-skill-candidate"
    assert payload["state_hint"] == "SKILL_CANDIDATE_CREATED"
    assert payload["artifacts"]["skill_candidate_path"] == str(output_path)
    assert payload["metrics"]["strategy_id"] == "fno_spectral_filter_v0"
    assert payload["metrics"]["skill_name"] == "pde-fno-spectral-filter"
    assert payload["metrics"]["procedure_step_count"] >= 1
    assert payload["next_hint"] == "review-skill-candidate"
