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
    evidence_path = tmp_path / "evidence_summary.json"
    promotion_path = tmp_path / "promotion_report.json"
    evidence_path.write_text(
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
    evidence_path = tmp_path / "evidence_summary.json"
    promotion_path = tmp_path / "promotion_report.json"
    evidence_path.write_text(
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
