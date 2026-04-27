import json
from pathlib import Path


def _write_evidence(
    path: Path,
    *,
    model_family: str = "DeepONet",
    evaluation_window_count: int = 3,
    mean_score_delta: float = 1.2,
    long_window_score_delta: float = 0.8,
    failure_rate: float = 0.0,
    best_strategy_id: str = "spectral_sw5e2_h8",
    strategy_ranking: list[dict[str, object]] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_EVIDENCE_SUMMARY_BUILT",
                "model_family": model_family,
                "evaluation_window_count": evaluation_window_count,
                "mean_score_delta": mean_score_delta,
                "score_delta_std": 0.4,
                "long_window_score_delta": long_window_score_delta,
                "physics_residual_delta": -0.02,
                "inference_time_regression": 0.01,
                "failure_rate": failure_rate,
                "best_strategy_id": best_strategy_id,
                "strategy_ranking": strategy_ranking
                if strategy_ranking is not None
                else [
                    {
                        "rank": 1,
                        "strategy_id": best_strategy_id,
                        "training_strategy": "branch_trunk_spectral",
                        "record_count": evaluation_window_count,
                        "win_count": evaluation_window_count,
                        "win_rate": 1.0,
                        "mean_score": 42.0,
                        "mean_score_delta": mean_score_delta,
                        "best_score": 43.0,
                        "failure_rate": failure_rate,
                        "dominant_error_patterns": ["long_horizon_drift"],
                        "physics_issues": ["inconclusive"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_recommend_strategy_next_action_sends_strong_evidence_to_promotion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_decision import recommend_strategy_next_action

    evidence_path = _write_evidence(tmp_path / "strategy_evidence_summary.json")
    output_path = tmp_path / "strategy_decision.json"

    report = recommend_strategy_next_action(
        evidence_summary_path=evidence_path,
        output_path=output_path,
        min_evaluation_windows=3,
    )

    assert output_path.exists()
    assert report == json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["state_hint"] == "STRATEGY_NEXT_ACTION_RECOMMENDED"
    assert report["decision"] == "evaluate_promotion"
    assert report["reason_codes"] == ["strong_multi_window_evidence"]
    assert report["model_family"] == "DeepONet"
    assert report["best_strategy_id"] == "spectral_sw5e2_h8"
    assert report["next_command"] == "evaluate-strategy-promotion"


def test_recommend_strategy_next_action_expands_evidence_when_windows_are_insufficient(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_decision import recommend_strategy_next_action

    evidence_path = _write_evidence(
        tmp_path / "strategy_evidence_summary.json",
        model_family="FNO",
        evaluation_window_count=1,
        mean_score_delta=3.0,
        long_window_score_delta=0.0,
        best_strategy_id="scheduled_sampling",
    )

    report = recommend_strategy_next_action(
        evidence_summary_path=evidence_path,
        output_path=tmp_path / "strategy_decision.json",
        min_evaluation_windows=3,
    )

    assert report["decision"] == "expand_evidence"
    assert report["reason_codes"] == ["insufficient_evaluation_windows"]
    assert report["next_command"] == "run-fno-evidence-sweep"
    assert report["suggested_actions"] == [
        "add evaluation windows before promotion",
        "keep the current best strategy as a control arm",
    ]


def test_recommend_strategy_next_action_requests_external_strategy_after_failed_sweep(
    tmp_path: Path,
) -> None:
    from pde_agent.core.strategy_decision import recommend_strategy_next_action

    evidence_path = _write_evidence(
        tmp_path / "strategy_evidence_summary.json",
        evaluation_window_count=4,
        mean_score_delta=-0.6,
        long_window_score_delta=-0.3,
        failure_rate=0.75,
        best_strategy_id="supervised_h8",
    )

    report = recommend_strategy_next_action(
        evidence_summary_path=evidence_path,
        output_path=tmp_path / "strategy_decision.json",
        min_evaluation_windows=3,
    )

    assert report["decision"] == "retrieve_external_strategy"
    assert report["reason_codes"] == [
        "negative_mean_score_delta",
        "negative_long_window_score_delta",
        "high_failure_rate",
    ]
    assert report["next_command"] == "record-strategy-proposal"
    assert report["recommended_source_type"] == "external_retrieval"


def test_recommend_strategy_next_action_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    evidence_path = _write_evidence(tmp_path / "strategy_evidence_summary.json")
    output_path = tmp_path / "strategy_decision.json"

    result = CliRunner().invoke(
        app,
        [
            "recommend-strategy-next-action",
            "--evidence-summary-path",
            str(evidence_path),
            "--output-path",
            str(output_path),
            "--min-evaluation-windows",
            "3",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "recommend-strategy-next-action"
    assert payload["state_hint"] == "STRATEGY_NEXT_ACTION_RECOMMENDED"
    assert payload["artifacts"]["strategy_decision_path"] == str(output_path)
    assert payload["metrics"]["decision"] == "evaluate_promotion"
    assert payload["metrics"]["best_strategy_id"] == "spectral_sw5e2_h8"
    assert payload["next_hint"] == "evaluate-strategy-promotion"
