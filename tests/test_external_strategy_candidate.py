import json
from pathlib import Path

import pytest


def _write_retrieval_decision(path: Path, *, decision: str = "retrieve_external_strategy") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_NEXT_ACTION_RECOMMENDED",
                "model_family": "DeepONet",
                "best_strategy_id": "supervised_h8",
                "decision": decision,
                "reason_codes": [
                    "negative_mean_score_delta",
                    "negative_long_window_score_delta",
                ],
                "recommended_source_type": "external_retrieval"
                if decision == "retrieve_external_strategy"
                else "local_library",
                "metrics": {
                    "mean_score_delta": -0.6,
                    "long_window_score_delta": -0.3,
                    "failure_rate": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_formalize_external_strategy_candidate_writes_reusable_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.external_strategy_candidate import (
        formalize_external_strategy_candidate,
    )

    decision_path = _write_retrieval_decision(tmp_path / "strategy_decision.json")
    output_path = tmp_path / "retrieved_strategy_candidate.json"

    report = formalize_external_strategy_candidate(
        strategy_decision_path=decision_path,
        output_path=output_path,
        strategy_id="burgers_spectral_viscosity_retrieved_v0",
        title="Spectral viscosity stabilization for Burgers rollout",
        source_title="Local retrieval seed: spectral viscosity",
        source_url="local://strategy-library/spectral-viscosity",
        scientific_claim=(
            "Adding a small high-frequency viscosity term can damp unstable "
            "Burgers rollout modes without over-smoothing the resolved solution."
        ),
        mechanism=(
            "Penalize or filter high-wavenumber residual components during "
            "autoregressive rollout training."
        ),
        applicable_pde="1D viscous Burgers",
        targeted_failure_modes=["long_horizon_drift", "high_frequency_instability"],
        implementation_plan={
            "model_family": "DeepONet",
            "intervention": "spectral_viscosity_loss",
            "allowed_commands": ["run-deeponet-strategy-sweep"],
        },
        expected_observables=[
            "lower late-window RMSE",
            "reduced high-frequency residual energy",
        ],
        known_risks=["over-smoothing shocks"],
    )

    assert output_path.exists()
    assert report == json.loads(output_path.read_text(encoding="utf-8"))
    assert report["state_hint"] == "EXTERNAL_STRATEGY_CANDIDATE_FORMALIZED"
    assert report["source_type"] == "external_retrieval"
    assert report["strategy_decision_path"] == str(decision_path)
    assert report["retrieval_source"] == {
        "title": "Local retrieval seed: spectral viscosity",
        "url": "local://strategy-library/spectral-viscosity",
    }
    assert report["record_strategy_proposal_fields"]["strategy_id"] == (
        "burgers_spectral_viscosity_retrieved_v0"
    )
    assert report["record_strategy_proposal_fields"]["source_type"] == (
        "external_retrieval"
    )
    assert report["next_command"] == "record-strategy-proposal"


def test_formalize_external_strategy_candidate_requires_retrieval_decision(
    tmp_path: Path,
) -> None:
    from pde_agent.core.external_strategy_candidate import (
        ExternalStrategyCandidateError,
        formalize_external_strategy_candidate,
    )

    decision_path = _write_retrieval_decision(
        tmp_path / "strategy_decision.json",
        decision="evaluate_promotion",
    )

    with pytest.raises(ExternalStrategyCandidateError, match="retrieve_external_strategy"):
        formalize_external_strategy_candidate(
            strategy_decision_path=decision_path,
            output_path=tmp_path / "retrieved_strategy_candidate.json",
            strategy_id="bad",
            title="Bad",
            source_title="Bad source",
            source_url="local://bad",
            scientific_claim="Bad claim",
            mechanism="Bad mechanism",
            applicable_pde="1D Burgers",
            targeted_failure_modes=["long_horizon_drift"],
            implementation_plan={"model_family": "DeepONet"},
            expected_observables=["improves score"],
            known_risks=["unknown"],
        )


def test_formalize_external_strategy_candidate_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    decision_path = _write_retrieval_decision(tmp_path / "strategy_decision.json")
    output_path = tmp_path / "retrieved_strategy_candidate.json"

    result = CliRunner().invoke(
        app,
        [
            "formalize-external-strategy-candidate",
            "--strategy-decision-path",
            str(decision_path),
            "--output-path",
            str(output_path),
            "--strategy-id",
            "burgers_spectral_viscosity_retrieved_v0",
            "--title",
            "Spectral viscosity stabilization for Burgers rollout",
            "--source-title",
            "Local retrieval seed: spectral viscosity",
            "--source-url",
            "local://strategy-library/spectral-viscosity",
            "--scientific-claim",
            "Small high-frequency viscosity can stabilize Burgers rollouts.",
            "--mechanism",
            "Damp high-wavenumber residual components during rollout training.",
            "--applicable-pde",
            "1D viscous Burgers",
            "--targeted-failure-mode",
            "long_horizon_drift",
            "--targeted-failure-mode",
            "high_frequency_instability",
            "--implementation-plan-json",
            '{"model_family":"DeepONet","intervention":"spectral_viscosity_loss"}',
            "--expected-observable",
            "lower late-window RMSE",
            "--known-risk",
            "over-smoothing shocks",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "formalize-external-strategy-candidate"
    assert payload["state_hint"] == "EXTERNAL_STRATEGY_CANDIDATE_FORMALIZED"
    assert payload["artifacts"]["retrieved_strategy_candidate_path"] == str(output_path)
    assert payload["metrics"]["strategy_id"] == (
        "burgers_spectral_viscosity_retrieved_v0"
    )
    assert payload["metrics"]["targeted_failure_mode_count"] == 2
    assert payload["next_hint"] == "record-strategy-proposal"
