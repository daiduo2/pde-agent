import json
from pathlib import Path


def _write_operator_sweep_summary(
    path: Path,
    *,
    state_hint: str = "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
    forecast_steps: int = 31,
    winner_deltas: list[float] | None = None,
    records: list[dict[str, object]] | None = None,
) -> Path:
    deltas = winner_deltas or [1.5, 0.5, 2.5]
    comparison_winners = []
    for index, delta in enumerate(deltas):
        winner_record = (
            records[index] if records is not None and index < len(records) else {}
        )
        comparison_winners.append(
            {
                "start_index": index,
                "winner_experiment_id": winner_record.get(
                    "experiment_id",
                    f"deeponet_candidate_{index:03d}",
                ),
                "winner_training_strategy": winner_record.get(
                    "training_strategy",
                    "branch_trunk_rollout",
                ),
                "winner_score": winner_record.get("score", 30.0 + delta),
                "winner_score_delta": delta,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": state_hint,
                "inputs": {"forecast_steps": forecast_steps},
                "comparison_winners": comparison_winners,
                "comparison_count": len(comparison_winners),
                "record_count": (
                    len(records) if records is not None else len(comparison_winners) * 2
                ),
                "records": records or [],
                "physics_residual_delta": -0.1,
                "inference_time_regression": 0.05,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_convert_operator_evidence_summary_supports_deeponet_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json"
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=output_path,
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert output_path.exists()
    assert evidence == json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["model_family"] == "DeepONet"
    assert evidence["source_state_hint"] == "DEEPONET_EVIDENCE_SWEEP_COMPLETED"
    assert evidence["source_evidence_summary_path"] == str(sweep_path)
    assert evidence["source_deeponet_evidence_summary_path"] == str(sweep_path)
    assert evidence["evaluation_window_count"] == 3
    assert evidence["mean_score_delta"] == 1.5
    assert evidence["long_window_score_delta"] == 1.5
    assert evidence["physics_residual_delta"] == -0.1
    assert evidence["inference_time_regression"] == 0.05
    assert evidence["failure_rate"] == 0.0
    assert evidence["next_command"] == "evaluate-strategy-promotion"


def test_convert_operator_evidence_summary_supports_unet_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "unet_evidence_summary.json",
        state_hint="UNET_EVIDENCE_SWEEP_COMPLETED",
        records=[
            {
                "start_index": 0,
                "train_steps": 4,
                "experiment_id": "unet_a",
                "training_strategy": "unet_spectral",
                "strategy_label": "spectral_c8_d2_lr1e2_seed5",
                "base_channels": 8,
                "depth": 2,
                "score": 42.0,
                "score_delta": 2.0,
                "rel_mse": 0.2,
                "rmse": 0.3,
                "dominant_error_pattern": "long_horizon_drift",
                "physics_issue": "inconclusive",
            }
        ],
        winner_deltas=[2.0],
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=output_path,
        model_family="UNet",
        source_state_hint="UNET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["model_family"] == "UNet"
    assert evidence["source_unet_evidence_summary_path"] == str(sweep_path)
    assert evidence["candidate_records"][0]["hyperparameters"] == {
        "base_channels": 8,
        "depth": 2,
    }
    assert evidence["best_strategy_id"] == "spectral_c8_d2_lr1e2_seed5"


def test_convert_operator_evidence_summary_cli_returns_protocol_payload(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json"
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    result = CliRunner().invoke(
        app,
        [
            "convert-operator-evidence-summary",
            "--sweep-summary-path",
            str(sweep_path),
            "--output-path",
            str(output_path),
            "--model-family",
            "DeepONet",
            "--source-state-hint",
            "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
            "--long-window-threshold",
            "31",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "convert-operator-evidence-summary"
    assert payload["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert payload["artifacts"]["strategy_evidence_summary_path"] == str(output_path)
    assert payload["artifacts"]["source_evidence_summary_path"] == str(sweep_path)
    assert payload["metrics"]["model_family"] == "DeepONet"
    assert payload["metrics"]["evaluation_window_count"] == 3
    assert payload["metrics"]["mean_score_delta"] == 1.5
    assert payload["metrics"]["failure_rate"] == 0.0
    assert payload["next_hint"] == "evaluate-strategy-promotion"


def test_convert_operator_evidence_summary_cli_reports_normalized_ranking_metrics(
    tmp_path: Path,
) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json",
        winner_deltas=[2.0],
        records=[
            {
                "start_index": 0,
                "train_steps": 4,
                "experiment_id": "deeponet_a",
                "training_strategy": "branch_trunk_spectral",
                "strategy_label": "spectral_sw5e2_h8",
                "score": 42.0,
                "score_delta": 2.0,
                "rel_mse": 0.2,
                "rmse": 0.3,
                "dominant_error_pattern": "long_horizon_drift",
                "physics_issue": "burgers_residual_high",
            }
        ],
    )
    output_path = tmp_path / "strategy_evidence_summary.json"

    result = CliRunner().invoke(
        app,
        [
            "convert-operator-evidence-summary",
            "--sweep-summary-path",
            str(sweep_path),
            "--output-path",
            str(output_path),
            "--model-family",
            "DeepONet",
            "--source-state-hint",
            "DEEPONET_EVIDENCE_SWEEP_COMPLETED",
            "--long-window-threshold",
            "31",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["candidate_record_count"] == 1
    assert payload["metrics"]["strategy_ranking_count"] == 1
    assert payload["metrics"]["best_strategy_id"] == "spectral_sw5e2_h8"


def test_convert_operator_evidence_summary_normalizes_candidate_records_and_strategy_ranking(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    records: list[dict[str, object]] = [
        {
            "start_index": 0,
            "train_steps": 4,
            "experiment_id": "deeponet_a",
            "training_strategy": "branch_trunk_spectral",
            "strategy_label": "spectral_sw5e2_h8",
            "hidden_width": 8,
            "latent_dim": 4,
            "score": 42.0,
            "score_delta": 2.0,
            "rel_mse": 0.2,
            "rmse": 0.3,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "burgers_residual_high",
        },
        {
            "start_index": 1,
            "train_steps": 4,
            "experiment_id": "deeponet_b",
            "training_strategy": "branch_trunk_spectral",
            "strategy_label": "spectral_sw5e2_h8",
            "hidden_width": 8,
            "latent_dim": 4,
            "score": 41.0,
            "score_delta": -1.0,
            "rel_mse": 0.25,
            "rmse": 0.35,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "burgers_residual_high",
        },
        {
            "start_index": 0,
            "train_steps": 4,
            "experiment_id": "deeponet_c",
            "training_strategy": "branch_trunk_supervised",
            "strategy_label": "supervised_h8",
            "hidden_width": 8,
            "latent_dim": 4,
            "score": 39.0,
            "score_delta": 0.5,
            "rel_mse": 0.4,
            "rmse": 0.6,
            "dominant_error_pattern": "phase_shift",
            "physics_issue": "inconclusive",
        },
    ]
    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json",
        winner_deltas=[2.0, -1.0],
        records=records,
    )

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["operator_evidence_schema"] == {
        "schema_version": 1,
        "record_key": "model_family:strategy_id:experiment_id:start_index:train_steps",
        "ranking_order": [
            "win_count_desc",
            "mean_score_delta_desc",
            "mean_score_desc",
            "strategy_id_asc",
        ],
    }
    assert evidence["candidate_records"][0] == {
        "model_family": "DeepONet",
        "strategy_id": "spectral_sw5e2_h8",
        "training_strategy": "branch_trunk_spectral",
        "experiment_id": "deeponet_a",
        "start_index": 0,
        "train_steps": 4,
        "score": 42.0,
        "score_delta": 2.0,
        "rel_mse": 0.2,
        "rmse": 0.3,
        "dominant_error_pattern": "long_horizon_drift",
        "physics_issue": "burgers_residual_high",
        "hyperparameters": {
            "hidden_width": 8,
            "latent_dim": 4,
        },
    }
    assert evidence["strategy_ranking"] == [
        {
            "rank": 1,
            "strategy_id": "spectral_sw5e2_h8",
            "training_strategy": "branch_trunk_spectral",
            "record_count": 2,
            "win_count": 2,
            "win_rate": 1.0,
            "mean_score": 41.5,
            "mean_score_delta": 0.5,
            "best_score": 42.0,
            "failure_rate": 0.5,
            "dominant_error_patterns": ["long_horizon_drift"],
            "physics_issues": ["burgers_residual_high"],
        },
        {
            "rank": 2,
            "strategy_id": "supervised_h8",
            "training_strategy": "branch_trunk_supervised",
            "record_count": 1,
            "win_count": 0,
            "win_rate": 0.0,
            "mean_score": 39.0,
            "mean_score_delta": 0.5,
            "best_score": 39.0,
            "failure_rate": 0.0,
            "dominant_error_patterns": ["phase_shift"],
            "physics_issues": ["inconclusive"],
        },
    ]
    assert evidence["best_strategy_id"] == "spectral_sw5e2_h8"


def test_convert_operator_evidence_summary_normalizes_fno_records_without_strategy_label(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    records: list[dict[str, object]] = [
        {
            "start_index": 0,
            "train_steps": 2,
            "experiment_id": "fno_a",
            "training_strategy": "scheduled_sampling",
            "rollout_steps": 10,
            "teacher_forcing_decay": 0.95,
            "score": 44.0,
            "score_delta": 4.0,
            "rel_mse": 0.1,
            "rmse": 0.2,
            "dominant_error_pattern": "amplitude_damping",
            "physics_issue": "inconclusive",
        }
    ]
    sweep_path = _write_operator_sweep_summary(
        tmp_path / "fno_evidence_summary.json",
        state_hint="FNO_EVIDENCE_SWEEP_COMPLETED",
        winner_deltas=[4.0],
        records=records,
    )

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        model_family="FNO",
        source_state_hint="FNO_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["candidate_records"] == [
        {
            "model_family": "FNO",
            "strategy_id": "scheduled_sampling",
            "training_strategy": "scheduled_sampling",
            "experiment_id": "fno_a",
            "start_index": 0,
            "train_steps": 2,
            "score": 44.0,
            "score_delta": 4.0,
            "rel_mse": 0.1,
            "rmse": 0.2,
            "dominant_error_pattern": "amplitude_damping",
            "physics_issue": "inconclusive",
            "hyperparameters": {
                "rollout_steps": 10,
                "teacher_forcing_decay": 0.95,
            },
        }
    ]
    assert evidence["strategy_ranking"][0]["strategy_id"] == "scheduled_sampling"
    assert evidence["strategy_ranking"][0]["win_count"] == 1


def test_convert_operator_evidence_summary_preserves_pi_physics_metrics(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    records: list[dict[str, object]] = [
        {
            "start_index": 0,
            "train_steps": 4,
            "experiment_id": "deeponet_physics_a",
            "training_strategy": "branch_trunk_physics_residual",
            "strategy_label": "physics_residual_h8",
            "score": 43.0,
            "score_delta": 3.0,
            "rel_mse": 0.12,
            "rmse": 0.22,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "residual_mismatch",
            "residual_rmse": 0.08,
            "mass_drift": 0.01,
            "energy_growth_ratio": 1.02,
            "high_frequency_energy_ratio": 0.14,
        },
        {
            "start_index": 1,
            "train_steps": 4,
            "experiment_id": "deeponet_physics_b",
            "training_strategy": "branch_trunk_physics_residual",
            "strategy_label": "physics_residual_h8",
            "score": 44.0,
            "score_delta": 2.0,
            "rel_mse": 0.1,
            "rmse": 0.2,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "stable_low_residual",
            "residual_rmse": 0.04,
            "mass_drift": 0.02,
            "energy_growth_ratio": 1.01,
            "high_frequency_energy_ratio": 0.2,
        },
    ]
    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json",
        winner_deltas=[3.0, 2.0],
        records=records,
    )

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["candidate_records"][0]["physics_metrics"] == {
        "energy_growth_ratio": 1.02,
        "high_frequency_energy_ratio": 0.14,
        "mass_drift": 0.01,
        "residual_rmse": 0.08,
    }
    assert "residual_rmse" not in evidence["candidate_records"][0]["hyperparameters"]
    assert evidence["physics_metric_summary"] == {
        "record_count": 2,
        "mean_residual_rmse": 0.06,
        "max_high_frequency_energy_ratio": 0.2,
        "max_mass_drift": 0.02,
        "max_energy_growth_ratio": 1.02,
    }
    assert evidence["strategy_ranking"][0]["mean_residual_rmse"] == 0.06
    assert evidence["strategy_ranking"][0]["max_high_frequency_energy_ratio"] == 0.2


def test_convert_operator_evidence_summary_tolerates_partial_pi_metrics(
    tmp_path: Path,
) -> None:
    from pde_agent.core.operator_strategy_evidence import (
        convert_operator_evidence_summary,
    )

    records: list[dict[str, object]] = [
        {
            "start_index": 0,
            "train_steps": 4,
            "experiment_id": "deeponet_partial_physics",
            "training_strategy": "branch_trunk_physics_residual",
            "score": 43.0,
            "score_delta": 3.0,
            "rel_mse": 0.12,
            "rmse": 0.22,
            "dominant_error_pattern": "long_horizon_drift",
            "physics_issue": "residual_mismatch",
            "residual_rmse": 0.08,
        }
    ]
    sweep_path = _write_operator_sweep_summary(
        tmp_path / "deeponet_evidence_summary.json",
        winner_deltas=[3.0],
        records=records,
    )

    evidence = convert_operator_evidence_summary(
        sweep_summary_path=sweep_path,
        output_path=tmp_path / "strategy_evidence_summary.json",
        model_family="DeepONet",
        source_state_hint="DEEPONET_EVIDENCE_SWEEP_COMPLETED",
        long_window_threshold=31,
    )

    assert evidence["physics_metric_summary"] == {
        "record_count": 1,
        "mean_residual_rmse": 0.08,
        "max_high_frequency_energy_ratio": None,
        "max_mass_drift": None,
        "max_energy_growth_ratio": None,
    }
