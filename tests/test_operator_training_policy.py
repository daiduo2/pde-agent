from pde_agent.core.operator_training_policy import build_operator_training_decision


def test_operator_training_policy_detects_undertraining() -> None:
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 20.0,
                "early_rel_mse": 1.8,
                "final_train_loss": 2.0,
                "final_val_loss": 2.1,
                "loss_curve": {"train": [2.6, 2.0], "val": [2.5, 2.1]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 4,
                    "modes": 8,
                    "width": 32,
                    "depth": 1,
                    "normalization": "none",
                },
            }
        ],
        previous_memory=None,
    )

    assert decision["state_hint"] == "OPERATOR_TRAINING_DECISION_BUILT"
    codes = {item["code"] for item in decision["diagnosis"]}
    assert "undertrained" in codes
    policy = decision["policy"]
    assert policy["next_action"] == "increase_training_budget"
    assert "train_steps" in policy["required_axis_changes"]
    assert policy["budget_adjustments"]["train_steps_delta"] > 0


def test_operator_training_policy_detects_loss_score_mismatch() -> None:
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 20.0,
                "early_rel_mse": 1.8,
                "loss_curve": {"train": [2.5, 1.5], "val": [2.2, 2.3]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "mse",
                    "forecast_steps": 12,
                    "train_steps": 4,
                    "modes": 8,
                    "width": 32,
                    "depth": 1,
                    "normalization": "none",
                },
            }
        ],
        previous_memory=None,
    )

    codes = {item["code"] for item in decision["diagnosis"]}
    assert "loss_score_mismatch" in codes
    assert "loss_type" in decision["policy"]["required_axis_changes"]
    assert "early_weighted_mse" in decision["policy"]["preferred_loss_types"]


def test_operator_training_policy_detects_stagnation_from_previous_memory() -> None:
    previous = {
        "convergence_history": {
            "best_score_history": [40.0, 39.5],
            "no_improvement_count": 2,
        }
    }
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 39.0,
                "early_rel_mse": 1.2,
                "loss_curve": {"train": [1.2, 1.1], "val": [1.3, 1.25]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 8,
                    "modes": 8,
                    "width": 32,
                    "depth": 1,
                    "normalization": "none",
                },
            }
        ],
        previous_memory=previous,
    )

    codes = {item["code"] for item in decision["diagnosis"]}
    assert "stagnation" in codes
    assert decision["policy"]["next_action"] == "expand_search_radius"
    assert decision["policy"]["hypothesis_lab_requests"]
    assert decision["policy"]["preferred_model_families"][:2] == [
        "fno_ar1",
        "conv1d_residual",
    ]
    assert "model_family" in decision["policy"]["required_axis_changes"]


def test_operator_training_policy_detects_capacity_limit() -> None:
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 52.0,
                "early_rel_mse": 0.72,
                "loss_curve": {"train": [0.9, 0.88], "val": [0.95, 0.94]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 80,
                    "modes": 8,
                    "width": 32,
                    "depth": 1,
                    "normalization": "none",
                },
            }
        ],
        previous_memory=None,
    )

    codes = {item["code"] for item in decision["diagnosis"]}
    assert "capacity_limited" in codes
    assert decision["policy"]["next_action"] == "increase_capacity"
    assert "capacity" in decision["policy"]["required_axis_changes"]
    assert decision["policy"]["budget_adjustments"]["width_delta"] > 0


def test_operator_training_policy_prioritizes_new_dynamics_for_severe_early_gap() -> None:
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 5.0,
                "early_rel_mse": 2.8,
                "loss_curve": {"train": [2.0, 1.95], "val": [2.2, 2.18]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "early_weighted_mse",
                    "forecast_steps": 12,
                    "train_steps": 50,
                    "modes": 12,
                    "width": 48,
                    "depth": 2,
                    "normalization": "sample_standard",
                },
            }
        ],
        previous_memory=None,
    )

    codes = {item["code"] for item in decision["diagnosis"]}
    assert "short_horizon_dynamics_gap" in codes
    assert decision["policy"]["next_action"] == "request_new_dynamics_source"
    assert "model_family" in decision["policy"]["required_axis_changes"]
    assert decision["policy"]["hypothesis_lab_requests"][0]["request_type"] == (
        "new_dynamics_source"
    )


def test_operator_training_policy_repairs_score_regression() -> None:
    previous = {"best_so_far": {"score": 50.0}}
    decision = build_operator_training_decision(
        trajectory=[
            {
                "candidate_id": "c1",
                "score": 47.0,
                "early_rel_mse": 0.8,
                "loss_curve": {"train": [1.0, 0.9], "val": [1.1, 1.0]},
                "config": {
                    "model_family": "fno_lite",
                    "loss_type": "horizon_weighted_mse",
                    "forecast_steps": 47,
                    "train_steps": 120,
                    "modes": 16,
                    "width": 64,
                    "depth": 2,
                    "normalization": "sample_standard",
                },
            }
        ],
        previous_memory=previous,
    )

    codes = {item["code"] for item in decision["diagnosis"]}
    assert "score_regression" in codes
    assert decision["policy"]["next_action"] == "repair_regression"
    assert "capacity" in decision["policy"]["required_axis_changes"]
