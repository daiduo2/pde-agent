from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def _manual_report(path: Path) -> Path:
    payload = {
        "state_hint": "MANUAL_OPERATOR_TUNING_COMPLETE",
        "best_candidate": {
            "candidate_id": "linear_trend_spectral_w5_g-0.1_d60_s1",
            "family": "linear_trend_spectral",
            "params": {
                "slope_window": 5,
                "gain": -0.1,
                "decay": 60.0,
                "spectral_strength": 1.0,
            },
            "segment_score": 14.94,
            "task1_total_fast_proxy": 86.19,
            "segments": {"late": {"rmse": 0.234}},
        },
        "top_candidates": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manual_report_with_external_best(path: Path) -> Path:
    payload = {
        "state_hint": "MANUAL_OPERATOR_TUNING_COMPLETE",
        "best_candidate": {
            "candidate_id": "blend_persistence_unet_fast_w0.2",
            "family": "external_blend",
            "params": {"external": "unet_fast", "weight_right": 0.2},
            "segment_score": 14.96,
            "task1_total_fast_proxy": 86.21,
        },
        "top_candidates": [
            {
                "candidate_id": "blend_persistence_unet_fast_w0.2",
                "family": "external_blend",
                "params": {"external": "unet_fast", "weight_right": 0.2},
                "segment_score": 14.96,
                "task1_total_fast_proxy": 86.21,
            },
            {
                "candidate_id": "linear_trend_spectral_w5_g-0.1_d60_s1",
                "family": "linear_trend_spectral",
                "params": {
                    "slope_window": 5,
                    "gain": -0.1,
                    "decay": 60.0,
                    "spectral_strength": 1.0,
                },
                "segment_score": 14.94,
                "task1_total_fast_proxy": 86.19,
                "segments": {"late": {"rmse": 0.234}},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _adaptive_plateau_report(path: Path) -> Path:
    payload = {
        "state_hint": "HERMES_ADAPTIVE_HEARTBEAT_RAN",
        "stage_count": 2,
        "policy_events": [
            {
                "stage_index": 1,
                "action": "switch_to_verified_blueprint",
                "trajectory_state": "plateau",
                "reason_codes": ["stagnation_detected"],
                "best_candidate_id": "initial",
                "best_score": 85.61,
            },
            {
                "stage_index": 2,
                "action": "verified_blueprint_plateau",
                "trajectory_state": "plateau",
                "reason_codes": ["recent_batches_not_improved"],
                "best_candidate_id": "linear_trend_spectral_w5_g-0.075_d70_s1",
                "best_score": 86.21,
            },
        ],
        "stages": [
            {
                "stage_index": 2,
                "candidate_generator": "linear_trend_spectral",
                "best_candidate_id": "linear_trend_spectral_w5_g-0.075_d70_s1",
                "best_score": 86.21,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _adaptive_plateau_escape_report(path: Path) -> Path:
    payload = {
        "state_hint": "HERMES_ADAPTIVE_HEARTBEAT_RAN",
        "stage_count": 2,
        "policy_events": [
            {
                "stage_index": 2,
                "action": "verified_blueprint_plateau",
                "trajectory_state": "plateau",
                "reason_codes": ["stagnation_detected"],
                "best_candidate_id": "linear_trend_spectral_w5_g-0.075_d70_s1.5",
                "best_score": 86.24,
            }
        ],
        "stages": [
            {
                "stage_index": 2,
                "candidate_generator": "linear_trend_spectral",
                "best_candidate_id": "linear_trend_spectral_w5_g-0.075_d70_s1.5",
                "best_score": 86.24,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _adaptive_retrieval_refinement_sandbox_report(path: Path) -> Path:
    payload = {
        "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
        "blueprint_id": "bp_task1_retrieval_refinement_local_search_001",
        "candidate_family": "task1_retrieval_refinement_local_search",
        "best_candidate": {
            "candidate_id": "task1_retrieval_refinement_k48_affine_eb20_pw0.6_s0.5",
            "family": "task1_retrieval_refinement_local_search",
            "params": {
                "k": 48,
                "mode": "affine",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.6,
                "spectral_strength": 0.5,
            },
            "segment_score": 47.4,
            "task1_total_score_proxy": 110.3,
        },
        "top_candidates": [
            {
                "candidate_id": "task1_retrieval_refinement_k52_affine_eb20_pw0.6_s0.5",
                "family": "task1_retrieval_refinement_local_search",
                "params": {
                    "k": 52,
                    "mode": "affine",
                    "early_blend_steps": 20,
                    "persistence_weight_start": 0.6,
                    "spectral_strength": 0.5,
                },
                "segment_score": 47.0,
                "task1_total_score_proxy": 109.5,
            }
        ],
        "verifier": {"decision": "promote_to_strategy_proposal"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _adaptive_retrieval_refinement_weight_trend_report(path: Path) -> Path:
    payload = {
        "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
        "blueprint_id": "bp_task1_retrieval_refinement_local_search_001",
        "candidate_family": "task1_retrieval_refinement_local_search",
        "best_candidate": {
            "candidate_id": "task1_retrieval_refinement_k48_affine_eb20_pw0.7_s0.5",
            "family": "task1_retrieval_refinement_local_search",
            "params": {
                "k": 48,
                "mode": "affine",
                "early_blend_steps": 20,
                "persistence_weight_start": 0.7,
                "spectral_strength": 0.5,
            },
            "segment_score": 47.7,
            "task1_total_score_proxy": 111.0,
        },
        "top_candidates": [
            {
                "candidate_id": "task1_retrieval_refinement_k48_affine_eb20_pw0.65_s0.5",
                "params": {
                    "k": 48,
                    "mode": "affine",
                    "early_blend_steps": 20,
                    "persistence_weight_start": 0.65,
                    "spectral_strength": 0.5,
                },
                "segment_score": 47.55,
                "task1_total_score_proxy": 110.8,
            },
            {
                "candidate_id": "task1_retrieval_refinement_k48_affine_eb20_pw0.6_s0.5",
                "params": {
                    "k": 48,
                    "mode": "affine",
                    "early_blend_steps": 20,
                    "persistence_weight_start": 0.6,
                    "spectral_strength": 0.5,
                },
                "segment_score": 47.4,
                "task1_total_score_proxy": 110.5,
            },
        ],
        "verifier": {"decision": "promote_to_strategy_proposal"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _scientific_log(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Hermes PDE Agent Scientific Log",
                "Trace:",
                "- Hermes diagnosed heartbeat trajectory reports before selecting the next generator.",
                "Experiment Tracking:",
                "- stage=1 action=switch_to_verified_blueprint state=plateau best=initial score=85.6089",
                "- stage=2 action=verified_blueprint_plateau state=plateau best=linear_trend_spectral_w5_g-0.075_d70_s1 score=86.2087",
                "Reflection:",
                "- Linear trend spectral improved over persistence but plateaued after verified blueprint reuse.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _late_ramp_scientific_log(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Hermes PDE Agent Scientific Log",
                "Trajectory Summary:",
                "- round=1 candidate=linear_trend_spectral_w5_g-0.075_d70_s1.5_ls2.25_lt60 score=86.33 effective_improvement=False",
                "Hermes Adaptive Execution Policy",
                "Experiment Tracking:",
                "- stage=2 action=verified_blueprint_plateau state=plateau best=initial score=86.34",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _literature_note(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Title: Residual correction for neural operator rollouts",
                "Claim: Horizon-conditioned residual correction can repair systematic rollout bias after a coarse operator prediction.",
                "Mechanism: Fit a lightweight residual map over persistence or checkpoint rollouts and apply shrinkage by forecast horizon.",
                "Hook: learned_residual_correction",
                "Observable: lower early and middle Rel-MSE without changing context steps",
                "Risk: residual correction can overfit validation windows",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _local_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": (
                    "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
                ),
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "DataInference",
                "hypothesis": (
                    "Damped trend plus spectral smoothing can improve a "
                    "persistence plateau."
                ),
                "mechanism": (
                    "Use context slope, damped extrapolation, and spectral "
                    "smoothing while preserving context frames."
                ),
                "executor_spec": {
                    "cli_command": "run-hypothesis-lab",
                    "implementation_hooks": [
                        "run-hypothesis-lab",
                        "linear_trend_spectral",
                    ],
                },
                "expected_observables": [
                    "higher local segment score than persistence plateau"
                ],
                "risk_controls": ["strong smoothing may damp shocks"],
                "source_grounding": {
                    "algorithm_card_id": (
                        "local_strategy_data_inference_linear_trend_spectral_v0"
                    ),
                    "source_ids": ["data_inference_linear_trend_spectral_v0"],
                },
            },
            {
                "algorithm_candidate_id": "candidate_local_checkpoint_burgers_fno",
                "candidate_family": "official_checkpoint_family",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "hypothesis": "Evaluate FNO checkpoint.",
                "mechanism": "Run checkpoint family sandbox.",
                "executor_spec": {
                    "cli_command": "run-checkpoint-family-sandbox",
                    "implementation_hooks": [],
                },
                "source_grounding": {
                    "algorithm_card_id": "local_checkpoint_burgers_fno",
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _retrieval_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": "candidate_local_strategy_task1_retrieval_prior_v0",
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "DataInference",
                "hypothesis": (
                    "Affine kNN retrieval over Task 1 validation trajectories can "
                    "escape the persistence plateau."
                ),
                "mechanism": (
                    "Match first 10 frames against a reference library and affine-map "
                    "the retrieved futures to query context statistics."
                ),
                "executor_spec": {
                    "cli_command": "predict-task1-retrieval-prior",
                    "implementation_hooks": [
                        "predict-task1-retrieval-prior",
                        "materialize-competition-candidate",
                    ],
                },
                "expected_observables": [
                    "leave-one-out segment score improves over persistence"
                ],
                "risk_controls": ["nearest-neighbor validation can overfit"],
                "source_grounding": {
                    "algorithm_card_id": "local_strategy_task1_retrieval_prior_v0",
                    "source_ids": ["task1_retrieval_prior_v0"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _retrieval_refinement_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": (
                    "candidate_local_strategy_task1_retrieval_refinement_v0"
                ),
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "DataInference",
                "hypothesis": (
                    "The validated Task 1 retrieval refinement should be reused "
                    "as the default local plateau escape strategy."
                ),
                "mechanism": (
                    "Run the bounded retrieval refinement shortlist around "
                    "k48 affine early-blend candidates."
                ),
                "executor_spec": {
                    "cli_command": "run-hypothesis-lab",
                    "implementation_hooks": [
                        "run-hypothesis-lab",
                        "task1_retrieval_refinement",
                    ],
                },
                "expected_observables": [
                    "Task 1 proxy remains above the previous retrieval prior"
                ],
                "risk_controls": ["validation retrieval can overfit hidden tests"],
                "source_grounding": {
                    "algorithm_card_id": "local_strategy_task1_retrieval_refinement_v0",
                    "source_ids": ["task1_retrieval_refinement_v0"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sem_burgers_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": (
                    "candidate_local_strategy_task1_sem_burgers_rollout_v0"
                ),
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "PhysicsSolver",
                "hypothesis": (
                    "A fixed-nu spectral-element style Burgers rollout can "
                    "improve Task 1 early and middle windows over retrieval-only "
                    "postprocessing."
                ),
                "mechanism": (
                    "Use the last context frame as the initial condition and "
                    "advance Burgers with element-local spectral derivatives."
                ),
                "executor_spec": {
                    "cli_command": "run-hypothesis-lab",
                    "implementation_hooks": ["task1_sem_burgers_rollout"],
                },
                "expected_observables": [
                    "early and middle Rel-MSE improve versus persistence",
                    "context prefix remains exact",
                ],
                "risk_controls": ["explicit SEM rollout can be unstable"],
                "source_grounding": {
                    "algorithm_card_id": "local_strategy_task1_sem_burgers_rollout_v0",
                    "source_ids": ["task1_sem_burgers_rollout_v0"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _physics_residual_refinement_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": (
                    "candidate_local_strategy_task1_physics_residual_refinement_v0"
                ),
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "PhysicsSolver",
                "hypothesis": (
                    "Burgers residual can guide a bounded correction of the "
                    "current retrieval trajectory without replacing the data prior."
                ),
                "mechanism": (
                    "Compute finite-difference Burgers residual on a base "
                    "retrieval prediction and nudge future frames toward one-step "
                    "PDE consistency."
                ),
                "executor_spec": {
                    "cli_command": "run-hypothesis-lab",
                    "implementation_hooks": ["task1_physics_residual_refinement"],
                },
                "expected_observables": [
                    "candidate residual RMSE decreases versus base prediction",
                    "context prefix remains exact",
                ],
                "risk_controls": ["physics nudging can hurt data-prior accuracy"],
                "source_grounding": {
                    "algorithm_card_id": (
                        "local_strategy_task1_physics_residual_refinement_v0"
                    ),
                    "source_ids": ["task1_physics_residual_refinement_v0"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pi_deeponet_correction_asset_algorithm_candidates(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
        "source_state_hint": "LOCAL_ASSET_ALGORITHM_CANDIDATES_BUILT",
        "algorithm_candidates": [
            {
                "algorithm_candidate_id": (
                    "candidate_local_strategy_task1_pi_deeponet_correction_head_v0"
                ),
                "candidate_family": "local_strategy_skill",
                "target_problem": "PDE_Burgers",
                "target_task": "task1",
                "model_family": "PI-DeepONet",
                "hypothesis": (
                    "A small PI-DeepONet correction head can learn a low-rank "
                    "residual on top of the retrieval prior while keeping the "
                    "context prefix exact."
                ),
                "mechanism": (
                    "Use branch features from the first 10 frames and trunk "
                    "features from (t, x) to fit a bounded correction field, "
                    "with optional Burgers residual nudging."
                ),
                "executor_spec": {
                    "cli_command": "run-hypothesis-lab",
                    "implementation_hooks": ["task1_pi_deeponet_correction_head"],
                },
                "expected_observables": [
                    "correction head improves score over the base trajectory",
                    "context prefix remains exact",
                ],
                "risk_controls": ["correction head can overfit validation windows"],
                "source_grounding": {
                    "algorithm_card_id": (
                        "local_strategy_task1_pi_deeponet_correction_head_v0"
                    ),
                    "source_ids": ["task1_pi_deeponet_correction_head_v0"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hypothesis_lab_builds_cards_from_manual_operator_report(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    report_path = _manual_report(tmp_path / "manual_report.json")
    cards = build_knowledge_cards(
        manual_report_paths=[report_path],
        output_path=tmp_path / "cards.json",
        force=False,
    )

    assert cards["state_hint"] == "HYPOTHESIS_LAB_KNOWLEDGE_CARDS_BUILT"
    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "manual_tuning_report"
    assert "linear trend" in card["claim"].lower()
    assert "spectral" in card["procedural_heuristic"].lower()
    assert card["evidence_status"] == "tested"
    assert card["implementation_hooks"] == ["linear_trend_spectral"]


def test_hypothesis_lab_builds_cards_from_checkpoint_paths(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    fno_checkpoint = tmp_path / "1D_Burgers_Sols_Nu0.001_FNO.pt"
    unet_checkpoint = tmp_path / "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt"
    fno_checkpoint.write_bytes(b"fno")
    unet_checkpoint.write_bytes(b"unet")

    cards = build_knowledge_cards(
        manual_report_paths=[],
        checkpoint_paths=[fno_checkpoint, unet_checkpoint],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 2
    assert {card["evidence"]["model_family"] for card in cards["cards"]} == {
        "FNO",
        "UNet",
    }
    assert cards["cards"][0]["implementation_hooks"] == [
        "official_checkpoint_calibration"
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert "official_checkpoint_calibration" in [
        item["candidate_family"] for item in hypotheses["hypotheses"]
    ]


def test_hypothesis_lab_builds_cards_from_algorithm_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    candidates_path = _local_asset_algorithm_candidates(
        tmp_path / "local_asset_candidates.json"
    )
    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[candidates_path],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "algorithm_candidate"
    assert card["implementation_hooks"] == ["linear_trend_spectral"]
    assert card["evidence"]["algorithm_candidate_id"] == (
        "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
    )
    assert card["evidence"]["candidate_key"] == (
        "local_strategy_data_inference_linear_trend_spectral_v0|local_strategy_skill"
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == "linear_trend_spectral"


def test_hypothesis_lab_builds_retrieval_prior_from_algorithm_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _retrieval_asset_algorithm_candidates(tmp_path / "retrieval_candidates.json")
        ],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == ["task1_retrieval_prior"]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    families = [item["candidate_family"] for item in hypotheses["hypotheses"]]
    assert families == ["task1_retrieval_refinement", "task1_retrieval_prior"]
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    assert blueprints["executable_blueprint_audit"]["decision"] == "approved"
    assert blueprints["executable_blueprint_audit"]["issue_count"] == 0
    assert blueprints["executable_blueprint_audit"]["normalized_blueprint_count"] == 2
    by_family = {
        blueprint["candidate_family"]: blueprint
        for blueprint in blueprints["blueprints"]
    }
    prior = by_family["task1_retrieval_prior"]
    assert prior["sandbox"]["allowed_executor"] == "controlled_numpy_retrieval_prior"
    assert prior["parameter_grid"]["k"] == [8, 12, 20]
    assert prior["parameter_grid"]["mode"] == ["affine"]
    refinement = by_family["task1_retrieval_refinement"]
    assert refinement["sandbox"]["allowed_executor"] == (
        "controlled_numpy_retrieval_refinement"
    )
    assert refinement["search_mode"] == "evidence_focused_coarse_to_fine"
    candidate_params = refinement["parameter_grid"]["candidate_params"]
    assert 8 <= len(candidate_params) <= 24
    assert {
        "k": 48,
        "mode": "affine",
        "early_blend_steps": 20,
        "persistence_weight_start": 0.5,
        "spectral_strength": 0.5,
    } in candidate_params
    assert {
        "k": 48,
        "mode": "affine",
        "early_blend_steps": 20,
        "persistence_weight_start": 0.5,
        "spectral_strength": 1.0,
    } in candidate_params
    assert any(row["mode"] == "delta" for row in candidate_params)
    assert any(row["mode"] == "raw" for row in candidate_params)


def test_hypothesis_lab_builds_retrieval_refinement_from_local_strategy_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _retrieval_refinement_asset_algorithm_candidates(
                tmp_path / "refinement_candidates.json"
            )
        ],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == [
        "task1_retrieval_refinement"
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    families = [item["candidate_family"] for item in hypotheses["hypotheses"]]
    assert families == [
        "task1_retrieval_refinement_local_search",
        "task1_retrieval_refinement",
    ]
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "task1_retrieval_refinement_local_search"
    )
    assert hypotheses["hypotheses"][1]["candidate_family"] == (
        "task1_retrieval_refinement"
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    assert blueprints["blueprints"][0]["blueprint_id"] == (
        "bp_task1_retrieval_refinement_local_search_001"
    )
    local_search = blueprints["blueprints"][0]
    assert local_search["sandbox"]["allowed_executor"] == (
        "controlled_numpy_retrieval_refinement"
    )
    candidate_params = local_search["parameter_grid"]["candidate_params"]
    assert 12 <= len(candidate_params) <= 36
    assert any(row["k"] == 52 for row in candidate_params)
    assert any(row["early_blend_steps"] == 15 for row in candidate_params)
    assert any(row["early_blend_steps"] == 25 for row in candidate_params)
    assert any(row["persistence_weight_start"] == 0.4 for row in candidate_params)
    assert any(row["persistence_weight_start"] == 0.6 for row in candidate_params)
    assert any(row["spectral_strength"] == 0.75 for row in candidate_params)
    assert any(row["mode"] == "delta" for row in candidate_params)
    assert any(row["mode"] == "raw" for row in candidate_params)
    assert blueprints["blueprints"][1]["blueprint_id"] == (
        "bp_task1_retrieval_refinement_001"
    )


def test_hypothesis_lab_designs_sem_burgers_rollout_from_local_strategy_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _sem_burgers_asset_algorithm_candidates(tmp_path / "sem_candidates.json")
        ],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == ["task1_sem_burgers_rollout"]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "task1_sem_burgers_rollout"
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    blueprint = blueprints["blueprints"][0]
    assert blueprint["candidate_family"] == "task1_sem_burgers_rollout"
    assert blueprint["sandbox"]["allowed_executor"] == (
        "controlled_numpy_sem_burgers_rollout"
    )
    assert blueprint["parameter_grid"]["candidate_params"][0] == {
        "viscosity": 0.001,
        "dt": 0.05,
        "element_count": 16,
        "polynomial_order": 8,
        "substeps": 8,
        "filter_strength": 0.02,
        "interface_penalty": 0.02,
    }


def test_hypothesis_lab_run_executes_sem_burgers_rollout_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab

    target = np.full((1, 200, 256), 0.25, dtype=np.float32)
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_hypothesis_lab(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _sem_burgers_asset_algorithm_candidates(tmp_path / "sem_candidates.json")
        ],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        force=False,
    )

    assert report["sandbox"]["candidate_family"] == "task1_sem_burgers_rollout"
    assert report["sandbox"]["best_candidate"]["candidate_id"].startswith(
        "task1_sem_burgers_rollout"
    )


def test_hypothesis_lab_designs_physics_residual_refinement_from_local_strategy_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _physics_residual_refinement_asset_algorithm_candidates(
                tmp_path / "physics_residual_candidates.json"
            )
        ],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == [
        "task1_physics_residual_refinement"
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "task1_physics_residual_refinement"
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    blueprint = blueprints["blueprints"][0]
    assert blueprint["candidate_family"] == "task1_physics_residual_refinement"
    assert blueprint["sandbox"]["allowed_executor"] == (
        "controlled_numpy_burgers_residual_refinement"
    )
    assert blueprint["parameter_grid"]["candidate_params"][0]["base_family"] == (
        "task1_retrieval_refinement"
    )


def test_hypothesis_lab_designs_pi_deeponet_correction_from_local_strategy_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _pi_deeponet_correction_asset_algorithm_candidates(
                tmp_path / "pi_deeponet_candidates.json"
            )
        ],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == [
        "task1_pi_deeponet_correction_head"
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "task1_pi_deeponet_correction_head"
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    blueprint = blueprints["blueprints"][0]
    assert blueprint["candidate_family"] == "task1_pi_deeponet_correction_head"
    assert blueprint["sandbox"]["allowed_executor"] == (
        "controlled_numpy_pi_deeponet_correction_head"
    )
    assert blueprint["parameter_grid"]["candidate_params"][0]["base_family"] == (
        "task1_retrieval_refinement"
    )


def test_hypothesis_lab_designs_early_refinement_from_early_bottleneck_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    report_path = _adaptive_retrieval_refinement_sandbox_report(
        tmp_path / "early_bottleneck_sandbox.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["best_candidate"]["params"]["k"] = 92
    payload["best_candidate"]["candidate_id"] = (
        "task1_retrieval_refinement_k92_affine_eb20_pw0.6_s0.5"
    )
    payload["verifier"] = {
        "decision": "promote_to_strategy_proposal",
        "early_segment_diagnosis": {
            "schema_version": 2,
            "state_hint": "TASK1_EARLY_SEGMENT_DIAGNOSED",
            "decision": "early_segment_bottleneck",
            "reason_codes": ["early_score_low", "early_weaker_than_late"],
            "bottleneck_type": "initial_derivative_mismatch",
            "error_taxonomy": {
                "initial_derivative_mismatch": {"triggered": True},
                "over_smoothing": {"triggered": True},
            },
            "candidate_trend_attribution": {
                "primary_metric": "early_rel_mse",
                "recommended_axis": "mode",
            },
            "action_directives": [
                {
                    "directive": "expand_linear_velocity_anchor_search",
                    "target_family": "task1_early_refinement",
                    "rationale": "primary early bottleneck is initial_derivative_mismatch",
                },
                {
                    "directive": "prioritize_axis:mode",
                    "target_family": "task1_early_refinement",
                    "rationale": "mode axis has the largest early-rel-mse spread",
                },
            ],
            "objective_recommendation": {
                "primary": "early_rel_mse",
                "next_candidate_family": "task1_early_refinement",
            },
        },
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[report_path],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == [
        "task1_early_refinement",
        "task1_retrieval_refinement_local_search",
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == "task1_early_refinement"
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    early = blueprints["blueprints"][0]
    assert early["blueprint_id"] == "bp_task1_early_refinement_001"
    assert early["sandbox"]["allowed_executor"] == (
        "controlled_numpy_task1_early_refinement"
    )
    assert early["experiment_plan"]["objective"]["primary"] == "early_rel_mse"
    assert early["experiment_plan"]["decision"] == "two_stage_budgeted_early_search"
    assert early["experiment_plan"]["budget_policy"] == {
        "max_candidate_budget": 160,
        "preserve_required_directive_probes": True,
    }
    assert [stage["stage"] for stage in early["experiment_plan"]["stage_plan"]] == [
        "axis_screen",
        "focused_refine",
    ]
    assert early["experiment_plan"]["guardrails"] == [
        "middle_score_non_regression",
        "late_rmse_non_regression",
        "context_prefix_exact",
    ]
    assert any(
        row["early_anchor"] == "linear_velocity"
        for row in early["parameter_grid"]["candidate_params"]
    )
    candidate_params = early["parameter_grid"]["candidate_params"]
    assert len(candidate_params) <= 160
    assert max(row["k"] for row in candidate_params) <= 99
    anchors = {row["early_anchor"] for row in candidate_params}
    assert {
        "damped_linear_velocity",
        "retrieval_derivative_match",
        "scaled_damped_velocity",
    } <= anchors
    assert {0.25, 0.5} <= {
        row.get("velocity_scale")
        for row in candidate_params
        if row["early_anchor"] == "scaled_damped_velocity"
    }
    assert all(
        row.get("velocity_scale") in {0.25, 0.5, 0.75}
        for row in candidate_params
        if row["early_anchor"] == "scaled_damped_velocity"
    )
    modes = {row["mode"] for row in candidate_params}
    assert {"affine", "delta", "raw"} <= modes
    strengths = {row["spectral_strength"] for row in candidate_params}
    assert 0.0 in strengths
    derivative_probe_count = sum(
        row["early_anchor"]
        in {"damped_linear_velocity", "retrieval_derivative_match"}
        for row in candidate_params
    )
    assert 2 <= derivative_probe_count <= 32
    assert blueprints["executable_blueprint_audit"]["decision"] == "approved"


def test_hypothesis_lab_builds_early_policy_card_from_adaptive_search_policy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    policy = {
        "axis": "early_anchor",
        "focus_values": ["persistence", "retrieval_derivative_match"],
        "rescue_probe_values": ["scaled_damped_velocity"],
        "deprioritize_values": ["second_order_velocity"],
        "recommended_actions": [
            {
                "action": "budget_probe",
                "axis": "early_anchor",
                "value": "scaled_damped_velocity",
                "reason": "non_top_near_best_early_rel_mse",
            },
            {
                "action": "deprioritize",
                "axis": "early_anchor",
                "value": "second_order_velocity",
                "reason": "generated_but_not_competitive_in_candidate_pool",
            },
        ],
    }
    report_path = tmp_path / "adaptive_policy.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_ADAPTIVE_HEARTBEAT_RAN",
                "policy_events": [
                    {
                        "stage_index": 1,
                        "action": "continue_hypothesis_lab_with_search_policy",
                        "trajectory_state": "plateau",
                        "reason_codes": ["stagnation_detected"],
                        "best_candidate_id": "task1_scaled_anchor_probe",
                        "best_score": 85.6,
                        "hypothesis_lab_search_policy": {
                            "source_path": str(tmp_path / "hypothesis_lab_report.json"),
                            "policy": policy,
                        },
                    }
                ],
                "best_candidate": {
                    "candidate_id": "task1_scaled_anchor_probe",
                    "score": 85.6,
                    "metadata": {
                        "source": "hypothesis_lab",
                        "params": {
                            "k": 64,
                            "mode": "affine",
                            "early_blend_steps": 20,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.5,
                            "early_anchor": "persistence",
                        },
                    },
                },
                "next_hypothesis_lab_inputs": {
                    "search_space_policy": {
                        "source_path": str(tmp_path / "hypothesis_lab_report.json"),
                        "policy": policy,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[report_path],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["source_type"] == "adaptive_hypothesis_lab_search_policy"
    assert cards["cards"][0]["implementation_hooks"] == ["task1_early_refinement"]
    assert cards["cards"][0]["evidence"]["candidate_pool_attribution"] == {
        "search_space_policy": policy
    }
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    early_plan = blueprints["blueprints"][0]["experiment_plan"]
    assert early_plan["memory_policy"]["rescue_probe_anchors"] == [
        "scaled_damped_velocity"
    ]
    assert early_plan["memory_policy"]["deprioritized_anchors"] == [
        "second_order_velocity"
    ]


def test_hypothesis_lab_applies_mode_policy_from_adaptive_search_policy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    policy = {
        "axis": "mode",
        "focus_values": ["affine"],
        "rescue_probe_values": [],
        "deprioritize_values": ["delta", "raw"],
        "recommended_actions": [
            {
                "action": "keep_focus",
                "axis": "mode",
                "value": "affine",
                "reason": "top_candidate_near_best_early_rel_mse",
            },
            {
                "action": "deprioritize",
                "axis": "mode",
                "value": "delta",
                "reason": "generated_but_not_competitive_in_candidate_pool",
            },
            {
                "action": "deprioritize",
                "axis": "mode",
                "value": "raw",
                "reason": "generated_but_not_competitive_in_candidate_pool",
            },
        ],
    }
    report_path = tmp_path / "adaptive_mode_policy.json"
    report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_ADAPTIVE_HEARTBEAT_RAN",
                "policy_events": [
                    {
                        "stage_index": 1,
                        "action": "continue_current_generator",
                        "trajectory_state": "improving_or_unblocked",
                        "reason_codes": ["no_plateau_detected"],
                        "best_candidate_id": "task1_retrieval_refinement_k48_affine",
                        "best_score": 110.36,
                        "hypothesis_lab_search_policy": {
                            "source_path": str(tmp_path / "hypothesis_lab_report.json"),
                            "policy": policy,
                        },
                    }
                ],
                "best_candidate": {
                    "candidate_id": "task1_retrieval_refinement_k48_affine",
                    "score": 110.36,
                    "metadata": {
                        "source": "hypothesis_lab",
                        "params": {
                            "k": 48,
                            "mode": "affine",
                            "early_blend_steps": 25,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.5,
                        },
                    },
                },
                "next_hypothesis_lab_inputs": {
                    "search_space_policy": {
                        "source_path": str(tmp_path / "hypothesis_lab_report.json"),
                        "policy": policy,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[report_path],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["source_type"] == "adaptive_hypothesis_lab_search_policy"
    assert cards["cards"][0]["implementation_hooks"] == [
        "task1_retrieval_refinement_local_search"
    ]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    candidate_params = blueprints["blueprints"][0]["parameter_grid"][
        "candidate_params"
    ]
    assert {row["mode"] for row in candidate_params} == {"affine"}
    assert blueprints["blueprints"][0]["experiment_plan"]["axis_plan"][0] == {
        "axis": "mode",
        "direction": "focus",
        "reason": "candidate_pool_search_policy",
        "budget": 3,
    }


def test_hypothesis_lab_designs_phase_and_bias_aware_early_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    report_path = _adaptive_retrieval_refinement_sandbox_report(
        tmp_path / "phase_bias_bottleneck_sandbox.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["verifier"] = {
        "decision": "keep_as_evidence",
        "early_segment_diagnosis": {
            "schema_version": 2,
            "state_hint": "TASK1_EARLY_SEGMENT_DIAGNOSED",
            "decision": "early_segment_bottleneck",
            "reason_codes": ["early_rel_mse_high"],
            "bottleneck_type": "phase_shift",
            "error_taxonomy": {
                "phase_shift": {"triggered": True},
                "amplitude_bias": {"triggered": True},
                "high_frequency_noise": {"triggered": True},
            },
            "candidate_trend_attribution": {
                "primary_metric": "early_rel_mse",
                "recommended_axis": "early_blend_steps",
            },
            "action_directives": [
                {
                    "directive": "add_phase_alignment_or_derivative_distance_candidates",
                    "target_family": "task1_early_refinement",
                    "rationale": "phase shift dominates the short-horizon error",
                },
                {
                    "directive": "add_short_horizon_affine_bias_calibration",
                    "target_family": "task1_early_refinement",
                    "rationale": "mean amplitude bias is active",
                },
            ],
            "objective_recommendation": {
                "primary": "early_rel_mse",
                "next_candidate_family": "task1_early_refinement",
            },
        },
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[report_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    early = blueprints["blueprints"][0]
    candidate_params = early["parameter_grid"]["candidate_params"]
    anchors = {row["early_anchor"] for row in candidate_params}
    assert {"phase_aligned_retrieval", "mean_trend_bias"} <= anchors
    assert any(row.get("handoff_curve") == "cosine" for row in candidate_params)
    assert any(row.get("handoff_curve") == "quadratic" for row in candidate_params)
    assert any(row.get("phase_shift_scale") == 0.5 for row in candidate_params)
    assert any(row["spectral_strength"] > 0.5 for row in candidate_params)
    assert len(candidate_params) <= 160
    plan = early["experiment_plan"]
    assert "phase_alignment" in plan["axis_plan_by_bottleneck"]
    assert "bias_calibration" in plan["axis_plan_by_bottleneck"]


def test_hypothesis_lab_designs_compound_early_calibration_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    report_path = _adaptive_retrieval_refinement_sandbox_report(
        tmp_path / "compound_early_bottleneck_sandbox.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["verifier"] = {
        "decision": "keep_as_evidence",
        "early_segment_diagnosis": {
            "schema_version": 2,
            "state_hint": "TASK1_EARLY_SEGMENT_DIAGNOSED",
            "decision": "early_segment_bottleneck",
            "reason_codes": ["phase_shift", "amplitude_bias"],
            "bottleneck_type": "phase_shift",
            "error_taxonomy": {
                "phase_shift": {"triggered": True},
                "amplitude_bias": {"triggered": True},
                "initial_derivative_mismatch": {"triggered": True},
            },
            "candidate_trend_attribution": {
                "primary_metric": "early_rel_mse",
                "recommended_axis": "early_anchor",
            },
            "action_directives": [
                {
                    "directive": "add_phase_alignment_or_derivative_distance_candidates",
                    "target_family": "task1_early_refinement",
                    "rationale": "transport error remains active",
                },
                {
                    "directive": "add_short_horizon_affine_bias_calibration",
                    "target_family": "task1_early_refinement",
                    "rationale": "context amplitude statistics are miscalibrated",
                },
            ],
            "objective_recommendation": {
                "primary": "early_rel_mse",
                "next_candidate_family": "task1_early_refinement",
            },
        },
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[report_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    early = blueprints["blueprints"][0]
    candidate_params = early["parameter_grid"]["candidate_params"]
    anchors = {row["early_anchor"] for row in candidate_params}
    assert {"affine_context_calibration", "phase_derivative_hybrid"} <= anchors
    plan = early["experiment_plan"]
    compound = plan["axis_plan_by_bottleneck"]["compound_transport_calibration"]
    assert compound["candidate_hooks"] == [
        "phase_derivative_hybrid",
        "affine_context_calibration",
    ]
    assert len(candidate_params) <= 160


def test_hypothesis_lab_builds_retrieval_refinement_center_from_sandbox_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_retrieval_refinement_sandbox_report(
                tmp_path / "retrieval_refinement_sandbox.json"
            )
        ],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "adaptive_retrieval_refinement_sandbox"
    assert card["implementation_hooks"] == ["task1_retrieval_refinement_local_search"]
    assert card["evidence_status"] == "local_search_center_ready"
    assert card["evidence"]["params"]["persistence_weight_start"] == 0.6
    assert card["evidence"]["top_params"][0]["k"] == 52


def test_hypothesis_lab_centers_local_search_from_sandbox_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_retrieval_refinement_sandbox_report(
                tmp_path / "retrieval_refinement_sandbox.json"
            )
        ],
        algorithm_candidate_paths=[
            _retrieval_refinement_asset_algorithm_candidates(
                tmp_path / "refinement_candidates.json"
            )
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    local_search_hypothesis = hypotheses["hypotheses"][0]

    assert local_search_hypothesis["candidate_family"] == (
        "task1_retrieval_refinement_local_search"
    )
    assert local_search_hypothesis["evidence_params"][0]["params"][
        "persistence_weight_start"
    ] == 0.6

    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    candidate_params = blueprints["blueprints"][0]["parameter_grid"][
        "candidate_params"
    ]
    weights = {row["persistence_weight_start"] for row in candidate_params}
    assert 0.55 in weights
    assert 0.65 in weights
    assert {
        "k": 48,
        "mode": "affine",
        "early_blend_steps": 20,
        "persistence_weight_start": 0.6,
        "spectral_strength": 0.5,
    } in candidate_params


def test_hypothesis_lab_expands_axis_when_sandbox_top_candidates_show_trend(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_retrieval_refinement_weight_trend_report(
                tmp_path / "weight_trend_sandbox.json"
            )
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)

    directives = hypotheses["hypotheses"][0]["evidence_params"][0][
        "search_directives"
    ]
    assert {
        "axis": "persistence_weight_start",
        "direction": "increase",
        "reason": "top_candidate_score_trend",
    } in directives

    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    candidate_params = blueprints["blueprints"][0]["parameter_grid"][
        "candidate_params"
    ]
    weights = {row["persistence_weight_start"] for row in candidate_params}
    assert 0.85 in weights
    assert 0.9 in weights


def test_hypothesis_lab_blueprint_records_experiment_plan_from_axis_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_retrieval_refinement_weight_trend_report(
                tmp_path / "weight_trend_sandbox.json"
            )
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    blueprint = blueprints["blueprints"][0]
    plan = blueprint["experiment_plan"]
    assert plan["plan_id"] == "xp_task1_retrieval_refinement_local_search_001"
    assert plan["decision"] == "axis_directed_local_search"
    assert plan["objective"]["primary"] == "task1_total_score_proxy"
    assert plan["objective"]["focus"] == "score_trend"
    assert {
        "axis": "persistence_weight_start",
        "direction": "increase",
        "reason": "top_candidate_score_trend",
        "budget": 3,
    } in plan["axis_plan"]
    assert plan["candidate_budget"] == len(
        blueprint["parameter_grid"]["candidate_params"]
    )
    assert "stop_if_no_candidate_improves_seed" in plan["stop_criteria"]
    assert "switch_family_after_repeated_axis_plateau" in plan["switch_criteria"]


def test_hypothesis_lab_avoids_opposite_axis_region_after_score_trend(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_retrieval_refinement_weight_trend_report(
                tmp_path / "weight_trend_sandbox.json"
            )
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    blueprint = blueprints["blueprints"][0]
    plan = blueprint["experiment_plan"]
    assert {
        "axis": "persistence_weight_start",
        "relation": "below_center",
        "reason": "opposite_of_score_trend",
    } in plan["avoid_rules"]
    weights = {
        row["persistence_weight_start"]
        for row in blueprint["parameter_grid"]["candidate_params"]
    }
    assert min(weights) >= 0.7


def test_hypothesis_lab_builds_cards_from_literature_notes(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    cards = build_knowledge_cards(
        manual_report_paths=[],
        literature_paths=[_literature_note(tmp_path / "residual_note.md")],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "literature_note"
    assert card["implementation_hooks"] == ["learned_residual_correction"]
    assert "Horizon-conditioned residual correction" in card["claim"]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "learned_residual_correction"
    )


def test_hypothesis_lab_extracts_internalizable_top_candidate_not_only_best(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    cards = build_knowledge_cards(
        manual_report_paths=[_manual_report_with_external_best(tmp_path / "manual.json")],
        output_path=None,
        force=False,
    )

    hooks = {
        hook
        for card in cards["cards"]
        for hook in card["implementation_hooks"]
    }
    assert "linear_trend_spectral" in hooks
    assert "external_residual_blend" in hooks
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    families = [item["candidate_family"] for item in hypotheses["hypotheses"]]
    assert families == ["linear_trend_spectral", "external_residual_blend"]


def test_hypothesis_lab_builds_cards_from_adaptive_plateau_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[_adaptive_plateau_report(tmp_path / "adaptive.json")],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "adaptive_heartbeat_report"
    assert card["evidence_status"] == "plateaued"
    assert card["implementation_hooks"] == ["linear_trend_spectral_escape"]
    assert "verified_blueprint_plateau" in card["known_risks"]
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "linear_trend_spectral_escape"
    )
    assert hypotheses["hypotheses"][0]["priority"] < 1


def test_hypothesis_lab_builds_cards_from_scientific_logs(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    cards = build_knowledge_cards(
        manual_report_paths=[],
        log_paths=[_scientific_log(tmp_path / "task1_logs.log")],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "scientific_log"
    assert card["evidence_status"] == "observed_plateau"
    assert "linear_trend_spectral_escape" in card["implementation_hooks"]
    assert "verified blueprint reuse" in card["claim"].lower()


def test_hypothesis_lab_scientific_log_remembers_late_ramp_attempt(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    cards = build_knowledge_cards(
        manual_report_paths=[],
        log_paths=[_late_ramp_scientific_log(tmp_path / "task1_logs.log")],
        output_path=None,
        force=False,
    )

    assert cards["cards"][0]["implementation_hooks"] == [
        "linear_trend_late_spectral_ramp"
    ]
    assert cards["cards"][0]["evidence"]["candidate_id"] == (
        "linear_trend_spectral_w5_g-0.075_d70_s1.5_ls2.25_lt60"
    )


def test_hypothesis_lab_generates_hypotheses_and_blueprints(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[_manual_report(tmp_path / "manual_report.json")],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(
        cards=cards,
        output_path=tmp_path / "hypotheses.json",
        force=False,
    )
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=tmp_path / "blueprints.json",
        force=False,
    )

    assert hypotheses["state_hint"] == "HYPOTHESIS_LAB_HYPOTHESES_GENERATED"
    assert hypotheses["hypotheses"][0]["candidate_family"] == "linear_trend_spectral"
    assert hypotheses["hypothesis_count"] == 1
    blueprint = blueprints["blueprints"][0]
    assert blueprint["candidate_family"] == "linear_trend_spectral"
    assert blueprints["blueprint_count"] == 1
    assert blueprint["seed_params"][0]["params"] == {
        "slope_window": 5,
        "gain": -0.1,
        "decay": 60.0,
        "spectral_strength": 1.0,
    }
    assert blueprint["sandbox"]["allowed_executor"] == "controlled_numpy_operator"
    assert 1.0 in blueprint["parameter_grid"]["spectral_strength"]


def test_hypothesis_lab_designs_escape_blueprint_from_plateau_cards(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[_adaptive_plateau_report(tmp_path / "adaptive.json")],
        log_paths=[_scientific_log(tmp_path / "task1_logs.log")],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    blueprint = blueprints["blueprints"][0]
    assert blueprint["candidate_family"] == "linear_trend_spectral"
    assert blueprint["search_mode"] == "plateau_escape"
    assert 0.0 in blueprint["parameter_grid"]["spectral_strength"]
    assert 1.5 in blueprint["parameter_grid"]["spectral_strength"]
    assert -0.025 in blueprint["parameter_grid"]["gain"]


def test_hypothesis_lab_generates_late_spectral_ramp_after_escape_plateau(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        adaptive_report_paths=[
            _adaptive_plateau_escape_report(tmp_path / "adaptive_escape.json")
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    assert hypotheses["hypotheses"][0]["candidate_family"] == (
        "linear_trend_late_spectral_ramp"
    )
    blueprint = blueprints["blueprints"][0]
    assert blueprint["search_mode"] == "late_spectral_ramp"
    assert blueprint["candidate_family"] == "linear_trend_spectral"
    assert blueprint["parameter_grid"]["spectral_strength"] == [1.5]
    assert 2.0 in blueprint["parameter_grid"]["late_spectral_strength"]
    assert 80 in blueprint["parameter_grid"]["late_start"]


def test_hypothesis_lab_designs_checkpoint_and_residual_blueprints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    checkpoint = tmp_path / "burgers_FNO.pt"
    checkpoint.write_bytes(b"checkpoint")
    cards = build_knowledge_cards(
        manual_report_paths=[],
        checkpoint_paths=[checkpoint],
        literature_paths=[_literature_note(tmp_path / "residual_note.md")],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    by_family = {item["candidate_family"]: item for item in blueprints["blueprints"]}
    checkpoint_blueprint = by_family["official_checkpoint_calibration"]
    assert checkpoint_blueprint["sandbox"]["allowed_executor"] == (
        "official_checkpoint_candidate_pipeline"
    )
    assert checkpoint_blueprint["next_command"] == "run-checkpoint-family-sandbox"
    assert checkpoint_blueprint["execution_plan"]["cli_command"] == (
        "run-checkpoint-family-sandbox"
    )
    assert "--checkpoint-path" in checkpoint_blueprint["execution_plan"][
        "required_options"
    ]
    assert checkpoint_blueprint["checkpoint_candidates"][0]["model_family"] == "FNO"
    residual_blueprint = by_family["learned_residual_correction"]
    assert residual_blueprint["sandbox"]["allowed_executor"] == (
        "controlled_numpy_residual_operator"
    )
    assert residual_blueprint["parameter_grid"]["residual_shrinkage"] == [
        0.25,
        0.5,
        0.75,
    ]


def test_hypothesis_lab_ranks_hypotheses_by_evidence_budget_and_novelty(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    checkpoint = tmp_path / "burgers_FNO.pt"
    checkpoint.write_bytes(b"checkpoint")
    cards = build_knowledge_cards(
        manual_report_paths=[_manual_report(tmp_path / "manual.json")],
        checkpoint_paths=[checkpoint],
        literature_paths=[_literature_note(tmp_path / "residual_note.md")],
        output_path=None,
        force=False,
    )

    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)

    families = [item["candidate_family"] for item in hypotheses["hypotheses"]]
    assert families.index("learned_residual_correction") < families.index(
        "linear_trend_spectral"
    )
    assert families.index("official_checkpoint_calibration") < families.index(
        "linear_trend_spectral"
    )
    for hypothesis in hypotheses["hypotheses"]:
        assert "ranking_score" in hypothesis
        assert "ranking_reasons" in hypothesis


def test_hypothesis_lab_groups_duplicate_cards_by_candidate_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    cards = build_knowledge_cards(
        manual_report_paths=[
            _manual_report(tmp_path / "manual_a.json"),
            _manual_report(tmp_path / "manual_b.json"),
        ],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 2
    assert hypotheses["hypothesis_count"] == 1
    assert len(hypotheses["hypotheses"][0]["source_card_ids"]) == 2
    assert blueprints["blueprint_count"] == 1


def test_hypothesis_lab_sandbox_verifies_blueprint_with_submission(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
        run_blueprint_sandbox,
    )

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    base = np.sin(x)[None, None, :]
    context = np.repeat(base, 10, axis=1).astype(np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    last = context[:, 9:10, :]
    for step in range(190):
        target[:, 10 + step : 11 + step, :] = last - np.float32(0.002 * (step + 1))
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    cards = build_knowledge_cards(
        manual_report_paths=[_manual_report(tmp_path / "manual_report.json")],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )
    sandbox = run_blueprint_sandbox(
        blueprint=blueprints["blueprints"][0],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["state_hint"] == "HYPOTHESIS_LAB_SANDBOX_VERIFIED"
    assert sandbox["verifier"]["decision"] == "promote_to_strategy_proposal"
    assert sandbox["verifier"]["score_delta_vs_baseline"] > 0.0
    assert sandbox["verifier"]["metric_results"]["score_delta_vs_baseline"]["passed"]
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]
    assert "late_rmse_delta_vs_baseline" in sandbox["verifier"]["metric_results"]
    assert sandbox["verifier"]["metric_results"]["late_rmse_delta_vs_baseline"]["value"] > 0.0
    assert Path(sandbox["artifacts"]["best_submission_dir"]).exists()


def test_hypothesis_lab_sandbox_executes_late_spectral_ramp_params(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    context = np.ones((1, 10, 256), dtype=np.float32)
    target = np.concatenate(
        [context, np.repeat(context[:, -1:, :], 190, axis=1)],
        axis=1,
    )
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_linear_trend_late_spectral_ramp_001",
        "candidate_family": "linear_trend_spectral",
        "search_mode": "late_spectral_ramp",
        "parameter_grid": {
            "slope_window": [5],
            "gain": [-0.075],
            "decay": [70.0],
            "spectral_strength": [1.5],
            "late_spectral_strength": [2.0],
            "late_start": [80],
        },
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["best_candidate"]["params"]["late_spectral_strength"] == 2.0
    assert sandbox["best_candidate"]["params"]["late_start"] == 80
    assert "_ls2" in sandbox["best_candidate"]["candidate_id"]


def test_hypothesis_lab_sandbox_executes_learned_residual_correction(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    base = np.sin(x)[None, None, :]
    context = np.repeat(base, 10, axis=1).astype(np.float32)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, -1, :] + np.float32(0.1)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_learned_residual_correction_001",
        "candidate_family": "learned_residual_correction",
        "search_mode": "horizon_mean_residual",
        "parameter_grid": {
            "residual_shrinkage": [0.25, 0.5, 0.75, 1.0],
            "residual_spectral_strength": [0.0],
        },
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["best_candidate"]["candidate_id"].startswith("learned_residual")
    assert sandbox["verifier"]["score_delta_vs_baseline"] > 0.0
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]


def test_hypothesis_lab_sandbox_records_task1_multiview_judge(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((4, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.1, 0.2, 0.3]):
        context = np.sin(x + np.float32(offset))[None, :]
        for step in range(200):
            target[sample, step, :] = context + np.float32(0.001 * step)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", target[:, :10, :])
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_retrieval_prior_001",
        "candidate_family": "task1_retrieval_prior",
        "search_mode": "affine_knn_trajectory_prior",
        "parameter_grid": {"k": [1], "mode": ["affine"]},
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    multiview = sandbox["verifier"]["multiview_judge"]
    assert multiview["state_hint"] == "TASK1_MULTIVIEW_LOCAL_JUDGE_SCORED"
    assert multiview["decision"] in {
        "consistent_improvement",
        "mixed_or_insufficient_evidence",
    }
    assert set(multiview["views"]) == {
        "fast_full",
        "heldout_split",
        "early_accuracy",
        "late_stability",
    }
    assert multiview["views"]["heldout_split"]["sample_count"] == 2
    assert "segment_score_delta" in multiview["views"]["fast_full"]
    assert "late_rmse_delta" in multiview["views"]["late_stability"]


def test_hypothesis_lab_sandbox_records_early_segment_diagnosis(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((4, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.1, 0.2, 0.3]):
        context = np.sin(x + np.float32(offset))[None, :]
        for step in range(200):
            target[sample, step, :] = context + np.float32(0.001 * step)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", target[:, :10, :])
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_retrieval_prior_001",
        "candidate_family": "task1_retrieval_prior",
        "search_mode": "affine_knn_trajectory_prior",
        "parameter_grid": {"k": [1], "mode": ["affine"]},
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    diagnosis = sandbox["verifier"]["early_segment_diagnosis"]
    assert diagnosis["schema_version"] == 2
    assert diagnosis["state_hint"] == "TASK1_EARLY_SEGMENT_DIAGNOSED"
    assert diagnosis["objective_recommendation"]["primary"] == "early_rel_mse"
    assert diagnosis["objective_recommendation"]["next_candidate_family"] == (
        "task1_early_refinement"
    )
    assert set(diagnosis["segments"]) == {"baseline", "candidate"}
    assert isinstance(diagnosis["segments"]["candidate"]["early"]["rel_mse"], float)
    assert isinstance(diagnosis["segments"]["candidate"]["middle"]["score"], float)
    assert isinstance(diagnosis["segments"]["candidate"]["late"]["rmse"], float)
    assert [item["name"] for item in diagnosis["early_subsegments"]] == [
        "future_0_10",
        "future_10_25",
        "future_25_47",
    ]
    for item in diagnosis["early_subsegments"]:
        assert isinstance(item["candidate_rel_mse"], float)
        assert isinstance(item["baseline_rel_mse"], float)
        assert "rel_mse_delta_vs_baseline" in item
    taxonomy = diagnosis["error_taxonomy"]
    assert diagnosis["bottleneck_type"] in {
        "none",
        *taxonomy.keys(),
    }
    assert set(taxonomy) >= {
        "initial_derivative_mismatch",
        "phase_shift",
        "amplitude_bias",
        "over_smoothing",
        "high_frequency_noise",
    }
    assert diagnosis["guardrail_diagnosis"]["middle"]["metric"] == "score"
    assert diagnosis["guardrail_diagnosis"]["late"]["metric"] == "rmse"
    assert diagnosis["action_directives"]
    assert set(diagnosis["action_directives"][0]) >= {
        "directive",
        "rationale",
        "target_family",
    }
    metrics = sandbox["verifier"]["metric_results"]
    assert "early_score_delta_vs_baseline" in metrics
    assert "early_rel_mse_delta_vs_baseline" in metrics
    assert "middle_score_guardrail" in metrics
    assert "late_rmse_guardrail" in metrics
    multiview = sandbox["verifier"]["multiview_judge"]
    assert "early_accuracy" in multiview["views"]
    assert "early_rel_mse_delta" in multiview["views"]["early_accuracy"]


def test_hypothesis_lab_pareto_gate_promotes_dominating_candidate() -> None:
    from pde_agent.core.hypothesis_lab import _sandbox_pareto_promotion_gate

    metric_results = {
        "rules_passed": {"value": True, "passed": True},
        "context_preserved": {"value": True, "passed": True},
        "score_delta_vs_baseline": {
            "value": 0.25,
            "threshold": 0.01,
            "passed": True,
        },
        "segment_score_delta_vs_baseline": {"value": 0.2, "passed": True},
        "early_rel_mse_delta_vs_baseline": {"value": 0.03, "passed": True},
        "middle_score_guardrail": {"value": 0.01, "passed": True},
        "late_rmse_guardrail": {"value": 0.02, "passed": True},
        "inference_time_budget": {
            "value": 0.4,
            "threshold": 120.0,
            "passed": True,
        },
    }

    gate = _sandbox_pareto_promotion_gate(
        family="task1_early_refinement",
        metric_results=metric_results,
        selection_objective={"primary": "early_rel_mse", "direction": "minimize"},
        candidate_budget=80,
        promotion_score=0.26,
    )

    assert gate["decision"] == "promote_to_strategy_proposal"
    assert gate["promote_to_strategy_proposal"] is True
    assert gate["pareto_status"] == "dominates_baseline"
    assert gate["objective_axes"]["early_rel_mse"]["improved"] is True
    assert gate["objective_axes"]["task1_total_score_proxy"]["passed"] is True
    assert gate["objective_axes"]["runtime_seconds"]["improved"] is False
    assert gate["objective_axes"]["candidate_budget"]["value"] == 80


def test_hypothesis_lab_pareto_gate_keeps_primary_axis_tradeoff() -> None:
    from pde_agent.core.hypothesis_lab import _sandbox_pareto_promotion_gate

    metric_results = {
        "rules_passed": {"value": True, "passed": True},
        "context_preserved": {"value": True, "passed": True},
        "score_delta_vs_baseline": {
            "value": -0.03,
            "threshold": 0.01,
            "passed": False,
        },
        "segment_score_delta_vs_baseline": {"value": -0.02, "passed": False},
        "early_rel_mse_delta_vs_baseline": {"value": 0.04, "passed": True},
        "middle_score_guardrail": {"value": 0.0, "passed": True},
        "late_rmse_guardrail": {"value": 0.0, "passed": True},
        "inference_time_budget": {
            "value": 0.2,
            "threshold": 120.0,
            "passed": True,
        },
    }

    gate = _sandbox_pareto_promotion_gate(
        family="task1_early_refinement",
        metric_results=metric_results,
        selection_objective={"primary": "early_rel_mse", "direction": "minimize"},
        candidate_budget=80,
        promotion_score=-0.03,
    )

    assert gate["decision"] == "keep_as_evidence"
    assert gate["promote_to_strategy_proposal"] is False
    assert gate["pareto_status"] == "tradeoff_evidence"
    assert "primary_axis_improved_without_total_score" in gate["reason_codes"]
    assert gate["next_command"] == "revise-hypothesis"


def test_hypothesis_lab_pareto_gate_continues_on_guardrail_regression() -> None:
    from pde_agent.core.hypothesis_lab import _sandbox_pareto_promotion_gate

    metric_results = {
        "rules_passed": {"value": True, "passed": True},
        "context_preserved": {"value": True, "passed": True},
        "score_delta_vs_baseline": {
            "value": -0.01,
            "threshold": 0.01,
            "passed": False,
        },
        "segment_score_delta_vs_baseline": {"value": -0.01, "passed": False},
        "early_rel_mse_delta_vs_baseline": {"value": 0.05, "passed": True},
        "middle_score_guardrail": {"value": -0.1, "passed": False},
        "late_rmse_guardrail": {"value": 0.0, "passed": True},
        "inference_time_budget": {
            "value": 0.2,
            "threshold": 120.0,
            "passed": True,
        },
    }

    gate = _sandbox_pareto_promotion_gate(
        family="task1_early_refinement",
        metric_results=metric_results,
        selection_objective={"primary": "early_rel_mse", "direction": "minimize"},
        candidate_budget=80,
        promotion_score=-0.01,
    )

    assert gate["decision"] == "continue_local_search"
    assert gate["promote_to_strategy_proposal"] is False
    assert gate["pareto_status"] == "tradeoff_needs_local_search"
    assert "guardrail_regression" in gate["reason_codes"]


def test_hypothesis_lab_sandbox_executes_task1_early_refinement_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.2, 0.4]):
        for step in range(200):
            target[sample, step, :] = (
                offset
                + np.float32(0.01 * step)
                + np.float32(0.1) * np.sin(x + np.float32(0.01 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": [
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 12,
                    "persistence_weight_start": 0.0,
                    "spectral_strength": 0.0,
                    "early_anchor": "retrieval",
                },
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 12,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "linear_velocity",
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["blueprint_id"] == "bp_task1_early_refinement_001"
    assert sandbox["candidate_family"] == "task1_early_refinement"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_early_refinement"
    )
    assert sandbox["best_candidate"]["metadata"]["early_objective"] == "early_rel_mse"
    selection = sandbox["verifier"]["selection_objective"]
    assert selection["primary"] == "early_rel_mse"
    gate = sandbox["verifier"]["pareto_promotion_gate"]
    assert gate["state_hint"] == "PARETO_PROMOTION_GATE_EVALUATED"
    assert gate["objective_axes"]["early_rel_mse"]["metric"] == (
        "early_rel_mse_delta_vs_baseline"
    )
    assert gate["objective_axes"]["runtime_seconds"]["passed"] is True
    assert gate["objective_axes"]["runtime_seconds"]["improved"] is False
    best_rel_mse = sandbox["best_candidate"]["segments"]["early"]["rel_mse"]
    top_rel_mses = [
        row["segments"]["early"]["rel_mse"] for row in sandbox["top_candidates"]
    ]
    assert best_rel_mse == min(top_rel_mses)
    diagnosis = sandbox["verifier"]["early_segment_diagnosis"]
    attribution = diagnosis["candidate_trend_attribution"]
    assert attribution["primary_metric"] == "early_rel_mse"
    assert "early_anchor" in attribution["axes"]
    assert attribution["axes"]["early_anchor"]["best_values"]
    assert attribution["recommended_axis"] in attribution["axes"]
    assert sandbox["verifier"]["early_segment_diagnosis"]["objective_recommendation"][
        "next_candidate_family"
    ] == "task1_early_refinement"


def test_hypothesis_lab_sandbox_executes_derivative_aware_early_anchors(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.1, 0.2]):
        for step in range(200):
            target[sample, step, :] = (
                np.float32(offset)
                + np.float32(0.003 * step)
                + np.float32(0.15) * np.sin(x + np.float32(0.01 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": [
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 16,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "damped_linear_velocity",
                },
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 16,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "retrieval_derivative_match",
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    anchors = {row["params"]["early_anchor"] for row in sandbox["top_candidates"]}
    assert anchors == {"damped_linear_velocity", "retrieval_derivative_match"}
    assert sandbox["best_candidate"]["metadata"]["early_anchor"] in anchors
    assert sandbox["verifier"]["selection_objective"]["primary"] == "early_rel_mse"


def test_hypothesis_lab_sandbox_executes_scaled_damped_velocity_anchor(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.1, 0.2]):
        for step in range(200):
            target[sample, step, :] = (
                np.float32(offset)
                + np.float32(0.001 * step)
                + np.float32(0.13) * np.sin(x + np.float32(0.01 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": [
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 16,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "scaled_damped_velocity",
                    "velocity_scale": 0.25,
                },
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 16,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "scaled_damped_velocity",
                    "velocity_scale": 0.5,
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    scales = {
        row["metadata"]["velocity_scale"] for row in sandbox["top_candidates"]
    }
    assert scales == {0.25, 0.5}
    assert sandbox["best_candidate"]["metadata"]["early_anchor"] == (
        "scaled_damped_velocity"
    )


def test_hypothesis_lab_sandbox_executes_phase_bias_and_curved_handoff(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.15, 0.3]):
        for step in range(200):
            target[sample, step, :] = (
                np.float32(offset)
                + np.float32(0.002 * step)
                + np.float32(0.12) * np.sin(x + np.float32(0.03 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": [
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 18,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "phase_aligned_retrieval",
                    "phase_shift_scale": 0.5,
                    "handoff_curve": "cosine",
                },
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 18,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "mean_trend_bias",
                    "handoff_curve": "quadratic",
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    anchors = {row["metadata"]["early_anchor"] for row in sandbox["top_candidates"]}
    assert anchors == {"phase_aligned_retrieval", "mean_trend_bias"}
    curves = {row["metadata"]["handoff_curve"] for row in sandbox["top_candidates"]}
    assert curves == {"cosine", "quadratic"}
    assert sandbox["best_candidate"]["metadata"]["early_anchor"] in anchors
    assert sandbox["verifier"]["pareto_promotion_gate"]["objective_axes"][
        "early_rel_mse"
    ]["metric"] == "early_rel_mse_delta_vs_baseline"


def test_hypothesis_lab_sandbox_executes_compound_calibration_anchors(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.1, 0.2]):
        for step in range(200):
            amplitude = np.float32(0.1 + 0.0004 * step)
            target[sample, step, :] = (
                np.float32(offset)
                + np.float32(0.0025 * step)
                + amplitude * np.sin(x + np.float32(0.02 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "search_mode": "early_segment_objective_search",
        "parameter_grid": {
            "candidate_params": [
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 18,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "affine_context_calibration",
                    "handoff_curve": "cosine",
                },
                {
                    "k": 1,
                    "mode": "affine",
                    "early_blend_steps": 18,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.0,
                    "early_anchor": "phase_derivative_hybrid",
                    "phase_shift_scale": 0.5,
                    "handoff_curve": "linear",
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    anchors = {row["metadata"]["early_anchor"] for row in sandbox["top_candidates"]}
    assert anchors == {"affine_context_calibration", "phase_derivative_hybrid"}
    assert sandbox["best_candidate"]["metadata"]["early_anchor"] in anchors


def test_hypothesis_lab_candidate_pool_attribution_marks_non_top_anchors() -> None:
    from pde_agent.core.hypothesis_lab import _candidate_pool_attribution

    rows = [
        {
            "candidate_id": f"persistence_{index}",
            "params": {"early_anchor": "persistence"},
            "segment_score": 50.0 - index * 0.1,
            "task1_total_score_proxy": 50.0 - index * 0.1,
            "segments": {"early": {"rel_mse": 0.1 + index * 0.001}},
        }
        for index in range(20)
    ]
    rows.extend(
        [
            {
                "candidate_id": "scaled_bad_025",
                "params": {
                    "early_anchor": "scaled_damped_velocity",
                    "velocity_scale": 0.25,
                },
                "segment_score": 20.0,
                "task1_total_score_proxy": 20.0,
                "segments": {"early": {"rel_mse": 2.0}},
            },
            {
                "candidate_id": "scaled_bad_050",
                "params": {
                    "early_anchor": "scaled_damped_velocity",
                    "velocity_scale": 0.5,
                },
                "segment_score": 19.0,
                "task1_total_score_proxy": 19.0,
                "segments": {"early": {"rel_mse": 2.5}},
            },
        ]
    )

    attribution = _candidate_pool_attribution(
        rows=rows,
        top_candidate_count=20,
        family="task1_early_refinement",
    )

    assert attribution["axis"] == "early_anchor"
    assert attribution["candidate_count"] == 22
    assert attribution["non_top_anchors"] == ["scaled_damped_velocity"]
    scaled = attribution["by_early_anchor"]["scaled_damped_velocity"]
    assert scaled["candidate_count"] == 2
    assert scaled["top_candidate_count"] == 0
    assert scaled["best_early_rel_mse"] == 2.0
    assert scaled["best_candidate_id"] == "scaled_bad_025"


def test_hypothesis_lab_candidate_pool_policy_separates_rescue_from_deprioritize() -> None:
    from pde_agent.core.hypothesis_lab import _candidate_pool_attribution

    rows = [
        {
            "candidate_id": f"persistence_{index}",
            "params": {"early_anchor": "persistence"},
            "segment_score": 50.0 - index * 0.01,
            "task1_total_score_proxy": 50.0 - index * 0.01,
            "segments": {"early": {"rel_mse": 0.1 + index * 0.0001}},
        }
        for index in range(20)
    ]
    rows.extend(
        [
            {
                "candidate_id": "scaled_close",
                "params": {"early_anchor": "scaled_damped_velocity"},
                "segment_score": 49.9,
                "task1_total_score_proxy": 49.9,
                "segments": {"early": {"rel_mse": 0.105}},
            },
            {
                "candidate_id": "second_order_bad",
                "params": {"early_anchor": "second_order_velocity"},
                "segment_score": 20.0,
                "task1_total_score_proxy": 20.0,
                "segments": {"early": {"rel_mse": 3.0}},
            },
        ]
    )

    attribution = _candidate_pool_attribution(
        rows=rows,
        top_candidate_count=20,
        family="task1_early_refinement",
    )

    policy = attribution["search_space_policy"]
    assert policy["axis"] == "early_anchor"
    assert policy["focus_values"] == ["persistence"]
    assert policy["rescue_probe_values"] == ["scaled_damped_velocity"]
    assert policy["deprioritize_values"] == ["second_order_velocity"]
    assert {
        item["action"]: item["value"] for item in policy["recommended_actions"]
    } == {
        "keep_focus": "persistence",
        "budget_probe": "scaled_damped_velocity",
        "deprioritize": "second_order_velocity",
    }


def test_hypothesis_lab_sandbox_executes_retrieval_prior_leave_one_out(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.2, 0.4]):
        for step in range(200):
            target[sample, step, :] = (
                offset
                + np.float32(0.01 * step)
                + np.float32(0.1) * np.sin(x + np.float32(0.01 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_retrieval_prior_001",
        "candidate_family": "task1_retrieval_prior",
        "search_mode": "affine_knn_trajectory_prior",
        "parameter_grid": {"k": [1, 2], "mode": ["affine"]},
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["best_candidate"]["candidate_id"].startswith("task1_retrieval_prior")
    assert sandbox["best_candidate"]["params"]["mode"] == "affine"
    assert sandbox["verifier"]["score_delta_vs_baseline"] > 0.0
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]
    assert sandbox["best_candidate"]["metadata"]["leave_one_out"] is True


def test_hypothesis_lab_sandbox_executes_retrieval_refinement_leave_one_out(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((3, 200, 256), dtype=np.float32)
    for sample, offset in enumerate([0.0, 0.2, 0.4]):
        for step in range(200):
            target[sample, step, :] = (
                offset
                + np.float32(0.01 * step)
                + np.float32(0.1) * np.sin(x + np.float32(0.01 * step))
            )
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_retrieval_refinement_001",
        "candidate_family": "task1_retrieval_refinement",
        "search_mode": "horizon_blended_retrieval_prior",
        "parameter_grid": {
            "k": [1, 2],
            "mode": ["affine", "delta"],
            "early_blend_steps": [0, 20],
            "persistence_weight_start": [0.0, 0.5],
            "spectral_strength": [0.0],
        },
        "sandbox": {
            "promotion_min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_retrieval_refinement"
    )
    assert sandbox["verifier"]["score_delta_vs_baseline"] > 0.0
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]
    metadata = sandbox["best_candidate"]["metadata"]
    assert metadata["leave_one_out"] is True
    assert metadata["reference_count"] == 3
    assert set(metadata) >= {
        "k",
        "mode",
        "early_blend_steps",
        "persistence_weight_start",
        "spectral_strength",
    }


def test_hypothesis_lab_sandbox_executes_sem_burgers_rollout_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    target = np.full((2, 200, 256), 0.25, dtype=np.float32)
    for step in range(10):
        target[:, step, :] = np.float32(0.25 + 0.002 * step)
    target[:, 10:, :] = target[:, 9:10, :]
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_sem_burgers_rollout_001",
        "candidate_family": "task1_sem_burgers_rollout",
        "search_mode": "physics_numeric_rollout",
        "parameter_grid": {
            "candidate_params": [
                {
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "element_count": 16,
                    "polynomial_order": 8,
                    "substeps": 2,
                    "filter_strength": 0.0,
                }
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["candidate_family"] == "task1_sem_burgers_rollout"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_sem_burgers_rollout"
    )
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]
    assert sandbox["best_candidate"]["metadata"] == {
        "executor": "sem_burgers_rollout",
        "viscosity": 0.001,
        "dt": 0.05,
        "element_count": 16,
        "polynomial_order": 8,
        "substeps": 2,
        "filter_strength": 0.0,
    }


def test_hypothesis_lab_sandbox_executes_physics_residual_refinement_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    target = np.full((2, 200, 256), 0.25, dtype=np.float32)
    for step in range(10):
        target[:, step, :] = np.float32(0.25 + 0.002 * step)
    target[:, 10:, :] = target[:, 9:10, :]
    context = target[:, :10, :]
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_physics_residual_refinement_001",
        "candidate_family": "task1_physics_residual_refinement",
        "search_mode": "physics_residual_refinement",
        "parameter_grid": {
            "candidate_params": [
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": -0.2,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                    "correction_weight": 0.5,
                    "stop_step": 80,
                    "residual_spectral_strength": 0.0,
                }
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["candidate_family"] == "task1_physics_residual_refinement"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_physics_residual_refinement"
    )
    metadata = sandbox["best_candidate"]["metadata"]
    assert metadata["executor"] == "residual_guided_refinement"
    assert metadata["candidate_residual_rmse"] < metadata["base_residual_rmse"]
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]


def test_hypothesis_lab_physics_residual_noop_sentinel_blocks_false_promotion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    context = np.full((1, 10, 256), 0.25, dtype=np.float32)
    for step in range(10):
        context[:, step, :] = np.float32(0.25 + 0.002 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    slope = (context[:, -1:, :] - context[:, -3:-2, :]) / np.float32(2.0)
    horizon = np.arange(1, 191, dtype=np.float32).reshape(1, -1, 1)
    horizon = horizon * np.exp(-horizon / np.float32(100.0))
    target[:, 10:, :] = context[:, -1:, :] - np.float32(0.2) * horizon * slope
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_physics_residual_refinement_001",
        "candidate_family": "task1_physics_residual_refinement",
        "search_mode": "physics_residual_refinement",
        "parameter_grid": {
            "candidate_params": [
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": -0.2,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                    "correction_weight": 0.0,
                    "stop_step": 80,
                    "residual_spectral_strength": 0.0,
                },
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": -0.2,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                    "correction_weight": 0.5,
                    "stop_step": 80,
                    "residual_spectral_strength": 0.0,
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert "_cw0_" in sandbox["best_candidate"]["candidate_id"]
    assert sandbox["verifier"]["decision"] != "promote_to_strategy_proposal"
    assert "physics_residual_not_improved" in sandbox["verifier"][
        "pareto_promotion_gate"
    ]["reason_codes"]


def test_hypothesis_lab_sandbox_executes_pi_deeponet_correction_head_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    x = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    context = np.empty((4, 10, 256), dtype=np.float32)
    for sample in range(4):
        amplitude = np.float32(0.2 + 0.05 * sample)
        for step in range(10):
            context[sample, step, :] = 0.3 + amplitude * np.sin(
                2.0 * np.pi * (x + 0.01 * step)
            )
    target = np.repeat(context[:, -1:, :], 200, axis=1).astype(np.float32)
    target[:, :10, :] = context
    horizon = np.linspace(0.0, 1.0, 190, dtype=np.float32).reshape(1, 190, 1)
    spatial = np.sin(2.0 * np.pi * x).reshape(1, 1, 256)
    sample_weight = np.mean(context[:, -1:, :], axis=2, keepdims=True)
    target[:, 10:, :] += np.float32(0.08) * sample_weight * horizon * spatial
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_pi_deeponet_correction_head_001",
        "candidate_family": "task1_pi_deeponet_correction_head",
        "search_mode": "pi_deeponet_correction_head",
        "parameter_grid": {
            "candidate_params": [
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": 0.0,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "ridge_alpha": 1e-6,
                    "correction_weight": 1.0,
                    "physics_weight": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                }
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["candidate_family"] == "task1_pi_deeponet_correction_head"
    assert sandbox["best_candidate"]["candidate_id"].startswith(
        "task1_pi_deeponet_correction_head"
    )
    metadata = sandbox["best_candidate"]["metadata"]
    assert metadata["adapter_type"] == "numpy_pi_deeponet_correction_head"
    assert metadata["train_mode"] == "ridge_branch_trunk_correction"
    assert sandbox["verifier"]["metric_results"]["context_preserved"]["passed"]
    assert sandbox["verifier"]["score_delta_vs_baseline"] > 0.0


def test_hypothesis_lab_pi_deeponet_noop_sentinel_blocks_false_promotion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_blueprint_sandbox

    context = np.full((1, 10, 256), 0.25, dtype=np.float32)
    for step in range(10):
        context[:, step, :] = np.float32(0.25 + 0.001 * step)
    target = np.repeat(context[:, -1:, :], 200, axis=1).astype(np.float32)
    target[:, :10, :] = context
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    blueprint = {
        "blueprint_id": "bp_task1_pi_deeponet_correction_head_001",
        "candidate_family": "task1_pi_deeponet_correction_head",
        "search_mode": "pi_deeponet_correction_head",
        "parameter_grid": {
            "candidate_params": [
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": 0.0,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "ridge_alpha": 1e-6,
                    "correction_weight": 0.0,
                    "physics_weight": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                },
                {
                    "base_family": "linear_trend_spectral",
                    "slope_window": 2,
                    "gain": 0.0,
                    "decay": 100.0,
                    "spectral_strength": 0.0,
                    "ridge_alpha": 1e-6,
                    "correction_weight": 1.0,
                    "physics_weight": 0.0,
                    "viscosity": 0.001,
                    "dt": 0.05,
                    "dx": 1.0 / 256.0,
                },
            ]
        },
        "sandbox": {
            "promotion_min_score_delta": 0.0,
            "max_inference_seconds": 120.0,
        },
    }

    sandbox = run_blueprint_sandbox(
        blueprint=blueprint,
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "sandbox",
        submission_id="team_hermes",
        force=False,
    )

    assert sandbox["verifier"]["decision"] != "promote_to_strategy_proposal"
    assert "pi_deeponet_correction_inactive" in sandbox["verifier"][
        "pareto_promotion_gate"
    ]["reason_codes"]


def test_hypothesis_lab_run_writes_verified_blueprint_seeded_by_sandbox_best(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab

    context = np.ones((1, 10, 256), dtype=np.float32)
    target = np.concatenate(
        [context, np.repeat(context[:, -1:, :], 190, axis=1)],
        axis=1,
    )
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_hypothesis_lab(
        manual_report_paths=[_manual_report(tmp_path / "manual_report.json")],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        force=False,
    )

    verified_path = Path(report["artifacts"]["verified_blueprint_path"])
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    assert verified["state_hint"] == "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED"
    assert verified["blueprints"][0]["seed_params"][0]["source"] == "sandbox_best"
    assert verified["blueprints"][0]["seed_params"][0]["params"] == report["sandbox"][
        "best_candidate"
    ]["params"]


def test_hypothesis_lab_records_algorithm_candidate_outcome(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab

    context = np.ones((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, 9, :] - np.float32(0.002 * (step + 1))
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    memory_path = tmp_path / "failure_memory.json"

    report = run_hypothesis_lab(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _local_asset_algorithm_candidates(tmp_path / "local_asset_candidates.json")
        ],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        failure_memory_output_path=memory_path,
        force=False,
    )

    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert report["candidate_outcome"]["algorithm_candidate_id"] == (
        "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
    )
    assert memory["outcomes"][0]["outcome"] == "passed"
    assert memory["outcomes"][0]["algorithm_candidate_id"] == (
        "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
    )
    assert memory["failed_algorithm_candidate_ids"] == []
    event = memory["pareto_learning_events"][0]
    assert event["algorithm_candidate_id"] == (
        "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
    )
    assert event["decision"] == "promote_to_strategy_proposal"
    assert event["pareto_status"] == "dominates_baseline"
    assert event["outcome"] == "promoted"
    assert "task1_total_score_proxy" in event["objective_axes"]
    assert report["pareto_memory"]["event_count"] == 1


def test_hypothesis_lab_records_pareto_failure_memory_tradeoff(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import _record_pareto_failure_memory

    sandbox = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "best_candidate": {
            "candidate_id": "task1_early_refinement_k99_affine_eb28_pw0.5",
            "params": {
                "k": 99,
                "mode": "affine",
                "early_blend_steps": 28,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
                "early_anchor": "persistence",
            },
        },
        "verifier": {
            "decision": "keep_as_evidence",
            "score_delta_vs_baseline": -0.03,
            "best_segment_score": 47.7,
            "pareto_promotion_gate": {
                "decision": "keep_as_evidence",
                "pareto_status": "tradeoff_evidence",
                "reason_codes": ["primary_axis_improved_without_total_score"],
                "objective_axes": {
                    "task1_total_score_proxy": {
                        "value": -0.03,
                        "passed": False,
                        "improved": False,
                    },
                    "early_rel_mse": {
                        "value": 0.04,
                        "passed": True,
                        "improved": True,
                    },
                    "middle_score": {
                        "value": 0.0,
                        "passed": True,
                        "improved": False,
                    },
                    "late_rmse": {
                        "value": 0.0,
                        "passed": True,
                        "improved": False,
                    },
                },
            },
            "early_segment_diagnosis": {
                "bottleneck_type": "phase_shift",
                "reason_codes": ["early_rel_mse_high"],
                "action_directives": [
                    {
                        "directive": (
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ),
                        "target_family": "task1_early_refinement",
                    }
                ],
            },
        },
        "artifacts": {
            "sandbox_report_path": str(tmp_path / "sandbox_report.json"),
        },
    }

    report = _record_pareto_failure_memory(
        sandbox=sandbox,
        failure_memory_output_path=tmp_path / "failure_memory.json",
        algorithm_candidate_evidence={
            "algorithm_candidate_id": "candidate_early_phase_probe",
            "candidate_key": "card_phase|task1_early_refinement",
        },
        force=False,
    )

    memory = json.loads(
        Path(report["failure_memory_path"]).read_text(encoding="utf-8")
    )
    event = memory["pareto_learning_events"][0]
    assert event["outcome"] == "evidence_only"
    assert event["algorithm_candidate_id"] == "candidate_early_phase_probe"
    assert event["candidate_family"] == "task1_early_refinement"
    assert event["bottleneck_type"] == "phase_shift"
    assert event["effective_axes"] == ["early_rel_mse"]
    assert event["guardrail_regressions"] == []
    assert event["retry_directives"] == ["continue_guarded_local_search"]
    assert event["avoid_directives"] == [
        "do_not_promote_primary_axis_without_total_score"
    ]
    summary = memory["pareto_learning_summary"]["task1_early_refinement"]
    assert summary["evidence_only_count"] == 1
    assert summary["last_pareto_status"] == "tradeoff_evidence"


def test_hypothesis_lab_records_candidate_pool_negative_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import _record_pareto_failure_memory

    sandbox = {
        "blueprint_id": "bp_task1_early_refinement_001",
        "candidate_family": "task1_early_refinement",
        "best_candidate": {
            "candidate_id": "task1_early_refinement_persistence",
            "params": {
                "k": 99,
                "mode": "affine",
                "early_blend_steps": 28,
                "persistence_weight_start": 0.5,
                "spectral_strength": 0.5,
                "early_anchor": "persistence",
            },
        },
        "candidate_pool_attribution": {
            "axis": "early_anchor",
            "candidate_count": 24,
            "top_candidate_count": 20,
            "non_top_anchors": ["scaled_damped_velocity"],
            "by_early_anchor": {
                "persistence": {
                    "candidate_count": 20,
                    "top_candidate_count": 20,
                    "best_early_rel_mse": 0.1,
                    "best_segment_score": 47.8,
                    "best_candidate_id": "task1_early_refinement_persistence",
                },
                "scaled_damped_velocity": {
                    "candidate_count": 4,
                    "top_candidate_count": 0,
                    "best_early_rel_mse": 2.0,
                    "best_segment_score": 20.0,
                    "best_candidate_id": "scaled_bad_025",
                },
            },
        },
        "verifier": {
            "decision": "promote_to_strategy_proposal",
            "score_delta_vs_baseline": 1.0,
            "best_segment_score": 47.8,
            "pareto_promotion_gate": {
                "decision": "promote_to_strategy_proposal",
                "pareto_status": "dominates_baseline",
                "reason_codes": [],
                "objective_axes": {
                    "early_rel_mse": {
                        "value": -0.1,
                        "threshold": 0.0,
                        "passed": True,
                        "improved": True,
                        "metric": "early_rel_mse_delta_vs_baseline",
                    }
                },
            },
            "early_segment_diagnosis": {
                "bottleneck_type": "initial_derivative_mismatch",
                "reason_codes": ["early_rel_mse_high"],
                "action_directives": [
                    {
                        "directive": "expand_linear_velocity_anchor_search",
                        "target_family": "task1_early_refinement",
                    }
                ],
            },
        },
        "artifacts": {
            "sandbox_report_path": str(tmp_path / "sandbox_report.json"),
        },
    }

    report = _record_pareto_failure_memory(
        sandbox=sandbox,
        failure_memory_output_path=tmp_path / "failure_memory.json",
        algorithm_candidate_evidence=None,
        force=False,
    )

    memory = json.loads(
        Path(report["failure_memory_path"]).read_text(encoding="utf-8")
    )
    event = memory["pareto_learning_events"][0]
    assert event["candidate_pool_attribution"]["non_top_anchors"] == [
        "scaled_damped_velocity"
    ]
    assert event["negative_candidate_evidence"] == [
        {
            "axis": "early_anchor",
            "value": "scaled_damped_velocity",
            "reason": "generated_but_absent_from_top_candidates",
            "candidate_count": 4,
            "top_candidate_count": 0,
            "best_early_rel_mse": 2.0,
            "best_segment_score": 20.0,
        }
    ]
    summary = memory["pareto_learning_summary"]["task1_early_refinement"]
    assert summary["ineffective_anchors"] == ["scaled_damped_velocity"]


def test_hypothesis_lab_reads_pareto_memory_into_knowledge_cards(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "schema_version": 1,
                        "source": "hypothesis_lab_sandbox",
                        "blueprint_id": "bp_task1_early_refinement_001",
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_phase_probe",
                        "candidate_params": {
                            "k": 92,
                            "mode": "affine",
                            "early_blend_steps": 24,
                            "persistence_weight_start": 0.55,
                            "spectral_strength": 0.5,
                            "early_anchor": "phase_aligned_retrieval",
                        },
                        "decision": "keep_as_evidence",
                        "pareto_status": "tradeoff_evidence",
                        "outcome": "evidence_only",
                        "reason_codes": [
                            "primary_axis_improved_without_total_score"
                        ],
                        "bottleneck_type": "phase_shift",
                        "diagnosis_reason_codes": ["early_rel_mse_high"],
                        "action_directives": [
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ],
                        "effective_axes": ["early_rel_mse"],
                        "guardrail_regressions": [],
                        "retry_directives": ["continue_guarded_local_search"],
                        "avoid_directives": [
                            "do_not_promote_primary_axis_without_total_score"
                        ],
                        "objective_axes": {
                            "early_rel_mse": {"improved": True, "passed": True},
                            "task1_total_score_proxy": {
                                "improved": False,
                                "passed": False,
                            },
                        },
                    }
                ],
                "pareto_learning_summary": {
                    "task1_early_refinement": {
                        "evidence_only_count": 1,
                        "promoted_count": 0,
                        "needs_local_search_count": 0,
                        "rejected_count": 0,
                        "effective_axes": ["early_rel_mse"],
                        "guardrail_regressions": [],
                        "last_pareto_status": "tradeoff_evidence",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        output_path=None,
        force=False,
    )

    assert cards["card_count"] == 1
    card = cards["cards"][0]
    assert card["source_type"] == "pareto_learning_memory"
    assert card["implementation_hooks"] == ["task1_early_refinement"]
    assert card["evidence_status"] == "evidence_only"
    assert card["evidence"]["params"]["early_anchor"] == "phase_aligned_retrieval"
    assert card["evidence"]["early_segment_diagnosis"]["bottleneck_type"] == (
        "phase_shift"
    )
    assert card["evidence"]["early_segment_diagnosis"]["action_directives"][0][
        "directive"
    ] == "add_phase_alignment_or_derivative_distance_candidates"
    assert "do_not_promote_primary_axis_without_total_score" in card["known_risks"]


def test_hypothesis_lab_memory_readback_changes_early_candidate_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_phase_probe",
                        "candidate_params": {
                            "k": 92,
                            "mode": "affine",
                            "early_blend_steps": 24,
                            "persistence_weight_start": 0.55,
                            "spectral_strength": 0.5,
                            "early_anchor": "phase_aligned_retrieval",
                        },
                        "decision": "keep_as_evidence",
                        "pareto_status": "tradeoff_evidence",
                        "outcome": "evidence_only",
                        "bottleneck_type": "phase_shift",
                        "action_directives": [
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ],
                        "effective_axes": ["early_rel_mse"],
                        "guardrail_regressions": [],
                        "retry_directives": ["continue_guarded_local_search"],
                        "avoid_directives": [
                            "do_not_promote_primary_axis_without_total_score"
                        ],
                    }
                ],
                "pareto_learning_summary": {
                    "task1_early_refinement": {
                        "evidence_only_count": 1,
                        "effective_axes": ["early_rel_mse"],
                        "last_pareto_status": "tradeoff_evidence",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cards = build_knowledge_cards(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    early = blueprints["blueprints"][0]
    params = early["parameter_grid"]["candidate_params"]
    anchors = {row["early_anchor"] for row in params}
    assert "phase_aligned_retrieval" in anchors
    assert any(row.get("phase_shift_scale") == 0.5 for row in params)
    assert early["seed_params"][0]["source_kind"] == "pareto_memory"
    assert early["experiment_plan"]["axis_plan_by_bottleneck"][
        "phase_alignment"
    ]["reason"] == "phase_shift diagnosis requires transport-aware early anchors"


def test_hypothesis_lab_memory_directives_filter_repeated_evidence_only_params(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    repeated_params = {
        "k": 92,
        "mode": "affine",
        "early_blend_steps": 24,
        "persistence_weight_start": 0.55,
        "spectral_strength": 0.5,
        "early_anchor": "phase_aligned_retrieval",
    }
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_phase_probe",
                        "candidate_params": repeated_params,
                        "decision": "keep_as_evidence",
                        "pareto_status": "tradeoff_evidence",
                        "outcome": "evidence_only",
                        "bottleneck_type": "phase_shift",
                        "action_directives": [
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ],
                        "effective_axes": ["early_rel_mse"],
                        "retry_directives": ["continue_guarded_local_search"],
                        "avoid_directives": [
                            "do_not_promote_primary_axis_without_total_score"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    early = blueprints["blueprints"][0]
    candidate_params = early["parameter_grid"]["candidate_params"]
    assert repeated_params not in [
        {key: row[key] for key in repeated_params}
        for row in candidate_params
        if all(key in row for key in repeated_params)
    ]
    assert any(
        row["early_anchor"] == "phase_aligned_retrieval"
        for row in candidate_params
    )
    assert any(
        row["early_anchor"] == "phase_aligned_retrieval"
        and row["persistence_weight_start"]
        < repeated_params["persistence_weight_start"]
        for row in candidate_params
    )
    memory_policy = early["experiment_plan"]["memory_policy"]
    assert memory_policy["avoid_exact_count"] == 1
    assert "continue_guarded_local_search" in memory_policy["retry_directives"]
    assert "do_not_promote_primary_axis_without_total_score" in memory_policy[
        "avoid_directives"
    ]


def test_hypothesis_lab_memory_deprioritizes_non_top_anchors(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_persistence",
                        "candidate_params": {
                            "k": 99,
                            "mode": "affine",
                            "early_blend_steps": 28,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.5,
                            "early_anchor": "persistence",
                        },
                        "decision": "promote_to_strategy_proposal",
                        "pareto_status": "dominates_baseline",
                        "outcome": "promoted",
                        "bottleneck_type": "initial_derivative_mismatch",
                        "action_directives": [
                            "expand_linear_velocity_anchor_search"
                        ],
                        "effective_axes": ["early_rel_mse"],
                        "candidate_pool_attribution": {
                            "axis": "early_anchor",
                            "candidate_count": 24,
                            "top_candidate_count": 20,
                            "non_top_anchors": ["scaled_damped_velocity"],
                        },
                        "negative_candidate_evidence": [
                            {
                                "axis": "early_anchor",
                                "value": "scaled_damped_velocity",
                                "reason": (
                                    "generated_but_absent_from_top_candidates"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    params = blueprints["blueprints"][0]["parameter_grid"]["candidate_params"]
    first_scaled_index = next(
        index
        for index, row in enumerate(params)
        if row["early_anchor"] == "scaled_damped_velocity"
    )
    first_persistence_index = next(
        index
        for index, row in enumerate(params)
        if row["early_anchor"] == "persistence"
    )
    assert first_scaled_index > first_persistence_index
    memory_policy = blueprints["blueprints"][0]["experiment_plan"]["memory_policy"]
    assert memory_policy["deprioritized_anchors"] == ["scaled_damped_velocity"]


def test_hypothesis_lab_memory_uses_candidate_pool_policy_for_rescue_probes(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import (
        build_knowledge_cards,
        design_algorithm_blueprints,
        generate_hypotheses,
    )

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_persistence",
                        "candidate_params": {
                            "k": 99,
                            "mode": "affine",
                            "early_blend_steps": 28,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.5,
                            "early_anchor": "persistence",
                        },
                        "decision": "promote_to_strategy_proposal",
                        "pareto_status": "dominates_baseline",
                        "outcome": "promoted",
                        "bottleneck_type": "initial_derivative_mismatch",
                        "action_directives": [
                            "expand_linear_velocity_anchor_search"
                        ],
                        "effective_axes": ["early_rel_mse"],
                        "candidate_pool_attribution": {
                            "axis": "early_anchor",
                            "candidate_count": 24,
                            "top_candidate_count": 20,
                            "non_top_anchors": [
                                "scaled_damped_velocity",
                                "second_order_velocity",
                            ],
                            "search_space_policy": {
                                "axis": "early_anchor",
                                "focus_values": ["persistence"],
                                "rescue_probe_values": [
                                    "scaled_damped_velocity"
                                ],
                                "deprioritize_values": [
                                    "second_order_velocity"
                                ],
                                "recommended_actions": [],
                            },
                        },
                        "negative_candidate_evidence": [
                            {
                                "axis": "early_anchor",
                                "value": "scaled_damped_velocity",
                                "reason": (
                                    "generated_but_absent_from_top_candidates"
                                ),
                            },
                            {
                                "axis": "early_anchor",
                                "value": "second_order_velocity",
                                "reason": (
                                    "generated_but_absent_from_top_candidates"
                                ),
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    blueprints = design_algorithm_blueprints(
        hypotheses=hypotheses,
        output_path=None,
        force=False,
    )

    memory_policy = blueprints["blueprints"][0]["experiment_plan"]["memory_policy"]
    assert memory_policy["rescue_probe_anchors"] == ["scaled_damped_velocity"]
    assert memory_policy["deprioritized_anchors"] == ["second_order_velocity"]
    params = blueprints["blueprints"][0]["parameter_grid"]["candidate_params"]
    first_scaled_index = next(
        index
        for index, row in enumerate(params)
        if row["early_anchor"] == "scaled_damped_velocity"
    )
    first_second_order_index = next(
        index
        for index, row in enumerate(params)
        if row["early_anchor"] == "second_order_velocity"
    )
    assert first_scaled_index < first_second_order_index


def test_hypothesis_lab_run_accepts_failure_memory_input(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_phase_probe",
                        "candidate_params": {
                            "k": 1,
                            "mode": "affine",
                            "early_blend_steps": 18,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.0,
                            "early_anchor": "phase_aligned_retrieval",
                        },
                        "decision": "keep_as_evidence",
                        "pareto_status": "tradeoff_evidence",
                        "outcome": "evidence_only",
                        "bottleneck_type": "phase_shift",
                        "action_directives": [
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ],
                        "effective_axes": ["early_rel_mse"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    target = np.empty((32, 200, 256), dtype=np.float32)
    for sample in range(target.shape[0]):
        offset = np.float32(0.05 * sample)
        for step in range(200):
            target[sample, step, :] = (
                offset
                + np.float32(0.002 * step)
                + np.float32(0.1) * np.sin(x + np.float32(0.02 * step))
            )
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", target[:, :10, :])
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_hypothesis_lab(
        manual_report_paths=[],
        failure_memory_paths=[memory_path],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        force=False,
    )

    cards = json.loads(Path(report["cards"]["path"]).read_text(encoding="utf-8"))
    assert cards["cards"][0]["source_type"] == "pareto_learning_memory"
    assert report["sandbox"]["candidate_family"] == "task1_early_refinement"
    anchors = {
        row["params"]["early_anchor"] for row in report["sandbox"]["top_candidates"]
    }
    assert "phase_aligned_retrieval" in anchors


def test_hypothesis_lab_promoted_candidate_records_strategy_growth_artifacts(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab

    context = np.ones((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, 9, :] - np.float32(0.002 * (step + 1))
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    report = run_hypothesis_lab(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _local_asset_algorithm_candidates(tmp_path / "local_asset_candidates.json")
        ],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        force=False,
    )

    proposal_path = Path(report["artifacts"]["strategy_proposal_path"])
    evidence_path = Path(report["artifacts"]["strategy_evidence_summary_path"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert report["strategy_growth"]["strategy_id"] == (
        "candidate_local_strategy_data_inference_linear_trend_spectral_v0"
    )
    assert report["next_command"] == "build-strategy-rubric"
    assert proposal["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert proposal["source_type"] == "agent_hypothesis"
    assert "run-hypothesis-lab" in proposal["implementation_plan"]["allowed_commands"]
    assert evidence["state_hint"] == "STRATEGY_EVIDENCE_SUMMARY_BUILT"
    assert evidence["evaluation_window_count"] == 1
    assert evidence["mean_score_delta"] > 0.0
    assert evidence["next_command"] == "evaluate-strategy-promotion"


def test_hypothesis_lab_single_window_evidence_does_not_promote_to_ledger(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import run_hypothesis_lab
    from pde_agent.core.strategy_growth import (
        build_strategy_rubric,
        evaluate_strategy_promotion,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, 9, :] - np.float32(0.002 * (step + 1))
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    report = run_hypothesis_lab(
        manual_report_paths=[],
        algorithm_candidate_paths=[
            _local_asset_algorithm_candidates(tmp_path / "local_asset_candidates.json")
        ],
        context_path=context_path,
        target_path=target_path,
        output_dir=tmp_path / "lab",
        submission_id="team_hermes",
        force=False,
    )

    rubric_path = tmp_path / "rubric.json"
    promotion_path = tmp_path / "promotion.json"
    build_strategy_rubric(
        proposal_path=Path(report["artifacts"]["strategy_proposal_path"]),
        output_path=rubric_path,
        min_evaluation_windows=2,
        max_inference_time_regression=120.0,
    )
    promotion = evaluate_strategy_promotion(
        proposal_path=Path(report["artifacts"]["strategy_proposal_path"]),
        rubric_path=rubric_path,
        evidence_summary_path=Path(
            report["artifacts"]["strategy_evidence_summary_path"]
        ),
        output_path=promotion_path,
    )

    assert promotion["decision"] == "reject"
    assert promotion["promote_to_skill_candidate"] is False
    assert "insufficient_evaluation_windows" in promotion["reason_codes"]


def test_hypothesis_lab_cli_accepts_algorithm_candidates_and_memory_output(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    context = np.ones((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, 9, :] - np.float32(0.002 * (step + 1))
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    memory_path = tmp_path / "failure_memory.json"

    result = CliRunner().invoke(
        app,
        [
            "run-hypothesis-lab",
            "--algorithm-candidates-path",
            str(_local_asset_algorithm_candidates(tmp_path / "candidates.json")),
            "--failure-memory-output-path",
            str(memory_path),
            "--context-path",
            str(context_path),
            "--target-path",
            str(target_path),
            "--output-dir",
            str(tmp_path / "lab"),
            "--submission-id",
            "team_hermes",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["card_count"] == 1
    assert payload["metrics"]["decision"] == "promote_to_strategy_proposal"
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert memory["outcomes"][0]["outcome"] == "passed"


def test_hypothesis_lab_cli_accepts_failure_memory_input(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": [],
                "failed_candidate_keys": [],
                "failed_candidate_families": [],
                "outcomes": [],
                "pareto_learning_events": [
                    {
                        "candidate_family": "task1_early_refinement",
                        "candidate_id": "task1_early_refinement_phase_probe",
                        "candidate_params": {
                            "k": 1,
                            "mode": "affine",
                            "early_blend_steps": 18,
                            "persistence_weight_start": 0.5,
                            "spectral_strength": 0.0,
                            "early_anchor": "phase_aligned_retrieval",
                        },
                        "decision": "keep_as_evidence",
                        "pareto_status": "tradeoff_evidence",
                        "outcome": "evidence_only",
                        "bottleneck_type": "phase_shift",
                        "action_directives": [
                            "add_phase_alignment_or_derivative_distance_candidates"
                        ],
                        "effective_axes": ["early_rel_mse"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target = np.ones((32, 200, 256), dtype=np.float32)
    for step in range(200):
        target[:, step, :] += np.float32(0.001 * step)
    context_path = _write_hdf5(tmp_path / "task1_test.hdf5", target[:, :10, :])
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)

    result = CliRunner().invoke(
        app,
        [
            "run-hypothesis-lab",
            "--failure-memory-path",
            str(memory_path),
            "--context-path",
            str(context_path),
            "--target-path",
            str(target_path),
            "--output-dir",
            str(tmp_path / "lab"),
            "--submission-id",
            "team_hermes",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["card_count"] == 1
    cards = json.loads(
        Path(tmp_path / "lab" / "knowledge_cards.json").read_text(encoding="utf-8")
    )
    assert cards["cards"][0]["source_type"] == "pareto_learning_memory"


def test_hermes_batch_loop_can_run_linear_trend_spectral_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    for step in range(10):
        context[:, step, :] += np.float32(0.01 * step)
    target = np.empty((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    for step in range(190):
        target[:, 10 + step, :] = context[:, 9, :] - np.float32(0.002 * (step + 1))
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    initial = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "initial",
        submission_id="team_hermes",
        method="persistence",
        force=False,
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial["artifacts"]["submission_dir"],
        max_rounds=3,
        patience=2,
        min_score_improvement=0.0,
        submit_score_threshold=150.0,
        task1_test_path=test_path,
        task1_target_path=target_path,
        candidate_generator="linear_trend_spectral",
        force=False,
    )

    assert report["best_candidate"]["metadata"]["source"] == (
        "linear_trend_spectral_generator"
    )
    assert report["best_candidate"]["objective_signals"]["segment_score"] > (
        report["trajectory"][0]["objective_signals"]["segment_score"]
    )


def test_hermes_batch_loop_can_drive_linear_trend_spectral_from_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    target = np.concatenate(
        [context, np.repeat(context[:, -1:, :], 190, axis=1)],
        axis=1,
    )
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    initial = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "initial",
        submission_id="team_hermes",
        method="persistence",
        force=False,
    )
    blueprint_path = tmp_path / "blueprints.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_DESIGNED",
                "blueprints": [
                    {
                        "candidate_family": "linear_trend_spectral",
                        "seed_params": [
                            {
                                "params": {
                                    "slope_window": 5,
                                    "gain": -0.1,
                                    "decay": 60.0,
                                    "spectral_strength": 1.0,
                                },
                                "source": "sandbox_best",
                            }
                        ],
                        "parameter_grid": {
                            "slope_window": [4],
                            "gain": [-0.075],
                            "decay": [70.0],
                            "spectral_strength": [0.5],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial["artifacts"]["submission_dir"],
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=150.0,
        task1_test_path=test_path,
        task1_target_path=target_path,
        candidate_generator="linear_trend_spectral",
        hypothesis_blueprint_path=blueprint_path,
        force=False,
    )

    assert report["trajectory"][1]["metadata"]["params"] == {
        "slope_window": 5,
        "gain": -0.1,
        "decay": 60.0,
        "spectral_strength": 1.0,
    }


def test_hermes_batch_loop_can_drive_late_spectral_ramp_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    target = np.concatenate(
        [context, np.repeat(context[:, -1:, :], 190, axis=1)],
        axis=1,
    )
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    initial = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "initial",
        submission_id="team_hermes",
        method="persistence",
        force=False,
    )
    blueprint_path = tmp_path / "blueprints.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_BLUEPRINTS_VERIFIED",
                "blueprints": [
                    {
                        "candidate_family": "linear_trend_spectral",
                        "search_mode": "late_spectral_ramp",
                        "seed_params": [
                            {
                                "params": {
                                    "slope_window": 5,
                                    "gain": -0.075,
                                    "decay": 70.0,
                                    "spectral_strength": 1.5,
                                    "late_spectral_strength": 2.0,
                                    "late_start": 80,
                                },
                                "source": "sandbox_best",
                            }
                        ],
                        "parameter_grid": {
                            "slope_window": [5],
                            "gain": [-0.075],
                            "decay": [70.0],
                            "spectral_strength": [1.5],
                            "late_spectral_strength": [2.0],
                            "late_start": [80],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial["artifacts"]["submission_dir"],
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=150.0,
        task1_test_path=test_path,
        task1_target_path=target_path,
        candidate_generator="linear_trend_spectral",
        hypothesis_blueprint_path=blueprint_path,
        force=False,
    )

    params = report["trajectory"][1]["metadata"]["params"]
    assert params["late_spectral_strength"] == 2.0
    assert params["late_start"] == 80
    assert "_ls2" in report["trajectory"][1]["candidate_id"]
