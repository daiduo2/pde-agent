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
