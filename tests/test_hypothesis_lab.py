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
    assert blueprint["sandbox"]["allowed_executor"] == "controlled_numpy_operator"
    assert 1.0 in blueprint["parameter_grid"]["spectral_strength"]


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
    assert Path(sandbox["artifacts"]["best_submission_dir"]).exists()


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
