import json
from pathlib import Path


def _write_loop_report(path: Path) -> Path:
    payload = {
        "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
        "stop_reason": "patience_exhausted",
        "decision": "iterate_more",
        "best_candidate": {
            "candidate_id": "postprocess_spectral_smooth_w0.375",
            "score": 88.0,
            "objective": {
                "requested_objective": "auto",
                "selected_objective": "runtime",
                "objective_score": -118.0,
                "objective_reason": "auto_runtime_because_reference_candidate_exceeds_inference_limit",
                "inference_time": 118.0,
            },
            "metadata": {
                "source": "postprocess_generator",
                "strategy": "spectral_smooth",
                "blend_weight": 0.375,
            },
        },
        "trajectory": [
            {
                "round_index": 0,
                "candidate_id": "initial",
                "score": 82.0,
                "rules_passed": True,
                "improved": True,
                "objective": {
                    "requested_objective": "auto",
                    "selected_objective": "runtime",
                    "objective_score": -157.0,
                    "inference_time": 157.0,
                },
                "metadata": {"source": "initial_submission"},
            },
            {
                "round_index": 1,
                "candidate_id": "postprocess_context_anchor_blend_w0.25",
                "score": 84.0,
                "rules_passed": True,
                "improved": True,
                "objective": {
                    "requested_objective": "auto",
                    "selected_objective": "runtime",
                    "objective_score": -130.0,
                    "inference_time": 130.0,
                },
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "context_anchor_blend",
                    "blend_weight": 0.25,
                },
            },
            {
                "round_index": 2,
                "candidate_id": "postprocess_spectral_smooth_w0.5",
                "score": 87.9,
                "rules_passed": True,
                "improved": False,
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "spectral_smooth",
                    "blend_weight": 0.5,
                },
            },
            {
                "round_index": 3,
                "candidate_id": "bad_rule_candidate",
                "score": 0.0,
                "rules_passed": False,
                "improved": False,
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "context_anchor_blend",
                    "blend_weight": 1.0,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_hermes_iteration_memory_extracts_avoid_list_and_next_hints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = _write_loop_report(tmp_path / "loop.json")
    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=None,
        task_ids=None,
        force=False,
    )

    memory_text = Path(report["artifacts"]["memory_markdown_path"]).read_text(
        encoding="utf-8"
    )
    memory_json = json.loads(
        Path(report["artifacts"]["memory_json_path"]).read_text(encoding="utf-8")
    )

    assert report["state_hint"] == "HERMES_ITERATION_MEMORY_BUILT"
    assert memory_json["best_candidate"]["candidate_id"] == (
        "postprocess_spectral_smooth_w0.375"
    )
    assert memory_json["best_candidate"]["objective"]["selected_objective"] == "runtime"
    assert memory_json["trajectory_summary"][0]["objective"]["inference_time"] == 157.0
    assert {
        "strategy": "context_anchor_blend",
        "blend_weight": 1.0,
        "reason": "rule_gate_failed",
    } in memory_json["avoid_repeating"]
    assert {
        "strategy": "spectral_smooth",
        "blend_weight": 0.5,
        "reason": "below_min_effective_improvement",
    } in memory_json["avoid_repeating"]
    assert "Do not repeat" in memory_text
    assert "selected_objective=runtime" in memory_text
    assert "spectral_smooth" in memory_text
    assert "Try local refinement around blend_weight=0.375" in memory_text


def test_build_hermes_iteration_memory_records_sem_window_near_miss_guidance(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = tmp_path / "loop.json"
    loop_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
                "stop_reason": "max_rounds_exhausted",
                "decision": "iterate_more",
                "best_candidate": {
                    "candidate_id": "initial",
                    "score": 120.02600438716073,
                    "objective": {"objective_score": 120.02600438716073},
                    "metadata": {"source": "initial_submission"},
                },
                "trajectory": [
                    {
                        "round_index": 0,
                        "candidate_id": "initial",
                        "score": 120.02600438716073,
                        "rules_passed": True,
                        "improved": True,
                        "objective": {"objective_score": 120.02600438716073},
                        "metadata": {"source": "initial_submission"},
                    },
                    {
                        "round_index": 1,
                        "candidate_id": (
                            "sem_window_gate_blend_sem_base_disagreement_20_57"
                        ),
                        "score": 120.01558538716073,
                        "rules_passed": True,
                        "improved": False,
                        "objective": {"objective_score": 120.01558538716073},
                        "metadata": {
                            "source": "sem_window_gate_blend_multi_search",
                            "feature_name": "sem_base_disagreement_20_57",
                            "threshold": -1.0,
                            "selected_sample_count": 0,
                            "search_evaluated_count": 144,
                            "validation_selection_metric": "early_segment_score",
                            "validation_selection_score": 33.63761423879183,
                            "window_start": 10,
                            "window_stop": 20,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=None,
        task_ids=None,
        force=False,
    )
    memory_json = json.loads(
        Path(report["artifacts"]["memory_json_path"]).read_text(encoding="utf-8")
    )
    memory_text = Path(report["artifacts"]["memory_markdown_path"]).read_text(
        encoding="utf-8"
    )

    guidance = memory_json["search_guidance"]

    assert report["metrics"]["search_guidance_count"] == 1
    assert guidance == [
        {
            "guidance_type": "refine_gating_after_near_miss",
            "reason": "near_miss_with_empty_gate_selection",
            "candidate_id": "sem_window_gate_blend_sem_base_disagreement_20_57",
            "source": "sem_window_gate_blend_multi_search",
            "objective_delta_vs_initial": -0.010419,
            "feature_name": "sem_base_disagreement_20_57",
            "threshold": -1.0,
            "selected_sample_count": 0,
            "search_evaluated_count": 144,
            "validation_selection_metric": "early_segment_score",
            "validation_selection_score": 33.63761423879183,
            "window_start": 10,
            "window_stop": 20,
            "recommended_action": (
                "relax_or_recenter_gate_threshold_and_window_before_repeating_family"
            ),
        }
    ]
    assert "Search Guidance" in memory_text
    assert "near_miss_with_empty_gate_selection" in memory_text
    assert "relax_or_recenter_gate_threshold_and_window" in memory_text


def test_build_hermes_iteration_memory_records_guided_sem_search_exhaustion(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = tmp_path / "loop.json"
    loop_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
                "stop_reason": "max_rounds_exhausted",
                "decision": "iterate_more",
                "best_candidate": {
                    "candidate_id": "initial",
                    "score": 120.02600438716073,
                    "objective": {"objective_score": 120.02600438716073},
                    "metadata": {"source": "initial_submission"},
                },
                "trajectory": [
                    {
                        "round_index": 0,
                        "candidate_id": "initial",
                        "score": 120.02600438716073,
                        "rules_passed": True,
                        "improved": True,
                        "objective": {"objective_score": 120.02600438716073},
                        "metadata": {"source": "initial_submission"},
                    },
                    {
                        "round_index": 1,
                        "candidate_id": (
                            "sem_window_gate_blend_sem_base_disagreement_20_57"
                        ),
                        "score": 120.01558538716073,
                        "rules_passed": True,
                        "improved": False,
                        "objective": {"objective_score": 120.01558538716073},
                        "metadata": {
                            "source": "sem_window_gate_blend_multi_search",
                            "feature_name": "sem_base_disagreement_20_57",
                            "threshold": -1.0,
                            "selected_sample_count": 0,
                            "search_evaluated_count": 220,
                            "memory_guidance_spec_count": 76,
                            "validation_selection_metric": "early_segment_score",
                            "validation_selection_score": 33.63761423879183,
                            "window_start": 10,
                            "window_stop": 20,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=None,
        task_ids=None,
        force=False,
    )
    memory_json = json.loads(
        Path(report["artifacts"]["memory_json_path"]).read_text(encoding="utf-8")
    )

    guidance = memory_json["search_guidance"]

    assert report["metrics"]["search_guidance_count"] == 1
    assert guidance[0]["guidance_type"] == "escalate_after_guided_search_exhaustion"
    assert guidance[0]["reason"] == "memory_guided_specs_failed_to_improve"
    assert guidance[0]["memory_guidance_spec_count"] == 76
    assert guidance[0]["recommended_action"] == (
        "switch_gate_feature_family_or_try_new_dynamics_source"
    )
    assert (
        "Switch gate feature family or dynamics source"
        in memory_json["next_hints"][0]
    )


def test_build_hermes_iteration_memory_records_family_failure_and_positive_signal(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = tmp_path / "loop.json"
    loop_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
                "stop_reason": "candidate_runner_exhausted",
                "decision": "provide_more_candidates",
                "best_candidate": {
                    "candidate_id": "postprocess_spectral_smooth_w0.7325",
                    "score": 120.34767254492024,
                    "objective": {
                        "objective_score": 120.34767254492024,
                        "segment_score": 60.47934694878255,
                        "early_segment_score": 34.43165771395293,
                        "inference_time": 0.035513,
                    },
                    "metadata": {
                        "source": "postprocess_generator",
                        "strategy": "spectral_smooth",
                        "blend_weight": 0.7325,
                    },
                },
                "trajectory": [
                    {
                        "round_index": 0,
                        "candidate_id": "initial",
                        "score": 120.02600438716073,
                        "rules_passed": True,
                        "improved": True,
                        "objective": {
                            "objective_score": 120.02600438716073,
                            "segment_score": 60.0346725162143,
                            "early_segment_score": 33.63761423879183,
                            "inference_time": 0.0,
                        },
                        "metadata": {"source": "initial_submission"},
                    },
                    {
                        "round_index": 1,
                        "candidate_id": "linear_trend_spectral_worst",
                        "score": 86.2,
                        "rules_passed": True,
                        "improved": False,
                        "objective": {
                            "objective_score": 86.2,
                            "segment_score": 14.9,
                            "early_segment_score": 0.0,
                            "inference_time": 0.01,
                        },
                        "metadata": {"source": "linear_trend_spectral"},
                    },
                    {
                        "round_index": 2,
                        "candidate_id": "physics_residual_nudge_near_miss",
                        "score": 120.01945342249716,
                        "rules_passed": True,
                        "improved": False,
                        "objective": {
                            "objective_score": 120.01945342249716,
                            "segment_score": 60.028,
                            "early_segment_score": 33.63,
                            "inference_time": 0.02,
                        },
                        "metadata": {"source": "physics_residual_nudge"},
                    },
                    {
                        "round_index": 3,
                        "candidate_id": "postprocess_spectral_smooth_w0.7325",
                        "score": 120.34767254492024,
                        "rules_passed": True,
                        "improved": True,
                        "objective": {
                            "objective_score": 120.34767254492024,
                            "segment_score": 60.47934694878255,
                            "early_segment_score": 34.43165771395293,
                            "inference_time": 0.035513,
                        },
                        "metadata": {
                            "source": "postprocess_generator",
                            "strategy": "spectral_smooth",
                            "blend_weight": 0.7325,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=None,
        task_ids=None,
        force=False,
    )
    memory_json = json.loads(
        Path(report["artifacts"]["memory_json_path"]).read_text(encoding="utf-8")
    )
    memory_text = Path(report["artifacts"]["memory_markdown_path"]).read_text(
        encoding="utf-8"
    )

    assert {
        "candidate_family": "linear_trend_spectral",
        "reason": "severe_score_regression",
    } in memory_json["avoid_candidate_families"]
    assert memory_json["family_outcomes"]["linear_trend_spectral"]["status"] == (
        "blocked"
    )
    assert memory_json["family_outcomes"]["physics_residual_nudge"]["status"] == (
        "near_miss"
    )
    assert memory_json["family_outcomes"]["postprocess_generator"]["status"] == (
        "positive_signal"
    )
    assert memory_json["family_outcomes"]["postprocess_generator"][
        "best_strategy"
    ] == "spectral_smooth"
    assert memory_json["family_outcomes"]["postprocess_generator"][
        "best_blend_weight"
    ] == 0.7325
    assert memory_json["positive_signals"][0]["candidate_id"] == (
        "postprocess_spectral_smooth_w0.7325"
    )
    assert "Family Outcomes" in memory_text
    assert "linear_trend_spectral" in memory_text


def test_build_hermes_iteration_memory_appends_to_submission_log(tmp_path: Path) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = _write_loop_report(tmp_path / "loop.json")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    log_path = submission_dir / "task1_logs.log"
    log_path.write_text(
        "Hermes PDE Agent Scientific Log\n\nTrace:\n- prior\n\nExperiment Tracking:\n- prior\n",
        encoding="utf-8",
    )

    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=submission_dir,
        task_ids=[1],
        force=False,
    )

    text = log_path.read_text(encoding="utf-8")
    assert report["metrics"]["updated_log_count"] == 1
    assert "Hermes Iteration Memory" in text
    assert "Do not repeat" in text
    assert "postprocess_spectral_smooth_w0.375" in text


def test_build_hermes_iteration_memory_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.build_hermes_iteration_memory as command_module
    from typer.testing import CliRunner

    loop_path = _write_loop_report(tmp_path / "loop.json")

    def fake_builder(**_: object) -> dict[str, object]:
        memory_path = tmp_path / "memory.md"
        json_path = tmp_path / "memory.json"
        memory_path.write_text("Hermes Iteration Memory\n", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(memory_path),
                "memory_json_path": str(json_path),
                "updated_log_paths": [],
            },
            "metrics": {
                "trajectory_count": 4,
                "avoid_count": 2,
                "updated_log_count": 0,
            },
            "next_command": "run-hermes-batch-optimization-loop",
        }

    monkeypatch.setattr(command_module, "build_hermes_iteration_memory", fake_builder)

    result = CliRunner().invoke(
        app,
        [
            "build-hermes-iteration-memory",
            "--batch-loop-report-path",
            str(loop_path),
            "--output-path",
            str(tmp_path / "memory.md"),
            "--memory-json-path",
            str(tmp_path / "memory.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_ITERATION_MEMORY_BUILT"
    assert payload["metrics"]["avoid_count"] == 2
