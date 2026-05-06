from pathlib import Path
from typing import Any
import json


def test_hermes_candidate_generator_registry_marks_auto_safe_costs() -> None:
    from pde_agent.core.hermes_heartbeat import get_candidate_generator_registry

    registry = get_candidate_generator_registry()

    assert registry["physics_residual_nudge"]["cost_level"] == "light"
    assert registry["physics_residual_nudge"]["auto_fallback_allowed"] is True
    assert registry["sem_window_gate_blend_multi_search"]["cost_level"] == "heavy"
    assert (
        registry["sem_window_gate_blend_multi_search"]["auto_fallback_allowed"]
        is False
    )


def test_hermes_candidate_generator_registry_exposes_factory_front25_hook() -> None:
    from pde_agent.core.hermes_heartbeat import get_candidate_generator_registry

    registry = get_candidate_generator_registry()

    assert registry["hypothesis_lab_front25_dynamics_smoke"] == {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["front25_dynamics"],
    }
    assert registry["hypothesis_lab_early_subsegment_patch_search"] == {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["residual_corrector", "spectral_physics"],
    }


def test_hermes_heartbeat_runs_batches_and_carries_best_and_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    batch_calls: list[dict[str, Any]] = []
    memory_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        batch_index = len(batch_calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_index}"
        best_dir.mkdir()
        score = 80.0 + 5.0 * batch_index
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 3,
            "best_candidate": {
                "candidate_id": f"best_{batch_index}",
                "score": score,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "patience_exhausted",
            "decision": "iterate_more",
            "next_command": "run-hermes-batch-optimization-loop",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        memory_calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {"avoid_count": 0, "updated_log_count": 0},
            "next_command": "run-hermes-batch-optimization-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=2,
        batch_deadline_seconds=5.0,
        max_rounds_per_batch=4,
        checkpoint_every=2,
        max_stagnant_batches=2,
        submit_score_threshold=999.0,
        learning_objective="auto",
        unet_checkpoint_path="checkpoint.pt",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        timing_profile="mac_mini_m4_local_dev",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        hypothesis_blueprint_path="blueprints.json",
        force=False,
    )

    assert len(batch_calls) == 2
    assert batch_calls[0]["initial_submission_dir"] == initial_dir
    assert batch_calls[0]["learning_objective"] == "auto"
    assert batch_calls[0]["timing_profile"] == "mac_mini_m4_local_dev"
    assert batch_calls[0]["unet_checkpoint_path"] == Path("checkpoint.pt")
    assert batch_calls[0]["hypothesis_blueprint_path"] == Path("blueprints.json")
    assert batch_calls[1]["initial_submission_dir"] == tmp_path / "best_1"
    assert batch_calls[1]["hypothesis_blueprint_path"] == Path("blueprints.json")
    assert batch_calls[1]["memory_json_path"] == Path(
        report["batches"][0]["memory_json_path"]
    )
    assert len(memory_calls) == 2
    assert report["state_hint"] == "HERMES_HEARTBEAT_RAN"
    assert report["best_candidate"]["candidate_id"] == "best_2"
    assert report["best_score"] == 90.0
    assert report["stop_reason"] == "max_batches_exhausted"
    assert report["parameters"]["candidate_generator"] == "postprocess"
    assert report["parameters"]["submit_score_threshold"] == 999.0
    assert Path(report["heartbeat_report_path"]).exists()


def test_hermes_heartbeat_allows_bounded_directive_guard_override(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    doom_path = tmp_path / "doom.json"
    doom_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": ["postprocess"],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "bounded_reprobe",
                "score": 121.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        batch_deadline_seconds=5.0,
        max_rounds_per_batch=1,
        submit_score_threshold=999.0,
        candidate_generator="postprocess",
        doom_loop_report_path=doom_path,
        candidate_generator_fallback="auto",
        candidate_generator_guard_overrides=["postprocess"],
        hypothesis_candidate_options={"source_stability_guard": True},
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "postprocess"
    assert report["parameters"]["candidate_generator_guard_overrides"] == [
        "postprocess"
    ]
    assert report["parameters"]["hypothesis_candidate_options"] == {
        "source_stability_guard": True
    }
    assert report["batches"][0]["directive_guard_override"] is True
    assert report["batches"][0]["guarded_candidate_generator"] is False


def test_hermes_heartbeat_summarizes_batch_candidate_diagnostics(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir()
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 2,
            "best_candidate": {
                "candidate_id": "initial",
                "score": 120.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": 120.0},
            },
            "trajectory": [
                {
                    "candidate_id": "initial",
                    "score": 120.0,
                    "improved": True,
                    "objective": {"objective_score": 120.0},
                    "metadata": {"source": "initial_submission"},
                },
                {
                    "candidate_id": "local_probe_a",
                    "score": 119.5,
                    "improved": False,
                    "objective": {"objective_score": 119.5},
                    "rules_passed": True,
                    "judge_decision": "submit_candidate",
                    "metadata": {
                        "source": "sem_window_gate_blend_multi_search",
                        "search_evaluated_count": 144,
                        "selected_sample_count": 0,
                        "validation_selection_metric": "early_segment_score",
                        "validation_selection_score": 33.6,
                        "search_report_path": "search.json",
                    },
                },
                {
                    "candidate_id": "local_probe_b",
                    "score": 119.75,
                    "improved": False,
                    "objective": {"objective_score": 119.75},
                    "rules_passed": True,
                    "judge_decision": "submit_candidate",
                    "metadata": {
                        "source": "sem_window_gate_blend_multi_search",
                        "search_evaluated_count": 96,
                        "memory_guidance_spec_count": 7,
                        "selected_sample_count": 3,
                    },
                },
            ],
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    diagnostics = report["batches"][0]["candidate_diagnostics"]

    assert diagnostics["trajectory_count"] == 3
    assert diagnostics["attempted_candidate_count"] == 2
    assert diagnostics["improved_candidate_count"] == 0
    assert diagnostics["best_attempted_candidate_id"] == "local_probe_b"
    assert diagnostics["best_attempted_delta_vs_initial_objective"] == -0.25
    assert diagnostics["best_attempted_metadata"] == {
        "source": "sem_window_gate_blend_multi_search",
        "search_evaluated_count": 96,
        "memory_guidance_spec_count": 7,
        "selected_sample_count": 3,
    }


def test_hermes_heartbeat_preserves_search_guidance_across_batches(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        batch_index = len(batch_calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_index}"
        best_dir.mkdir()
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{batch_index}",
                "score": 100.0 + batch_index,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": 100.0 + batch_index},
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    escalation_guidance = {
        "guidance_type": "escalate_after_guided_search_exhaustion",
        "reason": "memory_guided_specs_failed_to_improve",
        "source": "sem_window_gate_blend_multi_search",
        "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
    }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        memory_json_path = Path(kwargs["memory_json_path"])
        memory_json_path.parent.mkdir(parents=True, exist_ok=True)
        if "batch_001" in str(kwargs["batch_loop_report_path"]):
            payload = {"search_guidance": [escalation_guidance]}
        else:
            payload = {"search_guidance": []}
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        memory_json_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(memory_json_path),
                "updated_log_paths": [],
            },
            "metrics": {"search_guidance_count": len(payload["search_guidance"])},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=3,
        max_stagnant_batches=3,
        candidate_generator_schedule=[
            "sem_window_gate_blend_multi_early_smoke",
            "postprocess",
            "sem_window_gate_blend_multi_early_smoke",
        ],
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    third_memory_path = Path(batch_calls[2]["memory_json_path"])
    third_memory = json.loads(third_memory_path.read_text(encoding="utf-8"))

    assert third_memory["search_guidance"] == [escalation_guidance]
    assert report["batches"][2]["batch_input_memory_json_path"] == str(
        third_memory_path
    )
    assert report["parameters"]["persistent_search_guidance_count"] == 1


def test_hermes_heartbeat_search_guidance_escalation_switches_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        batch_index = len(batch_calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_index}"
        best_dir.mkdir()
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{batch_index}",
                "score": 100.0 + batch_index,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": 100.0 + batch_index},
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    escalation_guidance = {
        "guidance_type": "escalate_after_guided_search_exhaustion",
        "reason": "memory_guided_specs_failed_to_improve",
        "source": "sem_window_gate_blend_multi_search",
        "recommended_action": "switch_gate_feature_family_or_try_new_dynamics_source",
    }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        memory_json_path = Path(kwargs["memory_json_path"])
        memory_json_path.parent.mkdir(parents=True, exist_ok=True)
        if "batch_001" in str(kwargs["batch_loop_report_path"]):
            payload = {"search_guidance": [escalation_guidance]}
        else:
            payload = {"search_guidance": []}
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        memory_json_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(memory_json_path),
                "updated_log_paths": [],
            },
            "metrics": {"search_guidance_count": len(payload["search_guidance"])},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=2,
        max_stagnant_batches=2,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert [call["candidate_generator"] for call in batch_calls] == [
        "sem_window_gate_blend_multi_early_smoke",
        "physics_residual_nudge",
    ]
    assert report["batches"][1]["original_candidate_generator"] == (
        "sem_window_gate_blend_multi_early_smoke"
    )
    assert report["batches"][1]["candidate_generator"] == "physics_residual_nudge"
    assert report["batches"][1]["guarded_candidate_generator"] is True
    assert report["batches"][1]["search_guidance_guarded_candidate_generator"] is True
    assert report["batches"][1]["search_guidance_guard_reason"] == (
        "guided_sem_window_search_exhausted"
    )


def test_hermes_heartbeat_loads_persistent_search_guidance_from_kpi(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "persistent_search_guidance": [
                    {
                        "guidance_type": "escalate_after_guided_search_exhaustion",
                        "reason": "memory_guided_specs_failed_to_improve",
                        "source": "sem_window_gate_blend_multi_search",
                        "recommended_action": (
                            "switch_gate_feature_family_or_try_new_dynamics_source"
                        ),
                    }
                ],
                "active_audit_repair_errors": [
                    "persistent_search_guidance_consistency",
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text(
            json.dumps({"search_guidance": []}),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "physics_residual_nudge"
    assert report["batches"][0]["search_guidance_guarded_candidate_generator"] is True
    assert report["parameters"]["persistent_search_guidance_count"] == 1
    assert report["parameters"]["active_audit_repair_error_count"] == 1
    assert report["active_audit_repair_errors"] == [
        "persistent_search_guidance_consistency"
    ]
    assert report["persistent_search_guidance"] == json.loads(
        kpi_path.read_text(encoding="utf-8")
    )["persistent_search_guidance"]


def test_hermes_heartbeat_records_repair_policy_actions(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    repair_action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    repair_policy_path = tmp_path / "repair_policy.json"
    repair_policy_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
                "recommended_actions": [repair_action],
                "metrics": {"recommended_action_count": 1},
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best_policy_visible",
                "score": 101.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": 101.0},
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        repair_policy_summary_path=repair_policy_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert report["parameters"]["repair_policy_summary_path"] == str(
        repair_policy_path
    )
    assert report["parameters"]["repair_policy_recommended_action_count"] == 1
    assert report["repair_policy_recommended_actions"] == [repair_action]
    assert report["batches"][0]["repair_policy_recommended_actions"] == [
        repair_action
    ]
    assert report["batches"][0]["repair_policy_recommended_action_count"] == 1
    assert batch_calls[0]["candidate_generator"] == "postprocess"


def test_hermes_heartbeat_seeds_batch_memory_with_repair_policy_actions(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    repair_action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    repair_policy_path = tmp_path / "repair_policy.json"
    repair_policy_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
                "recommended_actions": [repair_action],
                "metrics": {"recommended_action_count": 1},
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        memory_path = kwargs["memory_json_path"]
        assert memory_path is not None
        payload = json.loads(Path(memory_path).read_text(encoding="utf-8"))
        assert payload["repair_policy_recommended_actions"] == [repair_action]
        assert payload["repair_policy_recommended_action_count"] == 1
        assert "repair_policy_summary_path" in payload["input_sources"]
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "memory_constraints": {
                "repair_policy_recommended_action_count": 1,
                "repair_policy_recommended_actions": [repair_action],
            },
            "best_candidate": {
                "candidate_id": "best_policy_seeded",
                "score": 101.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": 101.0},
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        repair_policy_summary_path=repair_policy_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert len(batch_calls) == 1
    assert batch_calls[0]["avoid_repeating"] is True
    assert Path(report["batches"][0]["batch_input_memory_json_path"]).exists()
    assert report["batches"][0]["candidate_diagnostics"]["memory_constraints"] == {
        "repair_policy_recommended_action_count": 1,
        "repair_policy_recommended_actions": [repair_action],
    }


def test_hermes_heartbeat_merges_repair_policy_actions_into_existing_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    repair_action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    repair_policy_path = tmp_path / "repair_policy.json"
    repair_policy_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_REPAIR_POLICY_SUMMARY_BUILT",
                "recommended_actions": [repair_action],
                "metrics": {"recommended_action_count": 1},
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        if len(batch_calls) == 2:
            payload = json.loads(
                Path(kwargs["memory_json_path"]).read_text(encoding="utf-8")
            )
            assert payload["search_guidance"] == [
                {"guidance_type": "existing_guidance"}
            ]
            assert payload["repair_policy_recommended_actions"] == [repair_action]
            assert payload["repair_policy_recommended_action_count"] == 1
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{len(batch_calls)}"
        best_dir.mkdir(exist_ok=True)
        score = 101.0 + len(batch_calls)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{len(batch_calls)}",
                "score": score,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {"objective_score": score},
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        payload = {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "search_guidance": [{"guidance_type": "existing_guidance"}],
        }
        Path(kwargs["memory_json_path"]).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=2,
        max_stagnant_batches=2,
        repair_policy_summary_path=repair_policy_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert len(batch_calls) == 2
    assert batch_calls[1]["memory_json_path"] == Path(
        report["batches"][1]["batch_input_memory_json_path"]
    )


def test_hermes_heartbeat_can_alternate_learning_objectives_and_generators(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        batch_index = len(batch_calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_index}"
        best_dir.mkdir()
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{batch_index}",
                "score": 100.0 + batch_index,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {
                    "selected_objective": kwargs["learning_objective"],
                    "objective_score": 100.0 + batch_index,
                },
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {"avoid_count": 0, "updated_log_count": 0},
            "next_command": "run-hermes-batch-optimization-loop",
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=3,
        max_stagnant_batches=3,
        min_batch_score_improvement=0.0,
        min_score_improvement=0.0,
        learning_objective="score",
        candidate_generator="postprocess",
        learning_objective_schedule=["early_accuracy", "accuracy"],
        candidate_generator_schedule=[
            "sem_window_gate_blend_multi_early_search",
            "sem_window_gate_blend_multi_search",
        ],
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert [call["learning_objective"] for call in batch_calls] == [
        "early_accuracy",
        "accuracy",
        "early_accuracy",
    ]
    assert [call["candidate_generator"] for call in batch_calls] == [
        "sem_window_gate_blend_multi_early_search",
        "sem_window_gate_blend_multi_search",
        "sem_window_gate_blend_multi_early_search",
    ]
    assert [batch["learning_objective"] for batch in report["batches"]] == [
        "early_accuracy",
        "accuracy",
        "early_accuracy",
    ]
    assert report["parameters"]["learning_objective_schedule"] == [
        "early_accuracy",
        "accuracy",
    ]
    assert report["parameters"]["candidate_generator_schedule"] == [
        "sem_window_gate_blend_multi_early_search",
        "sem_window_gate_blend_multi_search",
    ]


def test_hermes_heartbeat_stops_when_score_stagnates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "flat",
                "score": 80.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "patience_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=5,
        batch_deadline_seconds=5.0,
        max_rounds_per_batch=2,
        max_stagnant_batches=1,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert report["batch_count"] == 2
    assert report["stop_reason"] == "stagnation_detected"
    assert report["decision"] == "review_before_next_heartbeat"


def test_hermes_heartbeat_uses_objective_score_for_batch_stagnation(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    objective_scores = iter([-150.0, -130.0, -110.0])
    batch_count = 0

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal batch_count
        batch_count += 1
        objective_score = next(objective_scores)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_count}"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"runtime_{batch_count}",
                "score": 0.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
                "objective": {
                    "selected_objective": "runtime",
                    "objective_score": objective_score,
                    "inference_time": -objective_score,
                },
            },
            "stop_reason": "patience_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=3,
        batch_deadline_seconds=5.0,
        max_rounds_per_batch=1,
        max_stagnant_batches=1,
        min_batch_score_improvement=1.0,
        learning_objective="auto",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert report["batch_count"] == 3
    assert report["best_candidate"]["candidate_id"] == "runtime_3"
    assert report["best_objective_score"] == -110.0
    assert report["stop_reason"] == "max_batches_exhausted"


def test_hermes_heartbeat_can_route_batches_to_hypothesis_lab(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "retrieval_best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 2,
            "best_candidate": {
                "candidate_id": "task1_retrieval_prior_k20_affine",
                "score": 109.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "candidate_runner_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        batch_deadline_seconds=5.0,
        max_rounds_per_batch=2,
        candidate_generator="hypothesis_lab",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "hypothesis_lab"
    assert batch_calls[0]["task1_target_path"] == Path("task1_val.hdf5")
    assert report["best_candidate"]["candidate_id"] == "task1_retrieval_prior_k20_affine"
    assert report["parameters"]["candidate_generator"] == "hypothesis_lab"


def test_hermes_heartbeat_auto_selects_unblocked_fallback_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke",
                    "postprocess",
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        candidate_generator_fallback="auto",
        doom_loop_report_path=guard_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "physics_residual_nudge"
    assert report["batches"][0]["candidate_generator"] == "physics_residual_nudge"
    assert report["batches"][0]["original_candidate_generator"] == (
        "sem_window_gate_blend_multi_early_smoke"
    )
    assert report["batches"][0]["guarded_candidate_generator"] is True


def test_hermes_heartbeat_default_fallback_is_auto(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke",
                    "postprocess",
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        doom_loop_report_path=guard_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "physics_residual_nudge"
    assert report["parameters"]["candidate_generator_fallback"] == "auto"
    metadata = report["batches"][0]["candidate_generator_metadata"]
    assert metadata["auto_fallback_allowed"] is True
    assert metadata["cost_level"] == "light"
    assert metadata["default_budget"] == "smoke"
    assert metadata["requires_reference"] is False
    assert metadata["requires_target"] is False
    assert "physical_residual" in metadata["targets_error_modes"]


def test_hermes_heartbeat_auto_fallback_prefers_kpi_positive_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke",
                    "postprocess",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "best_family_by_score_delta": {
                    "candidate_family": "fast_baseline_family",
                    "best_score_delta": 0.4,
                    "mean_score_delta": 0.2,
                    "count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        doom_loop_report_path=guard_path,
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "fast_baseline_family"
    assert report["batches"][0]["candidate_generator"] == "fast_baseline_family"
    assert report["parameters"]["hypothesis_kpi_report_path"] == str(kpi_path)


def test_hermes_heartbeat_auto_fallback_avoids_kpi_saturated_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "saturated_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke"
                ],
                "best_family_by_score_delta": {
                    "candidate_family": "sem_window_gate_blend_multi_early_smoke",
                    "best_score_delta": 0.4,
                    "mean_score_delta": 0.2,
                    "count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["original_candidate_generator"] == (
        "sem_window_gate_blend_multi_early_smoke"
    )
    assert report["batches"][0]["guarded_candidate_generator"] is True
    assert report["parameters"]["saturated_candidate_families"] == [
        "sem_window_gate_blend_multi_early_smoke"
    ]


def test_hermes_heartbeat_saturation_alias_blocks_local_strategy_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "saturated_candidate_families": ["local_strategy_skill"],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["candidate_generator"] == "postprocess"
    assert report["batches"][0]["guarded_candidate_generator"] is True
    assert report["parameters"]["saturated_candidate_families"] == [
        "local_strategy_skill"
    ]


def test_hermes_heartbeat_allows_controlled_saturation_probe(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "saturated_candidate_families": ["local_strategy_skill"],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        batch_index = len(batch_calls)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / f"best_{batch_index}"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": f"best_{batch_index}",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=2,
        max_stagnant_batches=2,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        hypothesis_kpi_report_path=kpi_path,
        saturation_probe_interval_batches=2,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert [call["candidate_generator"] for call in batch_calls] == [
        "postprocess",
        "sem_window_gate_blend_multi_early_smoke",
    ]
    assert report["batches"][0]["saturation_probe"] is False
    assert report["batches"][1]["saturation_probe"] is True
    assert report["batches"][1]["saturation_probe_reason"] == (
        "interval_reprobe_of_saturated_candidate_family"
    )
    assert report["parameters"]["saturation_probe_interval_batches"] == 2


def test_hermes_heartbeat_auto_fallback_skips_saturation_aliases(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke",
                    "postprocess",
                    "physics_residual_nudge",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "saturated_candidate_families": ["local_strategy_skill"],
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        doom_loop_report_path=guard_path,
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "fast_baseline_family"
    assert report["batches"][0]["candidate_generator"] == "fast_baseline_family"


def test_hermes_heartbeat_auto_fallback_ignores_heavy_kpi_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_heartbeat import run_hermes_heartbeat

    guard_path = tmp_path / "doom.json"
    guard_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_DOOM_LOOP_CHECKED",
                "guard_active": True,
                "blocked_candidate_families": [
                    "sem_window_gate_blend_multi_early_smoke",
                    "postprocess",
                ],
            }
        ),
        encoding="utf-8",
    )
    kpi_path = tmp_path / "kpi.json"
    kpi_path.write_text(
        __import__("json").dumps(
            {
                "state_hint": "HYPOTHESIS_KPI_REPORT_BUILT",
                "best_family_by_score_delta": {
                    "candidate_family": "sem_window_gate_blend_multi_search",
                    "best_score_delta": 0.4,
                    "mean_score_delta": 0.2,
                    "count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    batch_calls: list[dict[str, Any]] = []

    def fake_batch_runner(**kwargs: Any) -> dict[str, Any]:
        batch_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_batch_optimization_loop.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 1,
            "best_candidate": {
                "candidate_id": "best",
                "score": 90.0,
                "submission_dir": str(best_dir),
                "rules_passed": True,
            },
            "stop_reason": "max_rounds_exhausted",
            "decision": "iterate_more",
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("memory", encoding="utf-8")
        Path(kwargs["memory_json_path"]).write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(kwargs["output_path"]),
                "memory_json_path": str(kwargs["memory_json_path"]),
                "updated_log_paths": [],
            },
            "metrics": {},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    run_hermes_heartbeat(
        output_dir=tmp_path / "heartbeat",
        initial_submission_dir=initial_dir,
        max_batches=1,
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        doom_loop_report_path=guard_path,
        hypothesis_kpi_report_path=kpi_path,
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert batch_calls[0]["candidate_generator"] == "physics_residual_nudge"


def test_hermes_heartbeat_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_heartbeat as command_module
    from typer.testing import CliRunner

    heartbeat_calls: list[dict[str, Any]] = []

    def fake_heartbeat(**kwargs: Any) -> dict[str, Any]:
        heartbeat_calls.append(kwargs)
        report_path = tmp_path / "heartbeat.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {"candidate_id": "best", "score": 91.0},
            "best_score": 91.0,
            "parameters": {"repair_policy_recommended_action_count": 1},
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }

    monkeypatch.setattr(command_module, "run_hermes_heartbeat", fake_heartbeat)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-heartbeat",
            "--output-dir",
            str(tmp_path / "heartbeat"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
            "--learning-objective",
            "auto",
            "--repair-policy-summary-path",
            str(tmp_path / "repair_policy.json"),
        ],
    )

    assert result.exit_code == 0
    payload = __import__("json").loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_HEARTBEAT_RAN"
    assert payload["metrics"]["best_score"] == 91.0
    assert payload["metrics"]["repair_policy_recommended_action_count"] == 1
    assert heartbeat_calls[0]["learning_objective_schedule"] is None
    assert heartbeat_calls[0]["candidate_generator_schedule"] is None
    assert heartbeat_calls[0]["repair_policy_summary_path"] == (
        tmp_path / "repair_policy.json"
    )
    assert heartbeat_calls[0]["saturation_probe_interval_batches"] == 0
