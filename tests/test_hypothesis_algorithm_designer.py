from __future__ import annotations

import json
from pathlib import Path


def test_algorithm_designer_builds_diverse_blueprints_from_failure_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    checkpoint_memory = tmp_path / "checkpoint_failure.json"
    checkpoint_memory.write_text(
        json.dumps(
            {
                "state_hint": "CHECKPOINT_FRONT_WINDOW_ADAPTER_FAILURE_MEMORY_RECORDED",
                "objective": "checkpoint/front-window static adapter failed",
                "candidate_ablation": {
                    "adapter_classes": ["direct", "blend", "residual"],
                    "windows": ["10:20", "20:35", "35:57"],
                    "uses_target_as_reference": False,
                },
                "window_selection_result": [
                    {
                        "time_start": 20,
                        "time_stop": 35,
                        "selected_source_candidate_id": "initial",
                    },
                    {
                        "time_start": 35,
                        "time_stop": 57,
                        "selected_source_candidate_id": "initial",
                    },
                ],
                "conclusion": (
                    "Static checkpoint window replacement did not improve; continue "
                    "only with trained adapter, residual model, or nonlocal "
                    "sample-conditioned gate."
                ),
            }
        ),
        encoding="utf-8",
    )
    short_window_memory = tmp_path / "short_window_failure.json"
    short_window_memory.write_text(
        json.dumps(
            {
                "state_hint": "SHORT_WINDOW_CORRECTOR_FAILURE_MEMORY_RECORDED",
                "conclusion": (
                    "Short-window corrector did not beat initial in 10:20, 20:35, "
                    "or 35:57; likely phase drift and shock over-smoothing remain."
                ),
                "recommendation": "Use phase alignment or learned front-window dynamics.",
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[checkpoint_memory, short_window_memory],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[],
        output_path=tmp_path / "designer.json",
        min_hypothesis_count=5,
        min_mechanism_family_count=4,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_ALGORITHM_DESIGN_BUILT"
    assert report["diagnosis"]["failure_memory_count"] == 2
    mechanisms = {
        item["mechanism_family"] for item in report["mechanism_hypotheses"]
    }
    assert mechanisms >= {
        "uncertainty_gating",
        "residual_correction",
        "data_retrieval_nonlocal",
        "time_window_decomposition",
    }
    assert report["diversity_audit"]["near_duplicate_count"] == 0
    assert report["designer_rubric"]["decision"] == "ready_for_sandbox_planning"
    assert report["designer_rubric"]["mechanism_family_count"] >= 4
    assert len(report["algorithm_blueprints"]) >= 5
    for blueprint in report["algorithm_blueprints"]:
        assert blueprint["execution_constraints"]["forbidden_data"] == [
            "task1_target_as_prediction_source",
            "local_validation_target_as_training_reference_for_submission",
        ]
        assert blueprint["evaluation_plan"]["expected_observables"]
        assert blueprint["evaluation_plan"]["segment_level_metrics"]
        assert blueprint["evaluation_plan"]["physics_diagnostics"]
        assert blueprint["evaluation_plan"]["runtime_metrics"]
        assert blueprint["evaluation_plan"]["ablation_plan"]
        assert blueprint["evaluation_plan"]["falsification_criteria"]
        assert blueprint["internalization_plan"]["algorithm_card_update"]
        assert blueprint["internalization_plan"]["evidence_record"]
        assert blueprint["internalization_plan"]["failure_taxonomy_update"]


def test_algorithm_designer_validator_rejects_duplicate_or_nonrunnable_designs(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        validate_hypothesis_algorithm_design,
    )

    design_path = tmp_path / "bad_design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "mechanism_hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "mechanism_family": "checkpoint_adaptation",
                        "mechanism": "try window blend",
                    },
                    {
                        "hypothesis_id": "h2",
                        "mechanism_family": "checkpoint_adaptation",
                        "mechanism": "try another window blend",
                    },
                ],
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp1",
                        "hypothesis_id": "h1",
                        "candidate_family": "window_blend",
                        "algorithm_design": {},
                        "execution_constraints": {"forbidden_data": []},
                        "evaluation_plan": {},
                    }
                ],
                "diversity_audit": {"near_duplicate_count": 1},
            }
        ),
        encoding="utf-8",
    )

    report = validate_hypothesis_algorithm_design(
        design_path=design_path,
        output_path=tmp_path / "audit.json",
        min_hypothesis_count=5,
        min_mechanism_family_count=4,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"
    assert report["decision"] == "reject"
    assert "insufficient_hypothesis_count" in report["violations"]
    assert "insufficient_mechanism_diversity" in report["violations"]
    assert "near_duplicate_hypotheses" in report["violations"]
    assert "blueprint_missing_forbidden_data_constraints" in report["violations"]
    assert "blueprint_missing_evaluation_plan" in report["violations"]


def test_algorithm_designer_adapts_ready_items_to_executor_schema(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        adapt_hypothesis_algorithm_sandbox_item_to_blueprint,
        build_hypothesis_algorithm_design,
        plan_hypothesis_algorithm_sandbox,
    )
    from pde_agent.core.hypothesis_lab import _iter_parameter_grid

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "FAILURE_MEMORY",
                "conclusion": (
                    "checkpoint static window blend failed, residual scaling "
                    "plateaued, and phase drift suggests retrieval alignment"
                ),
            }
        ),
        encoding="utf-8",
    )
    design_path = tmp_path / "design.json"
    build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[],
        output_path=design_path,
    )
    plan = plan_hypothesis_algorithm_sandbox(
        design_path=design_path,
        output_path=tmp_path / "plan.json",
        max_ready_items=6,
    )

    families = []
    for index, item in enumerate(plan["sandbox_items"]):
        blueprint = adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
            sandbox_item=item,
            output_path=tmp_path / f"adapted_{index}.json",
        )
        families.append(item["candidate_family"])
        rows = _iter_parameter_grid(
            blueprint["parameter_grid"],
            family=blueprint["candidate_family"],
        )
        assert rows

    assert families == [
        "uncertainty_gating",
        "residual_correction",
        "data_retrieval_nonlocal",
        "time_window_decomposition",
        "physics_constraint",
        "operator_structure",
    ]


def test_algorithm_designer_commands_return_artifact_payload(tmp_path: Path) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        build_hypothesis_algorithm_design_command,
        plan_hypothesis_algorithm_sandbox_command,
        record_hypothesis_algorithm_evidence_command,
        validate_hypothesis_algorithm_design_command,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "CHECKPOINT_FRONT_WINDOW_ADAPTER_FAILURE_MEMORY_RECORDED",
                "conclusion": (
                    "Static checkpoint blend failed; phase drift and checkpoint "
                    "unreliability require a learned or gated algorithm."
                ),
            }
        ),
        encoding="utf-8",
    )
    design_path = tmp_path / "design.json"
    build_payload = build_hypothesis_algorithm_design_command(
        output_path=str(design_path),
        failure_memory_paths=[str(memory_path)],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[],
        min_hypothesis_count=5,
        min_mechanism_family_count=4,
        force=False,
    )

    assert build_payload["status"] == "ok"
    assert build_payload["state_hint"] == "HYPOTHESIS_ALGORITHM_DESIGN_BUILT"
    assert build_payload["artifacts"]["algorithm_design_path"] == str(design_path)
    assert build_payload["metrics"]["mechanism_family_count"] >= 4

    audit_path = tmp_path / "audit.json"
    validate_payload = validate_hypothesis_algorithm_design_command(
        design_path=str(design_path),
        output_path=str(audit_path),
        min_hypothesis_count=5,
        min_mechanism_family_count=4,
        force=False,
    )

    assert validate_payload["status"] == "ok"
    assert validate_payload["state_hint"] == "HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"
    assert validate_payload["metrics"]["decision"] == "accept"

    sandbox_plan_payload = plan_hypothesis_algorithm_sandbox_command(
        design_path=str(design_path),
        output_path=str(tmp_path / "sandbox_plan.json"),
        max_ready_items=2,
        force=False,
    )

    assert sandbox_plan_payload["status"] == "ok"
    assert sandbox_plan_payload["state_hint"] == "HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED"
    assert sandbox_plan_payload["metrics"]["ready_count"] == 2
    assert sandbox_plan_payload["artifacts"]["sandbox_items"]

    sandbox_path = tmp_path / "sandbox.json"
    sandbox_path.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_VERIFIED",
                "blueprint_id": "bp_algorithm_designer_001_uncertainty_gating",
                "judge_report_path": "judge.json",
                "rules_passed": False,
                "score_delta": -1.0,
                "failure_codes": ["candidate_no_segment_improvement"],
            }
        ),
        encoding="utf-8",
    )
    evidence_payload = record_hypothesis_algorithm_evidence_command(
        design_path=str(design_path),
        sandbox_report_path=str(sandbox_path),
        output_path=str(tmp_path / "evidence.json"),
        force=False,
    )

    assert evidence_payload["status"] == "ok"
    assert evidence_payload["state_hint"] == "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED"
    assert evidence_payload["metrics"]["decision"] == "reject_with_failure_memory"


def test_algorithm_designer_command_accepts_explicit_evidence_paths(
    tmp_path: Path,
) -> None:
    from pde_agent.commands.hypothesis_lab_runtime import (
        build_hypothesis_algorithm_design_command,
    )

    evidence_path = tmp_path / "fno_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED",
                "source_type": "fno_training_optimization",
                "training_directive": {
                    "action": "continue_current_fno_family",
                    "target_family": "fno_lite",
                    "reason_codes": ["early_signal_learned"],
                },
                "selected_candidate": {
                    "candidate_id": "fno_train_candidate_002",
                    "config": {
                        "modes": 8,
                        "width": 32,
                        "depth": 1,
                        "loss_type": "early_weighted_mse",
                        "forecast_steps": 12,
                        "train_steps": 2,
                    },
                },
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": ["early_weighted_mse"],
                    "fno_training_configs": [],
                },
            }
        ),
        encoding="utf-8",
    )

    design_path = tmp_path / "design.json"
    payload = build_hypothesis_algorithm_design_command(
        output_path=str(design_path),
        failure_memory_paths=[],
        evidence_paths=[str(evidence_path)],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[],
        min_hypothesis_count=5,
        min_mechanism_family_count=4,
        force=False,
    )

    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["metrics"]["training_evidence_count"] == 1
    assert design["diagnosis"]["training_evidence_count"] == 1
    assert "fno_training_optimization" in payload["metrics"]["source_evidence_types"]
    assert any(
        "fno_training_optimization" in item["source_evidence_types"]
        for item in design["mechanism_hypotheses"]
    )


def test_algorithm_designer_uses_logs_strategy_and_literature_cards(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": "runtime plateau"}),
        encoding="utf-8",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "long-horizon instability and runtime bottleneck after repeated FNO search",
        encoding="utf-8",
    )
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "local_phase_alignment",
                        "mechanism_family": "data_retrieval_nonlocal",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    literature_path = tmp_path / "literature_card.json"
    literature_path.write_text(
        json.dumps(
            {
                "card_id": "paper_fno_stability",
                "mechanism_type": "physics_constraint",
                "claim": "Residual diagnostics can stabilize long rollouts.",
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[log_path],
        strategy_library_paths=[strategy_path],
        literature_card_paths=[literature_path],
        output_path=None,
        force=False,
    )

    diagnosis = report["diagnosis"]
    assert diagnosis["log_count"] == 1
    assert diagnosis["strategy_library_count"] == 1
    assert diagnosis["literature_card_count"] == 1
    assert any(
        source["source_type"] == "run_log" for source in diagnosis["evidence_summaries"]
    )
    assert any(
        source["source_type"] == "strategy_library"
        for source in diagnosis["evidence_summaries"]
    )
    assert any(
        source["source_type"] == "literature_card"
        for source in diagnosis["evidence_summaries"]
    )
    assert any(
        "run_log" in blueprint["source_evidence_types"]
        for blueprint in report["algorithm_blueprints"]
    )


def test_algorithm_designer_uses_evidence_constraints_to_reprioritize(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": "runtime plateau"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED",
                "decision": "record_near_miss",
                "algorithm_card": {"candidate_family": "operator_structure"},
                "next_generation_constraints": {
                    "prefer_mechanism_families": ["physics_constraint"],
                    "refine_mechanism_families": ["operator_structure"],
                    "near_miss_mechanism_families": ["residual_correction"],
                    "avoid_mechanism_families": ["uncertainty_gating"],
                    "quarantine_mechanism_families": ["data_retrieval_nonlocal"],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[evidence_path],
        output_path=tmp_path / "design.json",
    )

    rows = report["mechanism_hypotheses"]
    by_family = {item["mechanism_family"]: item for item in rows}
    assert rows[0]["mechanism_family"] == "physics_constraint"
    assert by_family["physics_constraint"]["next_generation_action"] == "prefer"
    assert by_family["operator_structure"]["next_generation_action"] == "refine"
    assert by_family["residual_correction"]["next_generation_action"] == "near_miss"
    assert by_family["uncertainty_gating"]["next_generation_action"] == "avoid"
    assert by_family["data_retrieval_nonlocal"]["next_generation_action"] == "quarantine"
    assert by_family["operator_structure"]["evidence_reuse"][
        "next_generation_constraints"
    ]["refine_mechanism_families"] == ["operator_structure"]


def test_algorithm_designer_consumes_task1_training_evidence_constraints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": "early score low"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "training_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_SHORT_WINDOW_TRAINING_EVIDENCE_RECORDED",
                "source_type": "task1_short_window_training",
                "model_family": "conv1d_residual",
                "decision": "continue_training",
                "metrics": {"early": {"rel_mse": 0.8, "score": 0.1}},
                "next_generation_constraints": {
                    "training_model_families": ["conv1d_residual"],
                    "prefer_training_objectives": ["short_window_mse"],
                    "refine_training_horizons": ["early_window"],
                    "avoid_training_model_families": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[evidence_path],
        output_path=None,
    )

    diagnosis = report["diagnosis"]
    assert diagnosis["training_evidence_count"] == 1
    assert any(
        source["source_type"] == "task1_short_window_training"
        for source in diagnosis["evidence_summaries"]
    )
    by_family = {
        item["mechanism_family"]: item for item in report["mechanism_hypotheses"]
    }
    assert by_family["operator_structure"]["next_generation_action"] == "prefer"
    assert (
        "task1_short_window_training"
        in by_family["operator_structure"]["source_evidence_types"]
    )
    assert by_family["time_window_decomposition"]["next_generation_action"] in {
        "prefer",
        "refine",
    }


def test_algorithm_designer_consumes_task1_family_selection_evidence(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        build_hypothesis_algorithm_design,
    )

    memory_path = tmp_path / "failure.json"
    memory_path.write_text(
        json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": "family plateau"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "family_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_SHORT_WINDOW_FAMILY_SELECTION_EVIDENCE_RECORDED",
                "source_type": "task1_short_window_family_selection",
                "selected_family": "fno_lite",
                "training_directive": {
                    "action": "continue_training",
                    "target_family": "fno_lite",
                    "reason_codes": ["best_early_score", "middle_guardrail_available"],
                },
                "family_records": [
                    {
                        "model_family": "conv1d_residual",
                        "early_score": 1.0,
                        "middle_guardrail": {"evaluated": False},
                    },
                    {
                        "model_family": "fno_lite",
                        "early_score": 2.0,
                        "middle_guardrail": {"evaluated": True},
                    },
                ],
                "next_generation_constraints": {
                    "training_model_families": ["fno_lite"],
                    "prefer_training_objectives": ["short_window_mse"],
                    "refine_training_horizons": ["middle_guardrail"],
                    "avoid_training_model_families": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hypothesis_algorithm_design(
        failure_memory_paths=[memory_path],
        log_paths=[],
        strategy_library_paths=[],
        literature_card_paths=[evidence_path],
        output_path=None,
    )

    diagnosis = report["diagnosis"]
    assert diagnosis["training_evidence_count"] == 1
    assert any(
        item["source_type"] == "task1_short_window_family_selection"
        for item in diagnosis["evidence_summaries"]
    )
    by_family = {
        item["mechanism_family"]: item for item in report["mechanism_hypotheses"]
    }
    assert by_family["operator_structure"]["next_generation_action"] == "prefer"
    assert (
        "task1_short_window_family_selection"
        in by_family["operator_structure"]["source_evidence_types"]
    )


def test_record_hypothesis_algorithm_evidence_writeback(tmp_path: Path) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        record_hypothesis_algorithm_evidence,
    )

    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp_residual",
                        "hypothesis_id": "h_residual",
                        "candidate_family": "residual_correction",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sandbox_path = tmp_path / "sandbox.json"
    sandbox_path.write_text(
        json.dumps(
            {
                "state_hint": "SANDBOX_VERIFIED",
                "blueprint_id": "bp_residual",
                "judge_report_path": "judge.json",
                "rules_passed": True,
                "score_delta": -0.1,
                "failure_codes": ["candidate_no_segment_improvement"],
            }
        ),
        encoding="utf-8",
    )

    report = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=sandbox_path,
        output_path=tmp_path / "writeback.json",
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED"
    assert report["decision"] == "reject_with_failure_memory"
    assert report["algorithm_card"]["evidence_status"] == "reject_with_failure_memory"
    assert report["evidence"]["judge_report_path"] == "judge.json"
    assert "candidate_no_segment_improvement" in report["failure_taxonomy"][
        "failure_codes"
    ]
    assert report["next_generation_constraints"]["avoid_mechanism_families"] == [
        "residual_correction"
    ]


def test_record_hypothesis_algorithm_evidence_maps_adapted_blueprint_id(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        record_hypothesis_algorithm_evidence,
    )

    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp_gate",
                        "hypothesis_id": "h_gate",
                        "candidate_family": "uncertainty_gating",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sandbox_path = tmp_path / "sandbox.json"
    sandbox_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
                "blueprint_id": "adapted_bp_gate",
                "verifier": {
                    "score_delta_vs_baseline": 0.25,
                    "rules_passed": True,
                },
                "artifacts": {"judge_report_path": "judge.json"},
                "adapted_blueprint": {
                    "source_algorithm_blueprint_id": "bp_gate",
                },
            }
        ),
        encoding="utf-8",
    )

    report = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=sandbox_path,
        output_path=None,
        force=False,
    )

    assert report["decision"] == "promote_to_strategy"
    assert report["algorithm_card"]["blueprint_id"] == "bp_gate"
    assert report["algorithm_card"]["evidence_status"] == "promote_to_strategy"


def test_record_hypothesis_algorithm_evidence_uses_multidimensional_rubric(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        record_hypothesis_algorithm_evidence,
    )

    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp_gate",
                        "hypothesis_id": "h_gate",
                        "candidate_family": "uncertainty_gating",
                    },
                    {
                        "blueprint_id": "bp_residual",
                        "hypothesis_id": "h_residual",
                        "candidate_family": "residual_correction",
                    },
                    {
                        "blueprint_id": "bp_physics",
                        "hypothesis_id": "h_physics",
                        "candidate_family": "physics_constraint",
                    },
                    {
                        "blueprint_id": "bp_operator",
                        "hypothesis_id": "h_operator",
                        "candidate_family": "operator_structure",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def write_sandbox(name: str, payload: dict[str, object]) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    promote_path = write_sandbox(
        "promote",
        _sandbox_payload(
            blueprint_id="bp_gate",
            score_delta=1.5,
            early_delta=0.4,
            middle_delta=0.1,
            late_delta=0.2,
            inference_time=0.5,
        ),
    )
    refine_path = write_sandbox(
        "refine",
        _sandbox_payload(
            blueprint_id="bp_residual",
            score_delta=-0.02,
            early_delta=0.3,
            middle_delta=0.0,
            late_delta=0.0,
            inference_time=0.5,
        ),
    )
    near_miss_path = write_sandbox(
        "near_miss",
        _sandbox_payload(
            blueprint_id="bp_operator",
            score_delta=0.03,
            early_delta=0.0,
            middle_delta=0.0,
            late_delta=0.0,
            inference_time=0.5,
        ),
    )
    reject_path = write_sandbox(
        "reject",
        _sandbox_payload(
            blueprint_id="bp_operator",
            score_delta=-0.5,
            early_delta=-0.2,
            middle_delta=-0.1,
            late_delta=-0.1,
            inference_time=0.5,
            repeated_failure_similarity=0.92,
        ),
    )
    quarantine_path = write_sandbox(
        "quarantine",
        _sandbox_payload(
            blueprint_id="bp_physics",
            score_delta=1.0,
            early_delta=0.1,
            middle_delta=0.1,
            late_delta=0.1,
            inference_time=130.0,
            context_preserved=False,
            uses_target_as_reference=True,
        ),
    )

    promote = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=promote_path,
        output_path=None,
    )
    refine = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=refine_path,
        output_path=None,
    )
    near_miss = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=near_miss_path,
        output_path=None,
    )
    reject = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=reject_path,
        output_path=None,
    )
    quarantine = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=quarantine_path,
        output_path=None,
    )

    assert promote["decision"] == "promote_to_strategy"
    assert refine["decision"] == "promote_to_refinement"
    assert near_miss["decision"] == "record_near_miss"
    assert reject["decision"] == "reject_with_failure_memory"
    assert quarantine["decision"] == "quarantine"
    assert promote["evidence"]["primary_improvement_axis"] == "early"
    assert reject["evidence"]["primary_failure_axis"] == "early"
    assert reject["evidence"]["repeated_failure_similarity"] == 0.92
    assert "residual_correction" in refine["next_generation_constraints"][
        "refine_mechanism_families"
    ]
    assert "operator_structure" in near_miss["next_generation_constraints"][
        "near_miss_mechanism_families"
    ]
    assert "operator_structure" in reject["next_generation_constraints"][
        "avoid_mechanism_families"
    ]
    assert "physics_constraint" in quarantine["next_generation_constraints"][
        "quarantine_mechanism_families"
    ]
    assert "repeated_failure_similarity_high" in reject["failure_taxonomy"][
        "failure_codes"
    ]


def test_record_hypothesis_algorithm_evidence_identifies_guardrail_cases(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        record_hypothesis_algorithm_evidence,
    )

    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp_gate",
                        "hypothesis_id": "h_gate",
                        "candidate_family": "uncertainty_gating",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    early_better_late_worse = tmp_path / "early_better_late_worse.json"
    early_better_late_worse.write_text(
        json.dumps(
            _sandbox_payload(
                blueprint_id="bp_gate",
                score_delta=-0.01,
                early_delta=0.2,
                middle_delta=0.0,
                late_delta=-0.2,
                inference_time=0.5,
            )
        ),
        encoding="utf-8",
    )
    runtime_regression = tmp_path / "runtime_regression.json"
    runtime_regression.write_text(
        json.dumps(
            _sandbox_payload(
                blueprint_id="bp_gate",
                score_delta=0.02,
                early_delta=0.0,
                middle_delta=0.0,
                late_delta=0.0,
                inference_time=130.0,
            )
        ),
        encoding="utf-8",
    )
    physics_regression = tmp_path / "physics_regression.json"
    physics_payload = _sandbox_payload(
        blueprint_id="bp_gate",
        score_delta=0.02,
        early_delta=0.0,
        middle_delta=0.0,
        late_delta=0.0,
        inference_time=0.5,
    )
    physics_payload["verifier"]["metric_results"]["physics_residual_rmse_delta"] = {
        "value": -0.2,
        "passed": False,
    }
    physics_regression.write_text(json.dumps(physics_payload), encoding="utf-8")

    refine = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=early_better_late_worse,
        output_path=None,
    )
    runtime = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=runtime_regression,
        output_path=None,
    )
    physics = record_hypothesis_algorithm_evidence(
        design_path=design_path,
        sandbox_report_path=physics_regression,
        output_path=None,
    )

    assert refine["decision"] == "promote_to_refinement"
    assert refine["evidence"]["primary_improvement_axis"] == "early"
    assert refine["evidence"]["primary_failure_axis"] == "late"
    assert runtime["decision"] == "quarantine"
    assert "runtime_guard_failed" in runtime["failure_taxonomy"]["failure_codes"]
    assert physics["decision"] == "quarantine"
    assert "physics_guard_failed" in physics["failure_taxonomy"]["failure_codes"]


def _sandbox_payload(
    *,
    blueprint_id: str,
    score_delta: float,
    early_delta: float,
    middle_delta: float,
    late_delta: float,
    inference_time: float,
    context_preserved: bool = True,
    uses_target_as_reference: bool = False,
    repeated_failure_similarity: float = 0.0,
) -> dict[str, object]:
    return {
        "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
        "blueprint_id": blueprint_id,
        "reference_audit": {"uses_target_as_reference": uses_target_as_reference},
        "failure_memory_similarity": {"max_similarity": repeated_failure_similarity},
        "verifier": {
            "rules_passed": context_preserved and inference_time <= 120.0,
            "score_delta_vs_baseline": score_delta,
            "metric_results": {
                "context_preserved": {
                    "value": context_preserved,
                    "passed": context_preserved,
                },
                "score_delta_vs_baseline": {
                    "value": score_delta,
                    "passed": score_delta > 0.0,
                },
                "early_score_delta_vs_baseline": {
                    "value": early_delta,
                    "passed": early_delta >= 0.0,
                },
                "middle_score_guardrail": {
                    "value": middle_delta,
                    "passed": middle_delta >= 0.0,
                },
                "late_rmse_guardrail": {
                    "value": late_delta,
                    "passed": late_delta >= 0.0,
                },
                "inference_time_budget": {
                    "value": inference_time,
                    "threshold": 120.0,
                    "passed": inference_time <= 120.0,
                },
                "physics_residual_rmse_delta": {
                    "value": None,
                    "passed": True,
                },
            },
        },
        "artifacts": {"judge_report_path": "judge.json"},
    }


def test_plan_hypothesis_algorithm_sandbox_routes_runnable_blueprints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        plan_hypothesis_algorithm_sandbox,
    )

    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_DESIGN_BUILT",
                "algorithm_blueprints": [
                    {
                        "blueprint_id": "bp_gate",
                        "hypothesis_id": "h_gate",
                        "candidate_family": "uncertainty_gating",
                        "algorithm_design": {
                            "inputs": ["task_context", "safe_baseline_prediction"],
                            "outputs": ["task1_candidate_prediction"],
                            "complexity_budget": {"sandbox_level": 0},
                        },
                        "execution_constraints": {
                            "forbidden_data": [
                                "task1_target_as_prediction_source",
                                "local_validation_target_as_training_reference_for_submission",
                            ],
                            "bounded_executor_required": True,
                        },
                        "evaluation_plan": {
                            "expected_observables": ["gate selects source"],
                            "segment_level_metrics": ["rel_mse_20_35"],
                            "physics_diagnostics": ["burgers_residual_mean"],
                            "runtime_metrics": ["inference_time_a100_equivalent"],
                            "ablation_plan": ["disable gate"],
                            "falsification_criteria": ["no segment improvement"],
                        },
                    },
                    {
                        "blueprint_id": "bp_unknown",
                        "hypothesis_id": "h_unknown",
                        "candidate_family": "unknown_new_executor",
                        "algorithm_design": {"inputs": [], "outputs": []},
                        "execution_constraints": {"bounded_executor_required": True},
                        "evaluation_plan": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = plan_hypothesis_algorithm_sandbox(
        design_path=design_path,
        output_path=tmp_path / "sandbox_plan.json",
        max_ready_items=5,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED"
    assert report["ready_count"] == 1
    assert report["blocked_count"] == 1
    ready = report["sandbox_items"][0]
    assert ready["blueprint_id"] == "bp_gate"
    assert ready["executor_route"] == "run-blueprint-sandbox"
    assert ready["executor_profile"] == "bounded_numpy_postprocess"
    assert ready["non_arbitrary_code_guard"] is True
    assert ready["required_inputs"] == [
        "context_path",
        "target_path",
        "initial_submission_dir",
        "output_dir",
    ]
    blocked = report["blocked_items"][0]
    assert blocked["blueprint_id"] == "bp_unknown"
    assert blocked["block_reason"] == "unsupported_executor_route"


def test_adapt_hypothesis_algorithm_sandbox_item_to_controlled_blueprint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer import (
        adapt_hypothesis_algorithm_sandbox_item_to_blueprint,
    )

    item = {
        "blueprint_id": "bp_gate",
        "hypothesis_id": "h_gate",
        "candidate_family": "uncertainty_gating",
        "executor_profile": "bounded_numpy_postprocess",
        "source_blueprint": {
            "algorithm_design": {"inputs": ["task_context"]},
            "evaluation_plan": {"expected_observables": ["gate"]},
        },
    }

    blueprint = adapt_hypothesis_algorithm_sandbox_item_to_blueprint(
        sandbox_item=item,
        output_path=tmp_path / "adapted.json",
        force=False,
    )

    assert blueprint["state_hint"] == "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED"
    assert blueprint["source_algorithm_blueprint_id"] == "bp_gate"
    assert blueprint["candidate_family"] == "task1_early_refinement"
    assert blueprint["sandbox"]["max_inference_seconds"] > 0
    assert blueprint["parameter_grid"]["candidate_params"]
    assert blueprint["algorithm_designer_metadata"]["mechanism_family"] == (
        "uncertainty_gating"
    )
    assert blueprint["algorithm_designer_metadata"]["executor_profile"] == (
        "bounded_numpy_postprocess"
    )


def test_run_hypothesis_algorithm_sandbox_command_invokes_controlled_sandbox(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.commands import hypothesis_lab_runtime as command_module

    blueprint_path = tmp_path / "adapted.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "state_hint": "HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED",
                "blueprint_id": "adapted_bp_gate",
                "source_algorithm_blueprint_id": "bp_gate",
                "candidate_family": "task1_early_refinement",
                "parameter_grid": {"candidate_params": [{"k": 4}]},
                "sandbox": {
                    "max_inference_seconds": 120.0,
                    "promotion_min_score_delta": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_blueprint_sandbox(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
            "blueprint_id": "adapted_bp_gate",
            "best_candidate": {"candidate_id": "candidate_a"},
            "verifier": {
                "decision": "revise_hypothesis",
                "score_delta_vs_baseline": -0.1,
            },
            "artifacts": {
                "sandbox_report_path": str(tmp_path / "sandbox_report.json"),
                "judge_report_path": str(tmp_path / "judge.json"),
                "best_submission_dir": str(tmp_path / "submission"),
            },
        }

    import pde_agent.commands.hypothesis_lab_runtime_algorithm as _algo_mod

    monkeypatch.setattr(
        _algo_mod,
        "run_blueprint_sandbox",
        fake_run_blueprint_sandbox,
    )

    payload = command_module.run_hypothesis_algorithm_sandbox_command(
        adapted_blueprint_path=str(blueprint_path),
        context_path="context.hdf5",
        target_path="target.hdf5",
        output_dir=str(tmp_path / "sandbox"),
        submission_id="team_test",
        reference_path=None,
        reference_sample_limit=None,
        timing_profile=None,
        timing_factor=None,
        force=False,
    )

    assert payload["state_hint"] == "HYPOTHESIS_ALGORITHM_SANDBOX_RAN"
    assert payload["metrics"]["decision"] == "revise_hypothesis"
    assert calls[0]["blueprint"]["blueprint_id"] == "adapted_bp_gate"
    assert calls[0]["context_path"] == Path("context.hdf5")
