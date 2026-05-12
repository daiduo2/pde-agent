from __future__ import annotations

import json
from pathlib import Path


def test_candidate_family_card_validation_rejects_unbounded_values() -> None:
    from pde_agent.core.task1_candidate_factory import (
        CandidateFactoryError,
        validate_candidate_family_card,
    )

    valid = _candidate_card("task1_front25_dynamics_v0")
    normalized = validate_candidate_family_card(valid)

    assert normalized["state_hint"] == "TASK1_CANDIDATE_FAMILY_CARD_V1"
    assert normalized["candidate_family_id"] == "task1_front25_dynamics_v0"
    assert normalized["mechanism"] == "front25_dynamics"

    invalid = dict(valid)
    invalid["sandbox_budget"] = "unbounded"
    try:
        validate_candidate_family_card(invalid)
    except CandidateFactoryError as exc:
        assert "sandbox_budget" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid card should fail validation")


def test_generate_task1_candidate_families_from_stagnated_benchmark(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        generate_task1_candidate_families,
    )

    benchmark_path = _write_benchmark(
        tmp_path / "benchmark.json",
        decision="infrastructure_only",
        stagnation_count=4,
    )
    chain_path = _write_chain_report(tmp_path / "chain.json")
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "candidate_families": [
                    _candidate_card("task1_front25_dynamics_v0"),
                ]
            }
        ),
        encoding="utf-8",
    )

    report = generate_task1_candidate_families(
        benchmark_path=benchmark_path,
        chain_report_path=chain_path,
        output_path=tmp_path / "families.json",
        prior_registry_path=prior_path,
        max_families=5,
        force=False,
    )

    assert report["state_hint"] == "TASK1_CANDIDATE_FAMILIES_GENERATED"
    assert report["decision"] == "generated"
    assert report["candidate_family_count"] == 5
    mechanisms = {card["mechanism"] for card in report["candidate_families"]}
    assert mechanisms == {
        "front25_dynamics",
        "residual_corrector",
        "spectral_physics",
        "checkpoint_adapter",
        "fno_spectral_truncation",
    }
    ids = {card["candidate_family_id"] for card in report["candidate_families"]}
    assert "task1_front25_dynamics_v0" not in ids
    assert "task1_front25_dynamics_v1" in ids
    assert report["inputs"]["benchmark_decision"] == "infrastructure_only"
    assert report["generation_context"]["score_delta"] == 0.0


def test_generate_task1_candidate_families_accepts_minimal_chain_report(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        generate_task1_candidate_families,
    )

    chain_path = tmp_path / "minimal_chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "needs_new_dynamics_source",
            }
        ),
        encoding="utf-8",
    )

    report = generate_task1_candidate_families(
        benchmark_path=_write_benchmark(tmp_path / "benchmark.json"),
        chain_report_path=chain_path,
        output_path=tmp_path / "families.json",
        max_families=2,
        force=False,
    )

    assert report["decision"] == "generated"
    assert report["generation_context"]["candidate_space_exhausted_count"] == 5


def test_generate_task1_candidate_families_includes_fno_skill_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        generate_task1_candidate_families,
        run_candidate_family_sandbox,
    )

    family_report = generate_task1_candidate_families(
        benchmark_path=_write_benchmark(tmp_path / "benchmark.json"),
        chain_report_path=_write_chain_report(tmp_path / "chain.json"),
        output_path=tmp_path / "families.json",
        max_families=9,
        force=False,
    )
    by_mechanism = {
        card["mechanism"]: card for card in family_report["candidate_families"]
    }

    assert set(by_mechanism) >= {
        "fno_spectral_truncation",
        "fno_modewise_correction",
        "fno_rollout_stabilization",
        "fno_multistep_consistency",
    }
    assert by_mechanism["fno_spectral_truncation"]["implementation_hooks"] == [
        "official_checkpoint_family"
    ]
    assert by_mechanism["fno_modewise_correction"]["target_segment"] == "full"
    assert by_mechanism["fno_rollout_stabilization"]["target_segment"] == "late"
    assert by_mechanism["fno_multistep_consistency"]["sandbox_budget"] == "medium"

    sandbox = run_candidate_family_sandbox(
        family_report_path=family_report["family_report_path"],
        output_path=tmp_path / "sandbox.json",
        force=False,
    )
    planned_by_mechanism = {
        item["mechanism"]: item
        for item in sandbox["sandbox_items"]
        if item["status"] == "planned"
    }
    for mechanism in {
        "fno_spectral_truncation",
        "fno_modewise_correction",
        "fno_rollout_stabilization",
        "fno_multistep_consistency",
    }:
        assert planned_by_mechanism[mechanism]["next_command"] == (
            "run-checkpoint-family-sandbox"
        )
        assert planned_by_mechanism[mechanism]["candidate_options"][
            "parameter_space"
        ]


def test_candidate_factory_includes_physics_and_literature_algorithm_families(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        generate_task1_candidate_families,
        run_candidate_family_sandbox,
    )

    family_report = generate_task1_candidate_families(
        benchmark_path=_write_benchmark(tmp_path / "benchmark.json"),
        chain_report_path=_write_chain_report(tmp_path / "chain.json"),
        output_path=tmp_path / "families.json",
        max_families=12,
        force=False,
    )
    by_mechanism = {
        card["mechanism"]: card for card in family_report["candidate_families"]
    }

    assert set(by_mechanism) >= {
        "pinn_physics_residual_judge",
        "pi_deeponet_correction_signal",
        "literature_algorithm_blueprint",
    }
    assert by_mechanism["pinn_physics_residual_judge"]["implementation_hooks"] == [
        "task1_physics_residual_refinement"
    ]
    assert by_mechanism["pi_deeponet_correction_signal"]["implementation_hooks"] == [
        "task1_pi_deeponet_correction_head"
    ]
    assert by_mechanism["literature_algorithm_blueprint"]["implementation_hooks"] == [
        "literature_algorithm_v2_pipeline"
    ]

    sandbox = run_candidate_family_sandbox(
        family_report_path=family_report["family_report_path"],
        output_path=tmp_path / "sandbox.json",
        force=False,
    )
    planned_by_mechanism = {
        item["mechanism"]: item
        for item in sandbox["sandbox_items"]
        if item["status"] == "planned"
    }
    assert planned_by_mechanism["pinn_physics_residual_judge"]["next_command"] == (
        "run-hypothesis-lab"
    )
    assert planned_by_mechanism["pi_deeponet_correction_signal"]["next_command"] == (
        "run-hypothesis-lab"
    )
    assert planned_by_mechanism["literature_algorithm_blueprint"]["next_command"] == (
        "run-literature-algorithm-v2-pipeline"
    )
    literature_chain = planned_by_mechanism["literature_algorithm_blueprint"][
        "evaluation_chain"
    ]
    assert [step["command"] for step in literature_chain] == [
        "validate-algorithm-cards-v2",
        "compile-algorithm-candidates",
        "validate-executable-blueprints",
        "run-bounded-executor-sandbox",
        "record-bounded-executor-sandbox-event",
    ]
    assert "bounded_executor_sandbox_path" in planned_by_mechanism[
        "literature_algorithm_blueprint"
    ]["required_evidence"]


def test_run_candidate_family_sandbox_plans_supported_and_blocked_hooks(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        run_candidate_family_sandbox,
    )

    family_report_path = tmp_path / "families.json"
    family_report_path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_CANDIDATE_FAMILIES_GENERATED",
                "candidate_families": [
                    _candidate_card("task1_front25_dynamics_v0"),
                    {
                        **_candidate_card("task1_unknown_v0"),
                        "candidate_family_id": "task1_unknown_v0",
                        "mechanism": "residual_corrector",
                        "implementation_hooks": ["unregistered-hook"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_candidate_family_sandbox(
        family_report_path=family_report_path,
        output_path=tmp_path / "sandbox.json",
        force=False,
    )

    assert report["state_hint"] == "TASK1_CANDIDATE_FAMILY_SANDBOX_PLANNED"
    assert report["decision"] == "planned"
    assert report["planned_count"] == 1
    assert report["blocked_count"] == 1
    by_id = {item["candidate_family_id"]: item for item in report["sandbox_items"]}
    assert by_id["task1_front25_dynamics_v0"]["status"] == "planned"
    assert by_id["task1_front25_dynamics_v0"]["next_command"] == (
        "run-hypothesis-lab"
    )
    assert by_id["task1_unknown_v0"]["status"] == "blocked"
    assert by_id["task1_unknown_v0"]["blocked_reason"] == "no_supported_hook"


def test_evaluate_candidate_factory_capability_passes_on_diverse_executable_space(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        evaluate_candidate_factory_capability,
        generate_task1_candidate_families,
        run_candidate_family_sandbox,
    )

    family_report = generate_task1_candidate_families(
        benchmark_path=_write_benchmark(tmp_path / "benchmark.json"),
        chain_report_path=_write_chain_report(tmp_path / "chain.json"),
        output_path=tmp_path / "families.json",
        max_families=5,
        force=False,
    )
    sandbox_report = run_candidate_family_sandbox(
        family_report_path=family_report["family_report_path"],
        output_path=tmp_path / "sandbox.json",
        force=False,
    )

    report = evaluate_candidate_factory_capability(
        family_report_path=family_report["family_report_path"],
        sandbox_report_path=sandbox_report["sandbox_report_path"],
        output_path=tmp_path / "factory_benchmark.json",
        min_family_count=4,
        min_executable_count=3,
        min_mechanism_count=4,
        force=False,
    )

    assert report["state_hint"] == "TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED"
    assert report["decision"] == "passed"
    assert report["candidate_space"]["family_count"] == 5
    assert report["candidate_space"]["mechanism_count"] == 5
    assert report["sandbox_readiness"]["planned_count"] >= 3
    assert report["capability_checks"]["executable_families"]["passed"] is True
    assert "heartbeat_handoff" not in report
    assert report["agent_runtime_handoff"] == {
        "ready": True,
        "next_command": "run-claude-agent-sdk-runtime",
        "active_substrate": "claude_agent_sdk",
        "candidate_generator_fallback": "auto",
        "sandbox_report_path": str(tmp_path / "sandbox.json"),
        "planned_candidate_family_ids": [
            item["candidate_family_id"]
            for item in sandbox_report["sandbox_items"]
            if item["status"] == "planned"
        ],
    }
    assert report["next_command"] == "run-claude-agent-sdk-runtime"


def test_materialize_candidate_family_item_outputs_algorithm_candidates(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task1_candidate_factory import (
        generate_task1_candidate_families,
        materialize_candidate_family_item,
        run_candidate_family_sandbox,
    )
    from pde_agent.core.hypothesis_lab import build_knowledge_cards, generate_hypotheses

    family_report = generate_task1_candidate_families(
        benchmark_path=_write_benchmark(tmp_path / "benchmark.json"),
        chain_report_path=_write_chain_report(tmp_path / "chain.json"),
        output_path=tmp_path / "families.json",
        max_families=2,
        force=False,
    )
    sandbox_report = run_candidate_family_sandbox(
        family_report_path=family_report["family_report_path"],
        output_path=tmp_path / "sandbox.json",
        force=False,
    )

    materialized = materialize_candidate_family_item(
        sandbox_report_path=sandbox_report["sandbox_report_path"],
        output_path=tmp_path / "algorithm_candidates.json",
        candidate_family_id="task1_front25_dynamics_v0",
        force=False,
    )

    candidate = materialized["algorithm_candidates"][0]
    assert materialized["state_hint"] == "ALGORITHM_CANDIDATES_COMPILED"
    assert materialized["selected_hook"] == "hypothesis_lab_front25_dynamics_smoke"
    assert candidate["algorithm_candidate_id"] == (
        "candidate_task1_front25_dynamics_v0"
    )
    assert candidate["executor_spec"]["implementation_hooks"] == [
        "run-hypothesis-lab",
        "hypothesis_lab_front25_dynamics_smoke",
    ]
    cards = build_knowledge_cards(
        manual_report_paths=[],
        algorithm_candidate_paths=[materialized["algorithm_candidates_path"]],
        output_path=None,
        force=False,
    )
    hypotheses = generate_hypotheses(cards=cards, output_path=None, force=False)
    assert hypotheses["hypotheses"][0]["candidate_family"] == "task1_front25_dynamics"


def test_task1_candidate_factory_cli_and_tool_catalog(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog

    benchmark_path = _write_benchmark(tmp_path / "benchmark.json")
    chain_path = _write_chain_report(tmp_path / "chain.json")
    family_path = tmp_path / "families.json"
    sandbox_path = tmp_path / "sandbox.json"
    factory_benchmark_path = tmp_path / "factory_benchmark.json"

    runner = CliRunner()
    generated = runner.invoke(
        app,
        [
            "generate-task1-candidate-families",
            "--benchmark-path",
            str(benchmark_path),
            "--chain-report-path",
            str(chain_path),
            "--output-path",
            str(family_path),
            "--max-families",
            "5",
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert json.loads(generated.output)["artifacts"]["candidate_family_report_json"] == (
        str(family_path)
    )

    sandboxed = runner.invoke(
        app,
        [
            "run-candidate-family-sandbox",
            "--family-report-path",
            str(family_path),
            "--output-path",
            str(sandbox_path),
        ],
    )
    assert sandboxed.exit_code == 0, sandboxed.output
    assert json.loads(sandboxed.output)["artifacts"]["candidate_family_sandbox_json"] == (
        str(sandbox_path)
    )

    evaluated = runner.invoke(
        app,
        [
            "evaluate-candidate-factory-capability",
            "--family-report-path",
            str(family_path),
            "--sandbox-report-path",
            str(sandbox_path),
            "--output-path",
            str(factory_benchmark_path),
            "--min-family-count",
            "4",
            "--min-executable-count",
            "3",
            "--min-mechanism-count",
            "4",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.output)["state_hint"] == (
        "TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED"
    )

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "catalog.json",
        force=False,
    )
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["generate-task1-candidate-families"]["risk_level"] == (
        "artifact_write"
    )
    assert tools["run-candidate-family-sandbox"]["risk_level"] == "sandbox"
    assert tools["evaluate-candidate-factory-capability"]["risk_level"] == (
        "read_only"
    )
    assert tools["build-claude-literature-candidate-closure-task"]["risk_level"] == (
        "artifact_write"
    )
    assert "reusable agent workspace" in tools[
        "build-claude-literature-candidate-closure-task"
    ]["agent_guidance"]
    assert tools["record-agent-algorithm-cards"]["risk_level"] == "artifact_write"
    assert tools["record-agent-algorithm-cards"]["state_hints"] == [
        "ALGORITHM_CARDS_EXTRACTED"
    ]
    assert "paper_parse_path" in tools[
        "build-claude-literature-candidate-closure-task"
    ]["required_inputs"]


def _candidate_card(candidate_family_id: str) -> dict[str, object]:
    return {
        "state_hint": "TASK1_CANDIDATE_FAMILY_CARD_V1",
        "candidate_family_id": candidate_family_id,
        "family_name": "Front25 dynamics source",
        "task": "task1",
        "target_segment": "early",
        "mechanism": "front25_dynamics",
        "hypothesis": "Improve the first 25 predicted steps with a bounded dynamics source.",
        "parameter_space": {
            "window": [10, 15, 20, 25],
            "blend": [0.1, 0.2, 0.35],
        },
        "implementation_hooks": ["hypothesis_lab_front25_dynamics_smoke"],
        "expected_observables": ["early_rel_mse_down", "runtime_under_budget"],
        "risk_factors": ["early_oversmoothing"],
        "sandbox_budget": "quick",
        "source": {
            "type": "agent_hypothesis",
            "artifact_paths": ["runs/example/benchmark.json"],
        },
        "status": "proposed",
    }


def _write_benchmark(
    path: Path,
    *,
    decision: str = "infrastructure_only",
    stagnation_count: int = 4,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_AGENT_CAPABILITY_BENCHMARKED",
                "decision": decision,
                "score_progress": {
                    "best_score": 120.09092223481395,
                    "score_delta": 0.0,
                    "stagnation_count": stagnation_count,
                },
                "candidate_generation": {
                    "unique_candidate_families": [
                        "postprocess",
                        "sem_window_blend",
                    ],
                    "unique_candidate_family_count": 2,
                },
                "sandbox_validation": {
                    "outcome_counts": {
                        "candidate_space_exhausted": 5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_chain_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {
                    "candidate_space_exhausted_count": 5,
                    "failed_audit_count": 0,
                    "completed_iteration_count": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
