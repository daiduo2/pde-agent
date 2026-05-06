from __future__ import annotations

import json
from pathlib import Path


def test_build_hermes_research_directive_prefers_spectral_postprocess_after_score_gain(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_research_directive import (
        build_hermes_research_directive,
    )

    directive = build_hermes_research_directive(
        task1_benchmark_path=_write_task1_benchmark(
            tmp_path / "task1_benchmark.json",
            decision="passed",
            score_delta=0.0995,
            best_score=120.1904,
        ),
        candidate_factory_benchmark_path=_write_factory_benchmark(
            tmp_path / "factory_benchmark.json"
        ),
        output_path=tmp_path / "directive.json",
        recent_judge_report_paths=[
            _write_judge(tmp_path / "round_005_judge.json", score=120.1904),
        ],
        force=False,
    )

    assert directive["state_hint"] == "HERMES_RESEARCH_DIRECTIVE_BUILT"
    assert directive["decision"] == "run_directed_experiment"
    assert directive["directive"]["candidate_generator"] == "postprocess"
    assert directive["directive"]["learning_objective"] == "score"
    assert directive["directive"]["research_focus"] == "spectral_postprocess_refinement"
    assert directive["directive"]["forbidden_actions"] == [
        "manual_parameter_selection_without_sandbox",
        "repeat_failed_candidate_family_without_mutation",
        "promote_without_evidence",
    ]
    assert directive["directive"]["guard_conflict_policy"] == {
        "allowed_candidate_generators": ["postprocess"],
        "max_reprobe_batches": 1,
        "mode": "evidence_bounded_reprobe",
        "reason": "positive_task1_benchmark_delta",
    }
    assert directive["directive"]["experiment_design"]["minimum_trials"] == 3
    assert directive["next_command"] == "run-hermes-runtime-audit-chain"


def test_runtime_closed_loop_applies_research_directive_to_heartbeat(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    directive_path = _write_directive(
        tmp_path / "directive.json",
        candidate_generator="hypothesis_lab_early_subsegment_patch_search",
        learning_objective="score",
        candidate_options={
            "source_stability_guard": True,
            "handoff_smoothing_weight": 0.2,
        },
    )
    heartbeat_calls: list[dict[str, object]] = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "hypothesis_candidate_options": kwargs.get(
                    "hypothesis_candidate_options"
                )
            },
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": kwargs["candidate_generator"],
                "score": 121.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 121.0,
            "best_objective": "score",
            "best_objective_score": 121.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
            "next_command": "run-hermes-heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    reference = _write_hdf5_reference(tmp_path / "task1_reference.hdf5")

    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        learning_objective="runtime",
        candidate_generator_fallback="auto",
        max_batches=1,
        task1_reference_path=reference,
        research_directive_path=directive_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert heartbeat_calls[0]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert heartbeat_calls[0]["learning_objective"] == "score"
    assert heartbeat_calls[0]["hypothesis_candidate_options"] == {
        "source_stability_guard": True,
        "handoff_smoothing_weight": 0.2,
    }
    assert report["artifacts"]["research_directive_path"] == str(directive_path)
    assert report["research_directive"]["directive"]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )


def test_runtime_closed_loop_forwards_directive_guard_override_to_heartbeat(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    directive_path = _write_directive(
        tmp_path / "directive.json",
        candidate_generator="postprocess",
        learning_objective="score",
        candidate_options={},
        guard_conflict_policy={
            "mode": "evidence_bounded_reprobe",
            "allowed_candidate_generators": ["postprocess"],
            "max_reprobe_batches": 1,
            "reason": "positive_task1_benchmark_delta",
        },
    )
    heartbeat_calls: list[dict[str, object]] = []

    def fake_heartbeat_runner(**kwargs):
        heartbeat_calls.append(kwargs)
        report_path = Path(kwargs["output_dir"]) / "hermes_heartbeat_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        best_dir = tmp_path / "best"
        best_dir.mkdir(exist_ok=True)
        report = {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "parameters": {
                "candidate_generator_guard_overrides": kwargs.get(
                    "candidate_generator_guard_overrides"
                )
            },
            "batch_count": 1,
            "best_candidate": {
                "candidate_id": "best_runtime",
                "family": "postprocess",
                "score": 121.0,
                "submission_dir": str(best_dir),
            },
            "best_score": 121.0,
            "best_objective": "score",
            "best_objective_score": 121.0,
            "stop_reason": "max_batches_exhausted",
            "decision": "continue_next_heartbeat",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime_loop",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="fast_baseline_family",
        learning_objective="runtime",
        candidate_generator_fallback="auto",
        max_batches=1,
        research_directive_path=directive_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert heartbeat_calls[0]["candidate_generator_guard_overrides"] == [
        "postprocess"
    ]
    assert report["candidate_plan"]["directive_guard_overrides"] == ["postprocess"]
    assert report["metrics"]["research_directive_guard_override_count"] == 1


def test_evaluate_research_directive_followthrough_passes_when_runtime_obeys(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_research_directive import (
        evaluate_research_directive_followthrough,
    )

    directive_path = _write_directive(
        tmp_path / "directive.json",
        candidate_generator="postprocess",
        learning_objective="score",
        candidate_options={},
    )
    chain_report_path = tmp_path / "chain.json"
    chain_report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {
                    "completed_iteration_count": 1,
                    "failed_audit_count": 0,
                    "last_heartbeat_best_score": 120.1904,
                },
                "iteration_reports": [
                    {
                        "runtime_report_path": str(tmp_path / "runtime.json"),
                        "audit_report_path": str(tmp_path / "audit.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_report_path = tmp_path / "runtime.json"
    runtime_report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "candidate_plan": {
                    "candidate_generator_schedule": ["postprocess"],
                },
                "artifacts": {
                    "heartbeat_report_path": str(tmp_path / "heartbeat.json"),
                },
                "metrics": {"recorded_candidate_attempt_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "audit.json").write_text(
        json.dumps({"decision": "passed", "failed_checks": []}),
        encoding="utf-8",
    )
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "parameters": {"hypothesis_candidate_options": {}},
                "best_candidate": {"family": "postprocess"},
                "best_score": 120.1904,
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_research_directive_followthrough(
        directive_path=directive_path,
        chain_report_path=chain_report_path,
        output_path=tmp_path / "followthrough.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
    assert report["decision"] == "passed"
    assert report["checks"]["candidate_generator_followed"]["passed"] is True
    assert report["checks"]["audit_clean"]["passed"] is True


def test_evaluate_research_directive_followthrough_reports_guard_conflict(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_research_directive import (
        evaluate_research_directive_followthrough,
    )

    directive_path = _write_directive(
        tmp_path / "directive.json",
        candidate_generator="postprocess",
        learning_objective="score",
        candidate_options={},
    )
    chain_report_path = tmp_path / "chain.json"
    runtime_report_path = tmp_path / "runtime.json"
    audit_report_path = tmp_path / "audit.json"
    chain_report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0},
                "iterations": [
                    {
                        "runtime_report_path": str(runtime_report_path),
                        "audit_report_path": str(audit_report_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_report_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "candidate_plan": {
                    "candidate_generator_schedule": ["fast_baseline_family"],
                    "selection_rationale": [
                        {
                            "candidate_generator": "postprocess",
                            "decision": "excluded",
                            "reason": "blocked_by_doom_loop_guard",
                        }
                    ],
                },
                "metrics": {"recorded_candidate_attempt_count": 0},
            }
        ),
        encoding="utf-8",
    )
    audit_report_path.write_text(
        json.dumps({"decision": "passed", "failed_checks": []}),
        encoding="utf-8",
    )

    report = evaluate_research_directive_followthrough(
        directive_path=directive_path,
        chain_report_path=chain_report_path,
        output_path=tmp_path / "followthrough.json",
        force=False,
    )

    assert report["decision"] == "failed"
    assert report["checks"]["candidate_generator_followed"]["observed"] == [
        "fast_baseline_family"
    ]
    assert report["checks"]["directive_conflict"]["passed"] is False
    assert report["checks"]["directive_conflict"]["reason"] == (
        "blocked_by_doom_loop_guard"
    )


def test_hermes_research_directive_cli_and_tool_catalog(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog

    task1_benchmark = _write_task1_benchmark(tmp_path / "task1_benchmark.json")
    factory_benchmark = _write_factory_benchmark(tmp_path / "factory.json")
    directive_path = tmp_path / "directive.json"

    result = CliRunner().invoke(
        app,
        [
            "build-hermes-research-directive",
            "--task1-benchmark-path",
            str(task1_benchmark),
            "--candidate-factory-benchmark-path",
            str(factory_benchmark),
            "--output-path",
            str(directive_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "HERMES_RESEARCH_DIRECTIVE_BUILT"
    assert payload["artifacts"]["research_directive_json"] == str(directive_path)

    followthrough_path = tmp_path / "followthrough.json"
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0},
                "iteration_reports": [],
            }
        ),
        encoding="utf-8",
    )
    follow = CliRunner().invoke(
        app,
        [
            "evaluate-research-directive-followthrough",
            "--directive-path",
            str(directive_path),
            "--chain-report-path",
            str(chain_path),
            "--output-path",
            str(followthrough_path),
        ],
    )
    assert follow.exit_code == 0, follow.output
    assert json.loads(follow.output)["state_hint"] == (
        "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
    )

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "catalog.json",
        force=False,
    )
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["build-hermes-research-directive"]["risk_level"] == "read_only"
    assert tools["evaluate-research-directive-followthrough"]["risk_level"] == (
        "read_only"
    )


def _write_task1_benchmark(
    path: Path,
    *,
    decision: str = "passed",
    score_delta: float = 0.0995,
    best_score: float = 120.1904,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_TASK1_AGENT_CAPABILITY_BENCHMARKED",
                "decision": decision,
                "score_progress": {
                    "best_score": best_score,
                    "score_delta": score_delta,
                    "stagnation_count": 0,
                },
                "candidate_generation": {
                    "unique_candidate_families": [
                        "postprocess",
                        "hypothesis_lab_early_subsegment_patch_search",
                    ],
                    "unique_candidate_family_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_factory_benchmark(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED",
                "decision": "passed",
                "candidate_space": {
                    "mechanisms": [
                        "front25_dynamics",
                        "residual_corrector",
                        "spectral_physics",
                    ],
                    "mechanism_count": 3,
                },
                "heartbeat_handoff": {
                    "ready": True,
                    "planned_candidate_family_ids": [
                        "task1_residual_corrector_v0",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_judge(path: Path, *, score: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
                "total_score_proxy": score,
                "tasks": {
                    "1": {
                        "score": {"segment_score": 60.25},
                        "validation_score": {
                            "segments": {
                                "early": {"rel_mse": 0.0535, "score": 34.28},
                                "middle": {"score": 50.12},
                                "late": {"score": 78.30},
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_directive(
    path: Path,
    *,
    candidate_generator: str,
    learning_objective: str,
    candidate_options: dict[str, object],
    guard_conflict_policy: dict[str, object] | None = None,
) -> Path:
    directive = {
        "research_focus": "directed_test",
        "candidate_generator": candidate_generator,
        "candidate_generator_fallback": "auto",
        "learning_objective": learning_objective,
        "hypothesis_candidate_options": candidate_options,
        "experiment_design": {"minimum_trials": 1},
        "promotion_gate": {"min_score_delta": 0.05},
    }
    if guard_conflict_policy is not None:
        directive["guard_conflict_policy"] = guard_conflict_policy
    path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RESEARCH_DIRECTIVE_BUILT",
                "decision": "run_directed_experiment",
                "directive": directive,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_hdf5_reference(path: Path) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", shape=(1, 200, 256), dtype="float32")
    return path
