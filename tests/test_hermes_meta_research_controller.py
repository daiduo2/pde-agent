from __future__ import annotations

import json
from pathlib import Path


def test_meta_research_controller_expands_space_after_followed_regression(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_meta_research_controller import (
        build_hermes_meta_research_directive,
    )

    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "best_score": 120.19,
                "batches": [
                    {
                        "candidate_generator": "postprocess",
                        "candidate_diagnostics": {
                            "attempted_candidate_count": 1,
                            "best_attempted_delta_vs_initial_objective": -20.63,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {"heartbeat_report_path": str(heartbeat_path)},
                "metrics": {
                    "candidate_space_exhausted": False,
                    "research_directive_loaded": True,
                },
                "candidate_plan": {
                    "candidate_generator_schedule": ["postprocess"],
                    "expansion_required": False,
                },
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0, "score_delta": 0.0},
                "iterations": [{"runtime_report_path": str(runtime_path)}],
            }
        ),
        encoding="utf-8",
    )
    followthrough_path = tmp_path / "followthrough.json"
    followthrough_path.write_text(
        json.dumps(
            {
                "state_hint": (
                    "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
                ),
                "decision": "passed",
                "checks": {
                    "candidate_generator_followed": {
                        "expected": "postprocess",
                        "observed": ["postprocess"],
                        "passed": True,
                    },
                    "directive_conflict": {"passed": True},
                    "audit_clean": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_meta_research_directive(
        followthrough_report_path=followthrough_path,
        chain_report_path=chain_path,
        task1_benchmark_path=_write_task1_benchmark(tmp_path / "task1.json"),
        candidate_factory_benchmark_path=_write_factory(tmp_path / "factory.json"),
        output_path=tmp_path / "meta_directive.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_META_RESEARCH_DIRECTIVE_BUILT"
    assert report["decision"] == "expand_candidate_space"
    assert report["research_assessment"]["last_candidate_generator"] == "postprocess"
    assert report["research_assessment"]["best_attempted_delta"] == -20.63
    assert report["directive"]["research_focus"] == "candidate_factory_expansion"
    assert report["directive"]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert report["next_command"] == "run-hermes-runtime-audit-chain"
    assert Path(report["directive_path"]).exists()


def test_meta_research_controller_resolves_guard_conflict_with_bounded_reprobe(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_meta_research_controller import (
        build_hermes_meta_research_directive,
    )

    followthrough_path = tmp_path / "followthrough.json"
    followthrough_path.write_text(
        json.dumps(
            {
                "state_hint": (
                    "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
                ),
                "decision": "failed",
                "checks": {
                    "candidate_generator_followed": {
                        "expected": "postprocess",
                        "observed": ["fast_baseline_family"],
                        "passed": False,
                    },
                    "directive_conflict": {
                        "candidate_generator": "postprocess",
                        "reason": "blocked_by_doom_loop_guard",
                        "passed": False,
                    },
                    "audit_clean": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0},
                "iterations": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_meta_research_directive(
        followthrough_report_path=followthrough_path,
        chain_report_path=chain_path,
        task1_benchmark_path=_write_task1_benchmark(tmp_path / "task1.json"),
        candidate_factory_benchmark_path=_write_factory(tmp_path / "factory.json"),
        output_path=tmp_path / "meta_directive.json",
        force=False,
    )

    assert report["decision"] == "resolve_directive_guard_conflict"
    assert report["directive"]["candidate_generator"] == "postprocess"
    assert report["directive"]["guard_conflict_policy"] == {
        "allowed_candidate_generators": ["postprocess"],
        "max_reprobe_batches": 1,
        "mode": "evidence_bounded_reprobe",
        "reason": "meta_controller_guard_conflict_resolution",
    }
    assert report["research_assessment"]["followthrough_decision"] == "failed"


def test_meta_research_controller_avoids_blocked_family_when_expanding_space(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_meta_research_controller import (
        build_hermes_meta_research_directive,
    )

    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_RAN",
                "batches": [
                    {
                        "candidate_generator": "postprocess",
                        "candidate_diagnostics": {
                            "attempted_candidate_count": 1,
                            "best_attempted_delta_vs_initial_objective": -1.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_CLOSED_LOOP_RAN",
                "artifacts": {"heartbeat_report_path": str(heartbeat_path)},
                "guard": {
                    "blocked_candidate_families": [
                        "hypothesis_lab_early_subsegment_patch_search"
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0},
                "iterations": [{"runtime_report_path": str(runtime_path)}],
            }
        ),
        encoding="utf-8",
    )
    followthrough_path = tmp_path / "followthrough.json"
    followthrough_path.write_text(
        json.dumps(
            {
                "state_hint": (
                    "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
                ),
                "decision": "passed",
                "checks": {
                    "candidate_generator_followed": {
                        "expected": "postprocess",
                        "observed": ["postprocess"],
                        "passed": True,
                    },
                    "directive_conflict": {"passed": True},
                    "audit_clean": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_hermes_meta_research_directive(
        followthrough_report_path=followthrough_path,
        chain_report_path=chain_path,
        task1_benchmark_path=_write_task1_benchmark(tmp_path / "task1.json"),
        candidate_factory_benchmark_path=_write_factory(tmp_path / "factory.json"),
        output_path=tmp_path / "meta_directive.json",
        force=False,
    )

    assert report["research_assessment"]["blocked_candidate_families"] == [
        "hypothesis_lab_early_subsegment_patch_search"
    ]
    assert report["directive"]["candidate_generator"] == (
        "hypothesis_lab_front25_dynamics_smoke"
    )


def test_meta_research_controller_cli_and_tool_catalog(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog

    followthrough_path = tmp_path / "followthrough.json"
    followthrough_path.write_text(
        json.dumps(
            {
                "state_hint": (
                    "HERMES_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
                ),
                "decision": "passed",
                "checks": {
                    "candidate_generator_followed": {
                        "expected": "postprocess",
                        "observed": ["postprocess"],
                        "passed": True,
                    },
                    "directive_conflict": {"passed": True},
                    "audit_clean": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"
    chain_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_RUNTIME_AUDIT_CHAIN_RAN",
                "decision": "passed",
                "metrics": {"failed_audit_count": 0},
                "iterations": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "meta_directive.json"

    result = CliRunner().invoke(
        app,
        [
            "build-hermes-meta-research-directive",
            "--followthrough-report-path",
            str(followthrough_path),
            "--chain-report-path",
            str(chain_path),
            "--task1-benchmark-path",
            str(_write_task1_benchmark(tmp_path / "task1.json")),
            "--candidate-factory-benchmark-path",
            str(_write_factory(tmp_path / "factory.json")),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "HERMES_META_RESEARCH_DIRECTIVE_BUILT"
    assert payload["artifacts"]["meta_research_directive_json"] == str(output_path)

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "catalog.json",
        force=False,
    )
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["build-hermes-meta-research-directive"]["risk_level"] == "read_only"


def test_runtime_closed_loop_accepts_meta_research_directive(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_runtime_closed_loop import (
        run_hermes_runtime_closed_loop,
    )

    meta_directive_path = tmp_path / "meta_directive.json"
    meta_directive_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_META_RESEARCH_DIRECTIVE_BUILT",
                "decision": "expand_candidate_space",
                "directive": {
                    "research_focus": "candidate_factory_expansion",
                    "candidate_generator": (
                        "hypothesis_lab_early_subsegment_patch_search"
                    ),
                    "candidate_generator_fallback": "auto",
                    "learning_objective": "score",
                    "hypothesis_candidate_options": {
                        "source_stability_guard": True
                    },
                },
            }
        ),
        encoding="utf-8",
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
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_runtime_closed_loop(
        output_dir=tmp_path / "runtime",
        trace_path=tmp_path / "trace.jsonl",
        initial_submission_dir=initial_dir,
        candidate_generator="postprocess",
        max_batches=1,
        research_directive_path=meta_directive_path,
        heartbeat_runner=fake_heartbeat_runner,
        force=False,
    )

    assert heartbeat_calls[0]["candidate_generator"] == (
        "hypothesis_lab_early_subsegment_patch_search"
    )
    assert heartbeat_calls[0]["hypothesis_candidate_options"] == {
        "source_stability_guard": True
    }
    assert report["research_directive"]["state_hint"] == (
        "HERMES_META_RESEARCH_DIRECTIVE_BUILT"
    )


def _write_task1_benchmark(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_TASK1_AGENT_CAPABILITY_BENCHMARKED",
                "decision": "passed",
                "score_progress": {
                    "best_score": 120.19,
                    "score_delta": 0.0,
                    "stagnation_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_factory(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "state_hint": "TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED",
                "decision": "passed",
                "heartbeat_handoff": {"ready": True},
                "candidate_space": {
                    "mechanism_count": 3,
                    "mechanisms": [
                        "front25_dynamics",
                        "residual_corrector",
                        "spectral_physics",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path
