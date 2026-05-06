from __future__ import annotations

import json
from pathlib import Path


def test_exploration_scheduler_refines_positive_spectral_smooth_signal(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_exploration_scheduler import (
        build_hermes_exploration_schedule,
    )

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "postprocess_spectral_smooth_w0.7325",
            "score": 120.34767254492024,
            "strategy": "spectral_smooth",
            "blend_weight": 0.7325,
            "source": "postprocess_generator",
        },
        family_outcomes={
            "postprocess_generator": {
                "status": "positive_signal",
                "best_score": 120.34767254492024,
                "best_delta_vs_initial": 0.3216681577595182,
            },
            "linear_trend_spectral": {
                "status": "blocked",
                "best_score": 86.2,
                "best_delta_vs_initial": -33.826004387160725,
            },
        },
        positive_signals=[
            {
                "candidate_id": "postprocess_spectral_smooth_w0.7325",
                "candidate_family": "postprocess_generator",
                "strategy": "spectral_smooth",
                "blend_weight": 0.7325,
                "objective_delta_vs_initial": 0.3216681577595182,
            }
        ],
    )

    report = build_hermes_exploration_schedule(
        memory_json_path=memory_path,
        output_path=tmp_path / "schedule.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_EXPLORATION_SCHEDULE_BUILT"
    assert report["decision"] == "refine_positive_signal"
    assert report["schedule"]["candidate_generator"] == "postprocess"
    assert report["schedule"]["postprocess_strategy"] == "spectral_smooth"
    assert report["schedule"]["postprocess_adaptive_step"] == 0.001
    assert report["schedule"]["blocked_candidate_families"] == [
        "linear_trend_spectral"
    ]
    assert report["schedule"]["promotion_gate"]["min_score_delta"] == 0.0001
    assert Path(report["schedule_path"]).exists()


def test_exploration_scheduler_switches_to_near_miss_after_no_positive_signal(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_exploration_scheduler import (
        build_hermes_exploration_schedule,
    )

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "initial",
            "score": 120.02600438716073,
            "source": "initial_submission",
        },
        family_outcomes={
            "physics_residual_nudge": {
                "status": "near_miss",
                "best_score": 120.01945342249716,
                "best_delta_vs_initial": -0.00655096466357,
            },
            "sem_window_gate_blend_multi_search": {
                "status": "blocked",
                "best_score": 120.015,
                "best_delta_vs_initial": -0.011,
            },
        },
        positive_signals=[],
    )

    report = build_hermes_exploration_schedule(
        memory_json_path=memory_path,
        output_path=tmp_path / "schedule.json",
        force=False,
    )

    assert report["decision"] == "refine_near_miss_family"
    assert report["schedule"]["candidate_generator"] == "physics_residual_nudge"
    assert report["schedule"]["search_budget"] == "micro"
    assert "sem_window_gate_blend_multi_search" in report["schedule"][
        "blocked_candidate_families"
    ]


def test_exploration_scheduler_maps_postprocess_family_to_supported_generator(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_exploration_scheduler import (
        build_hermes_exploration_schedule,
    )

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "initial",
            "score": 120.34767254492024,
            "source": "initial_submission",
        },
        family_outcomes={
            "postprocess_generator": {
                "status": "near_miss",
                "best_score": 120.34718209649671,
                "best_delta_vs_initial": -0.00049044842353,
            },
        },
        positive_signals=[],
    )

    report = build_hermes_exploration_schedule(
        memory_json_path=memory_path,
        output_path=tmp_path / "schedule.json",
        force=False,
    )

    assert report["decision"] == "refine_near_miss_family"
    assert report["schedule"]["candidate_generator"] == "postprocess"
    assert report["schedule"]["source_candidate_family"] == "postprocess_generator"


def test_exploration_scheduler_preserves_near_miss_postprocess_strategy(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_exploration_scheduler import (
        build_hermes_exploration_schedule,
    )

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "initial",
            "score": 120.34767254492024,
            "source": "initial_submission",
        },
        family_outcomes={
            "postprocess_generator": {
                "status": "near_miss",
                "best_candidate_id": "postprocess_spectral_smooth_w0.5",
                "best_score": 120.34743509649671,
                "best_delta_vs_initial": -0.00023744842353,
                "best_strategy": "spectral_smooth",
                "best_blend_weight": 0.5,
            },
        },
        positive_signals=[],
    )

    report = build_hermes_exploration_schedule(
        memory_json_path=memory_path,
        output_path=tmp_path / "schedule.json",
        force=False,
    )

    assert report["decision"] == "refine_near_miss_family"
    assert report["schedule"]["candidate_generator"] == "postprocess"
    assert report["schedule"]["source_candidate_family"] == "postprocess_generator"
    assert report["schedule"]["postprocess_strategy"] == "spectral_smooth"
    assert report["schedule"]["seed_candidate_id"] == (
        "postprocess_spectral_smooth_w0.5"
    )
    assert report["schedule"]["seed_blend_weight"] == 0.5


def test_exploration_scheduler_refines_positive_checkpoint_gate_family(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_exploration_scheduler import (
        build_hermes_exploration_schedule,
    )

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "checkpoint_window_gate_checkpoint_jump35_gt_0.000233021_10_20",
            "score": 120.51071265793594,
            "source": "checkpoint_window_gate",
        },
        family_outcomes={
            "checkpoint_window_gate": {
                "status": "positive_signal",
                "best_candidate_id": (
                    "checkpoint_window_gate_checkpoint_jump35_gt_0.000233021_10_20"
                ),
                "best_score": 120.51071265793594,
                "best_delta_vs_initial": 0.16304011301569687,
            },
        },
        positive_signals=[
            {
                "candidate_id": (
                    "checkpoint_window_gate_checkpoint_jump35_gt_0.000233021_10_20"
                ),
                "candidate_family": "checkpoint_window_gate",
                "objective_delta_vs_initial": 0.16304011301569687,
                "score": 120.51071265793594,
            }
        ],
    )

    report = build_hermes_exploration_schedule(
        memory_json_path=memory_path,
        output_path=tmp_path / "schedule.json",
        force=False,
    )

    assert report["decision"] == "refine_positive_signal"
    assert report["schedule"]["candidate_generator"] == "checkpoint_window_gate"
    assert report["schedule"]["source_candidate_family"] == "checkpoint_window_gate"
    assert report["schedule"]["search_budget"] == "local_refinement"
    assert report["schedule"]["seed_candidate_id"] == (
        "checkpoint_window_gate_checkpoint_jump35_gt_0.000233021_10_20"
    )


def test_exploration_scheduler_cli_and_tool_catalog(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app
    from pde_agent.core.scientific_tool_router import build_scientific_tool_catalog

    memory_path = _write_memory(
        tmp_path / "memory.json",
        best_candidate={
            "candidate_id": "initial",
            "score": 120.02600438716073,
            "source": "initial_submission",
        },
        family_outcomes={},
        positive_signals=[],
    )
    output_path = tmp_path / "schedule.json"

    result = CliRunner().invoke(
        app,
        [
            "build-hermes-exploration-schedule",
            "--memory-json-path",
            str(memory_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state_hint"] == "HERMES_EXPLORATION_SCHEDULE_BUILT"
    assert payload["artifacts"]["hermes_exploration_schedule_json"] == str(
        output_path
    )

    catalog = build_scientific_tool_catalog(
        output_path=tmp_path / "catalog.json",
        force=False,
    )
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert tools["build-hermes-exploration-schedule"]["risk_level"] == (
        "artifact_write"
    )


def _write_memory(
    path: Path,
    *,
    best_candidate: dict[str, object],
    family_outcomes: dict[str, object],
    positive_signals: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
                "best_candidate": best_candidate,
                "family_outcomes": family_outcomes,
                "avoid_candidate_families": [
                    {
                        "candidate_family": family,
                        "reason": "blocked_by_prior_evidence",
                    }
                    for family, outcome in family_outcomes.items()
                    if isinstance(outcome, dict)
                    and outcome.get("status") == "blocked"
                ],
                "positive_signals": positive_signals,
                "search_guidance": [],
            }
        ),
        encoding="utf-8",
    )
    return path
