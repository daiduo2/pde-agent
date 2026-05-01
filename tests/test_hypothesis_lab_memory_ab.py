from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fake_lab_report(
    *,
    arm: str,
    output_dir: Path,
    score_delta: float,
    segment_score: float,
    candidate_id: str,
    anchor: str,
    candidate_budget: int,
) -> dict[str, Any]:
    sandbox_path = output_dir / "sandbox" / "hypothesis_lab_sandbox_report.json"
    sandbox_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_LAB_RAN",
        "cards": {"count": 1, "path": str(output_dir / "knowledge_cards.json")},
        "hypotheses": {"count": 1, "path": str(output_dir / "hypotheses.json")},
        "blueprints": {"count": 1, "path": str(output_dir / "blueprints.json")},
        "sandbox": {
            "candidate_family": "task1_early_refinement",
            "best_candidate": {
                "candidate_id": candidate_id,
                "params": {
                    "k": 12,
                    "mode": "affine",
                    "early_blend_steps": 20,
                    "persistence_weight_start": 0.5,
                    "spectral_strength": 0.5,
                    "early_anchor": anchor,
                },
            },
            "top_candidates": [
                {
                    "candidate_id": candidate_id,
                    "params": {"early_anchor": anchor},
                }
            ],
            "verifier": {
                "decision": "promote_to_strategy_proposal",
                "score_delta_vs_baseline": score_delta,
                "best_segment_score": segment_score,
                "pareto_promotion_gate": {
                    "pareto_status": "dominates_baseline",
                    "objective_axes": {
                        "candidate_budget": {
                            "value": candidate_budget,
                            "passed": True,
                            "improved": False,
                        }
                    },
                },
            },
            "artifacts": {
                "sandbox_report_path": str(sandbox_path),
                "best_submission_dir": str(output_dir / "submission"),
            },
        },
        "artifacts": {
            "hypothesis_lab_report_path": str(output_dir / "hypothesis_lab_report.json"),
            "sandbox_report_path": str(sandbox_path),
            "best_submission_dir": str(output_dir / "submission"),
            "verified_blueprint_path": str(output_dir / "verified_blueprints.json"),
        },
        "next_command": "record-strategy-proposal",
        "arm": arm,
    }
    sandbox_path.write_text(json.dumps(report["sandbox"]), encoding="utf-8")
    (output_dir / "hypothesis_lab_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return report


def test_compare_hypothesis_lab_memory_runs_two_arms_and_scores_delta(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_memory_ab import (
        compare_hypothesis_lab_memory,
    )

    calls: list[dict[str, Any]] = []
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text("{}", encoding="utf-8")

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        if kwargs.get("failure_memory_paths"):
            return _fake_lab_report(
                arm="memory",
                output_dir=output_dir,
                score_delta=1.25,
                segment_score=47.8,
                candidate_id="memory_best",
                anchor="phase_aligned_retrieval",
                candidate_budget=28,
            )
        return _fake_lab_report(
            arm="no_memory",
            output_dir=output_dir,
            score_delta=1.0,
            segment_score=47.4,
            candidate_id="baseline_best",
            anchor="persistence",
            candidate_budget=34,
        )

    report = compare_hypothesis_lab_memory(
        manual_report_paths=[],
        adaptive_report_paths=[],
        failure_memory_paths=[memory_path],
        context_path=tmp_path / "context.hdf5",
        target_path=tmp_path / "target.hdf5",
        output_dir=tmp_path / "ab",
        submission_id="team_hermes",
        force=False,
        runner=fake_runner,
    )

    assert [Path(call["output_dir"]).name for call in calls] == [
        "no_memory",
        "memory",
    ]
    assert calls[0]["failure_memory_paths"] == []
    assert calls[1]["failure_memory_paths"] == [memory_path]
    assert report["state_hint"] == "HYPOTHESIS_LAB_MEMORY_AB_COMPARED"
    assert report["decision"] == "memory_helped"
    assert report["score_delta_diff"] == 0.25
    assert report["candidate_budget_diff"] == -6
    assert report["best_candidate_changed"] is True
    assert report["arms"]["memory"]["anchor_histogram"] == {
        "phase_aligned_retrieval": 1
    }
    assert Path(report["artifacts"]["comparison_report_path"]).exists()


def test_compare_hypothesis_lab_memory_prefers_absolute_best_score_for_decision(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab_memory_ab import (
        compare_hypothesis_lab_memory,
    )

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        if kwargs.get("failure_memory_paths"):
            return _fake_lab_report(
                arm="memory",
                output_dir=output_dir,
                score_delta=1.5,
                segment_score=47.8,
                candidate_id="same_best",
                anchor="persistence",
                candidate_budget=20,
            )
        return _fake_lab_report(
            arm="no_memory",
            output_dir=output_dir,
            score_delta=1.0,
            segment_score=47.8,
            candidate_id="same_best",
            anchor="persistence",
            candidate_budget=20,
        )

    report = compare_hypothesis_lab_memory(
        manual_report_paths=[],
        adaptive_report_paths=[],
        failure_memory_paths=[tmp_path / "failure_memory.json"],
        context_path=tmp_path / "context.hdf5",
        target_path=tmp_path / "target.hdf5",
        output_dir=tmp_path / "ab",
        submission_id="team_hermes",
        force=False,
        runner=fake_runner,
    )

    assert report["score_delta_diff"] == 0.5
    assert report["best_segment_score_diff"] == 0.0
    assert report["decision"] == "memory_neutral"


def test_compare_hypothesis_lab_memory_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from typer.testing import CliRunner

    import pde_agent.commands.compare_hypothesis_lab_memory as command_module
    from pde_agent.cli import app

    def fake_compare(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "memory_ab_report.json"
        report = {
            "schema_version": 1,
            "state_hint": "HYPOTHESIS_LAB_MEMORY_AB_COMPARED",
            "decision": "memory_changed_search",
            "score_delta_diff": 0.0,
            "best_segment_score_diff": 0.0,
            "candidate_budget_diff": -4,
            "best_candidate_changed": True,
            "arms": {},
            "artifacts": {"comparison_report_path": str(report_path)},
            "next_command": "run-hypothesis-lab",
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(command_module, "compare_hypothesis_lab_memory", fake_compare)
    result = CliRunner().invoke(
        app,
        [
            "compare-hypothesis-lab-memory",
            "--failure-memory-path",
            str(tmp_path / "failure_memory.json"),
            "--context-path",
            str(tmp_path / "context.hdf5"),
            "--target-path",
            str(tmp_path / "target.hdf5"),
            "--output-dir",
            str(tmp_path / "ab"),
            "--submission-id",
            "team_hermes",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HYPOTHESIS_LAB_MEMORY_AB_COMPARED"
    assert payload["metrics"]["decision"] == "memory_changed_search"
    assert payload["metrics"]["best_candidate_changed"] is True
