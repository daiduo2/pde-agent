from __future__ import annotations

import json
from pathlib import Path


def test_algorithm_designer_stability_batch_runs_multiple_memory_sets(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer_stability import (
        run_hypothesis_algorithm_designer_stability_batch,
    )
    import pde_agent.core.hypothesis_algorithm_designer_stability as stability_module

    memory_paths = []
    for index, text in enumerate(
        [
            "checkpoint static window blend failed and selected initial",
            "short window corrector failed due to phase drift and shock smoothing",
            "runtime plateau after residual family repeated",
        ]
    ):
        path = tmp_path / f"failure_{index}.json"
        path.write_text(
            json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": text}),
            encoding="utf-8",
        )
        memory_paths.append(path)

    def fake_sandbox(**kwargs: object) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        sandbox_path = output_dir / "sandbox.json"
        sandbox_path.write_text(
            json.dumps(
                {
                    "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
                    "blueprint_id": kwargs["blueprint"]["blueprint_id"],
                    "adapted_blueprint": {
                        "source_algorithm_blueprint_id": kwargs["blueprint"][
                            "source_algorithm_blueprint_id"
                        ]
                    },
                    "verifier": {
                        "rules_passed": True,
                        "score_delta_vs_baseline": -0.1,
                        "decision": "revise_hypothesis",
                    },
                    "artifacts": {"judge_report_path": str(output_dir / "judge.json")},
                }
            ),
            encoding="utf-8",
        )
        return json.loads(sandbox_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(stability_module, "run_blueprint_sandbox", fake_sandbox)

    report = run_hypothesis_algorithm_designer_stability_batch(
        memory_sets=[
            [memory_paths[0], memory_paths[1]],
            [memory_paths[1], memory_paths[2]],
        ],
        context_path=tmp_path / "context.hdf5",
        target_path=tmp_path / "target.hdf5",
        reference_path=None,
        reference_sample_limit=None,
        initial_submission_dir=tmp_path / "initial",
        output_dir=tmp_path / "batch",
        submission_id="team_test",
        max_sandbox_items_per_set=1,
        force=False,
    )

    assert report["state_hint"] == "HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_BATCH_RAN"
    assert report["summary"]["memory_set_count"] == 2
    assert report["summary"]["design_success_rate"] == 1.0
    assert report["summary"]["sandbox_success_rate"] == 1.0
    assert report["summary"]["evidence_writeback_success_rate"] == 1.0
    assert report["summary"]["min_mechanism_family_count"] >= 4
    assert all(item["hypothesis_count"] >= 5 for item in report["runs"])
    assert all(item["ready_count"] >= 1 for item in report["runs"])


def test_algorithm_designer_stability_batch_reports_execution_diversity(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hypothesis_algorithm_designer_stability import (
        run_hypothesis_algorithm_designer_stability_batch,
    )
    import pde_agent.core.hypothesis_algorithm_designer_stability as stability_module

    memory_paths = []
    for index, text in enumerate(
        [
            "checkpoint static window blend failed and selected initial",
            "short window corrector failed due to phase drift and shock smoothing",
            "runtime plateau after residual family repeated",
        ]
    ):
        path = tmp_path / f"failure_{index}.json"
        path.write_text(
            json.dumps({"state_hint": "FAILURE_MEMORY", "conclusion": text}),
            encoding="utf-8",
        )
        memory_paths.append(path)

    def fake_sandbox(**kwargs: object) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        blueprint = kwargs["blueprint"]
        family = blueprint["algorithm_designer_metadata"]["mechanism_family"]
        return {
            "state_hint": "HYPOTHESIS_LAB_SANDBOX_VERIFIED",
            "blueprint_id": blueprint["blueprint_id"],
            "adapted_blueprint": {
                "source_algorithm_blueprint_id": blueprint[
                    "source_algorithm_blueprint_id"
                ]
            },
            "verifier": {
                "rules_passed": True,
                "score_delta_vs_baseline": 0.2
                if family == "residual_correction"
                else -0.05,
                "metric_results": {
                    "context_preserved": {"value": True, "passed": True},
                    "score_delta_vs_baseline": {
                        "value": 0.2
                        if family == "residual_correction"
                        else -0.05,
                        "passed": family == "residual_correction",
                    },
                    "early_score_delta_vs_baseline": {
                        "value": 0.1
                        if family == "uncertainty_gating"
                        else 0.0,
                        "passed": True,
                    },
                    "middle_score_guardrail": {"value": 0.0, "passed": True},
                    "late_rmse_guardrail": {"value": 0.0, "passed": True},
                    "inference_time_budget": {
                        "value": 0.2,
                        "threshold": 120.0,
                        "passed": True,
                    },
                },
                "decision": "promote" if family == "residual_correction" else "revise",
            },
            "artifacts": {"judge_report_path": str(output_dir / "judge.json")},
        }

    monkeypatch.setattr(stability_module, "run_blueprint_sandbox", fake_sandbox)

    report = run_hypothesis_algorithm_designer_stability_batch(
        memory_sets=[
            [memory_paths[0], memory_paths[1]],
            [memory_paths[1], memory_paths[2]],
        ],
        context_path=tmp_path / "context.hdf5",
        target_path=tmp_path / "target.hdf5",
        reference_path=None,
        reference_sample_limit=None,
        initial_submission_dir=tmp_path / "initial",
        output_dir=tmp_path / "batch",
        submission_id="team_test",
        max_sandbox_items_per_set=3,
        force=False,
    )

    assert report["summary"]["unique_sandboxed_mechanism_family_count"] >= 3
    assert report["summary"]["unique_evidence_mechanism_family_count"] >= 3
    assert report["summary"]["unique_promoted_mechanism_family_count"] == 1
    assert report["summary"]["unique_refinement_mechanism_family_count"] >= 1
    assert report["summary"]["unique_rejected_mechanism_family_count"] >= 1
    assert report["summary"]["unique_evidence_decision_count"] >= 2
    assert "repeated_first_family_only" not in report["summary"]["warnings"]
    assert all(item["decision_rationales"] for item in report["runs"])
    assert all(
        {"mechanism_family", "decision", "reason_codes", "next_generation_constraints"}
        <= set(item["decision_rationales"][0])
        for item in report["runs"]
    )
    assert all(
        item["sandboxed_mechanism_families"]
        == ["uncertainty_gating", "residual_correction", "data_retrieval_nonlocal"]
        for item in report["runs"]
    )
    assert all(item["promoted_mechanism_families"] == ["residual_correction"] for item in report["runs"])
