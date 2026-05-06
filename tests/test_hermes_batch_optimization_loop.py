import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_hdf5_tensor(path: Path, data: np.ndarray) -> Path:
    h5py = __import__("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)
    return path


def test_hermes_batch_optimization_loop_tracks_best_and_stops_on_patience(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    scores = {
        "initial": 80.0,
        "candidate_1": 91.0,
        "candidate_2": 91.2,
        "candidate_3": 96.0,
    }
    candidate_calls: list[int] = []

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        candidate_calls.append(round_index)
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
            "strategy_id": "residual_calibration_v0",
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).name
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": scores[name],
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": scores[name]}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=95.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert candidate_calls == [1, 2]
    assert report["state_hint"] == "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
    assert report["round_count"] == 3
    assert report["best_candidate"]["candidate_id"] == "candidate_2"
    assert report["best_candidate"]["score"] == 91.2
    assert report["stop_reason"] == "patience_exhausted"
    assert report["decision"] == "iterate_more"
    assert report["next_command"] == "run-hermes-batch-optimization-loop"
    assert Path(report["loop_report_path"]).exists()


def test_hermes_batch_optimization_loop_recommends_submit_when_threshold_met(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        submission_dir = tmp_path / "candidate"
        submission_dir.mkdir()
        return {
            "candidate_id": "candidate",
            "submission_dir": str(submission_dir),
            "strategy_id": "rollout_stabilization_v0",
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 88.0 if Path(kwargs["submission_dir"]).name == "initial" else 97.5
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=2,
        min_score_improvement=0.5,
        submit_score_threshold=95.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert report["best_candidate"]["score"] == 97.5
    assert report["stop_reason"] == "submit_threshold_met"
    assert report["decision"] == "submit_best_candidate"
    assert report["next_command"] == "submit-best-candidate"


def test_hermes_batch_optimization_loop_can_generate_postprocess_candidates(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    generated: list[tuple[str, float]] = []

    def fake_generator(**kwargs: Any) -> dict[str, Any]:
        strategy = str(kwargs["strategy"])
        weight = float(kwargs["blend_weight"])
        generated.append((strategy, weight))
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "state_hint": "HERMES_POSTPROCESS_CANDIDATE_READY",
            "candidate_id": f"{strategy}_{weight}",
            "strategy": strategy,
            "blend_weight": weight,
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).parent.name
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 92.0
        if "round_002" in name:
            score = 95.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(loop_module, "generate_hermes_postprocess_candidate", fake_generator)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=1.0,
        submit_score_threshold=99.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="postprocess",
        postprocess_strategies=["context_anchor_blend", "spectral_smooth"],
        postprocess_blend_weights=[0.25, 0.5],
        judge=fake_judge,
        force=False,
    )

    assert generated == [("context_anchor_blend", 0.25), ("spectral_smooth", 0.5)]
    assert report["best_candidate"]["score"] == 95.0
    assert report["best_candidate"]["metadata"]["source"] == "postprocess_generator"
    assert report["round_count"] == 3


def test_hermes_batch_optimization_loop_postprocess_grid_uses_cartesian_product(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    generated: list[tuple[str, float]] = []

    def fake_generator(**kwargs: Any) -> dict[str, Any]:
        strategy = str(kwargs["strategy"])
        weight = float(kwargs["blend_weight"])
        generated.append((strategy, weight))
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "candidate_id": f"{strategy}_{weight}",
            "strategy": strategy,
            "blend_weight": weight,
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 81.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(loop_module, "generate_hermes_postprocess_candidate", fake_generator)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=4,
        patience=10,
        min_score_improvement=5.0,
        submit_score_threshold=120.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="postprocess",
        postprocess_search_mode="grid",
        postprocess_strategies=["context_anchor_blend", "spectral_smooth"],
        postprocess_blend_weights=[0.25, 0.5],
        judge=fake_judge,
        force=False,
    )

    assert generated == [
        ("context_anchor_blend", 0.25),
        ("context_anchor_blend", 0.5),
        ("spectral_smooth", 0.25),
        ("spectral_smooth", 0.5),
    ]


def test_hermes_batch_optimization_loop_postprocess_adaptive_refines_best_weight(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    generated: list[tuple[str, float]] = []

    def fake_generator(**kwargs: Any) -> dict[str, Any]:
        strategy = str(kwargs["strategy"])
        weight = float(kwargs["blend_weight"])
        generated.append((strategy, weight))
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "candidate_id": f"{strategy}_{weight}",
            "strategy": strategy,
            "blend_weight": weight,
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).parent.name
        scores = {
            "round_001": 90.0,
            "round_002": 94.0,
            "round_003": 96.0,
        }
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else scores[name]
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(loop_module, "generate_hermes_postprocess_candidate", fake_generator)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=10,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="postprocess",
        postprocess_search_mode="adaptive",
        postprocess_strategies=["context_anchor_blend"],
        postprocess_blend_weights=[0.25, 0.75],
        postprocess_adaptive_step=0.25,
        judge=fake_judge,
        force=False,
    )

    assert generated == [
        ("context_anchor_blend", 0.25),
        ("context_anchor_blend", 0.75),
        ("context_anchor_blend", 0.5),
    ]
    assert report["best_candidate"]["score"] == 96.0


def test_hermes_batch_optimization_loop_stops_before_deadline(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    candidate_calls: list[int] = []
    times = iter([0.0, 4.8, 6.1])

    def fake_time() -> float:
        return next(times)

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        candidate_calls.append(round_index)
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 81.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=10,
        patience=10,
        min_score_improvement=5.0,
        submit_score_threshold=120.0,
        deadline_seconds=5.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        clock=fake_time,
        force=False,
    )

    assert candidate_calls == [1]
    assert report["stop_reason"] == "deadline_exceeded"
    assert report["budget"]["deadline_seconds"] == 5.0


def test_hermes_batch_optimization_loop_checkpoints_iteration_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    checkpoint_calls: list[Path] = []

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 80.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    def fake_memory_builder(**kwargs: Any) -> dict[str, Any]:
        checkpoint_calls.append(Path(kwargs["batch_loop_report_path"]))
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
            "metrics": {"trajectory_count": 0, "avoid_count": 0, "updated_log_count": 0},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=10,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        checkpoint_every=2,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        memory_builder=fake_memory_builder,
        force=False,
    )

    assert len(checkpoint_calls) == 2
    assert report["memory_checkpoints"][0]["round_index"] == 2
    assert report["memory_checkpoints"][-1]["round_index"] == 3


def test_hermes_batch_optimization_loop_skips_memory_avoid_entries(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "avoid_repeating": [
                    {
                        "strategy": "spectral_smooth",
                        "blend_weight": 0.25,
                        "reason": "below_min_effective_improvement",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generated: list[tuple[str, float]] = []

    def fake_generator(**kwargs: Any) -> dict[str, Any]:
        strategy = str(kwargs["strategy"])
        weight = float(kwargs["blend_weight"])
        generated.append((strategy, weight))
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "candidate_id": f"{strategy}_{weight}",
            "strategy": strategy,
            "blend_weight": weight,
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 81.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(loop_module, "generate_hermes_postprocess_candidate", fake_generator)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=10,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        memory_json_path=memory_path,
        avoid_repeating=True,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="postprocess",
        postprocess_search_mode="grid",
        postprocess_strategies=["spectral_smooth"],
        postprocess_blend_weights=[0.25, 0.5],
        judge=fake_judge,
        force=False,
    )

    assert generated == [("spectral_smooth", 0.5)]
    assert report["skipped_candidates"][0]["strategy"] == "spectral_smooth"
    assert report["skipped_candidates"][0]["blend_weight"] == 0.25


def test_hermes_batch_optimization_loop_passes_memory_search_guidance_to_sem_runner(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "search_guidance": [
                    {
                        "guidance_type": "refine_gating_after_near_miss",
                        "reason": "near_miss_with_empty_gate_selection",
                        "source": "sem_window_gate_blend_multi_search",
                        "feature_name": "sem_base_disagreement_20_57",
                        "threshold": -1.0,
                        "selected_sample_count": 0,
                        "window_start": 10,
                        "window_stop": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_sem_runner(**kwargs: Any) -> object:
        captured["search_guidance"] = kwargs["search_guidance"]

        def run_candidate(**candidate_kwargs: Any) -> dict[str, Any] | None:
            if int(candidate_kwargs["round_index"]) != 1:
                return None
            submission_dir = Path(candidate_kwargs["output_dir"]) / "candidate"
            submission_dir.mkdir(parents=True)
            return {
                "candidate_id": "sem_guided",
                "submission_dir": str(submission_dir),
                "source": "sem_window_gate_blend_multi_search",
            }

        return run_candidate

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 81.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(
        loop_module,
        "_sem_window_gate_blend_multi_search_runner",
        fake_sem_runner,
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=10,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        memory_json_path=memory_path,
        avoid_repeating=True,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="sem_window_gate_blend_multi_early_smoke",
        judge=fake_judge,
        force=False,
    )

    assert captured["search_guidance"] == [
        {
            "guidance_type": "refine_gating_after_near_miss",
            "reason": "near_miss_with_empty_gate_selection",
            "source": "sem_window_gate_blend_multi_search",
            "feature_name": "sem_base_disagreement_20_57",
            "threshold": -1.0,
            "selected_sample_count": 0,
            "window_start": 10,
            "window_stop": 20,
        }
    ]


def test_hermes_batch_optimization_loop_reports_loaded_repair_policy_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    repair_action = {
        "audit_error": "repair_convergence_consistency",
        "action": "switch_repair_strategy_or_reload_missing_evidence",
        "reason": "repair_nonconverged",
    }
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_HEARTBEAT_INPUT_MEMORY_SEEDED",
                "repair_policy_recommended_actions": [repair_action],
                "repair_policy_recommended_action_count": 1,
                "input_sources": {
                    "repair_policy_summary_path": str(tmp_path / "repair_policy.json")
                },
                "search_guidance": [{"guidance_type": "existing_guidance"}],
            }
        ),
        encoding="utf-8",
    )

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any] | None:
        submission_dir = tmp_path / "candidate"
        submission_dir.mkdir(exist_ok=True)
        return {
            "candidate_id": "candidate_1",
            "submission_dir": str(submission_dir),
            "metadata": {"source": "test_candidate"},
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 81.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "decision": "submit_candidate",
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        memory_json_path=memory_path,
        avoid_repeating=True,
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert report["parameters"]["memory_json_path"] == str(memory_path)
    assert report["memory_constraints"] == {
        "memory_json_path": str(memory_path),
        "avoid_repeating": True,
        "avoid_count": 0,
        "search_guidance_count": 1,
        "repair_policy_recommended_action_count": 1,
        "repair_policy_recommended_actions": [repair_action],
        "input_sources": {
            "repair_policy_summary_path": str(tmp_path / "repair_policy.json")
        },
    }


def test_hermes_batch_optimization_loop_auto_objective_learns_runtime_when_timeout(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    timings = {
        "initial": 157.0,
        "candidate_1": 145.0,
        "candidate_2": 130.0,
    }

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).name
        inference_time = timings[name]
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_LOCAL_JUDGE_SCORED",
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": 0.0,
            "decision": "iterate_before_submit",
            "tasks": {
                "1": {
                    "timing": {"train_time": 0.0, "inference_time": inference_time},
                    "score": {
                        "task_total_score": 0.0,
                        "timeout_zeroed": True,
                        "segment_score": 10.0,
                        "prediction_accuracy_score": 7.5,
                    },
                }
            },
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="auto",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert report["parameters"]["learning_objective"] == "auto"
    assert report["best_candidate"]["candidate_id"] == "candidate_2"
    assert report["best_candidate"]["objective"]["selected_objective"] == "runtime"
    assert report["best_candidate"]["objective"]["inference_time"] == 130.0
    assert report["trajectory"][1]["improved"] is True
    assert report["trajectory"][2]["score"] == 0.0


def test_hermes_batch_optimization_loop_early_accuracy_objective_uses_early_segment(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )

    candidate_calls: list[int] = []

    def fake_candidate_runner(**kwargs: Any) -> dict[str, Any]:
        round_index = int(kwargs["round_index"])
        candidate_calls.append(round_index)
        submission_dir = tmp_path / f"candidate_{round_index}"
        submission_dir.mkdir()
        return {
            "candidate_id": f"candidate_{round_index}",
            "submission_dir": str(submission_dir),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).name
        if name == "initial":
            total_score = 110.0
            early_score = 1.0
            early_rel_mse = 0.23
        else:
            total_score = 92.0
            early_score = 35.0
            early_rel_mse = 0.052
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": total_score,
            "tasks": {
                "1": {
                    "timing": {"train_time": 0.0, "inference_time": 1.0},
                    "score": {
                        "task_total_score": total_score,
                        "prediction_accuracy_score": total_score - 75.0,
                        "segment_score": total_score - 70.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {
                                "score": early_score,
                                "rel_mse": early_rel_mse,
                            }
                        }
                    },
                }
            },
        }

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_runner=fake_candidate_runner,
        judge=fake_judge,
        force=False,
    )

    assert candidate_calls == [1]
    assert report["best_candidate"]["candidate_id"] == "candidate_1"
    assert report["best_candidate"]["objective"]["selected_objective"] == (
        "early_accuracy"
    )
    assert report["best_candidate"]["objective"]["early_segment_score"] == 35.0
    assert report["best_candidate"]["objective"]["early_rel_mse"] == 0.052
    assert report["trajectory"][1]["score"] == 92.0


def test_hermes_batch_optimization_loop_runtime_config_search_generates_unet_configs(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    predicted_configs: list[tuple[int | None, int | None]] = []
    materialize_profiles: list[str | None] = []

    def fake_predictor(**kwargs: Any) -> dict[str, Any]:
        predicted_configs.append(
            (kwargs["inference_batch_size"], kwargs["torch_num_threads"])
        )
        report_path = Path(kwargs["output_dir"]) / "candidate_execution_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "artifacts": {"candidate_execution_report_path": str(report_path)},
            "inference": {"elapsed_seconds": 100.0},
        }

    def fake_materializer(**kwargs: Any) -> dict[str, Any]:
        materialize_profiles.append(kwargs.get("timing_profile"))
        submission_dir = Path(kwargs["output_dir"]) / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "artifacts": {"submission_dir": str(submission_dir)},
            "submission": {"metrics": {"inference_time": 100.0}},
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 0.0 if Path(kwargs["submission_dir"]).name == "initial" else 80.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"train_time": 0.0, "inference_time": 100.0},
                    "score": {"task_total_score": score},
                }
            },
        }

    monkeypatch.setattr(loop_module, "predict_unet_competition", fake_predictor)
    monkeypatch.setattr(loop_module, "materialize_competition_candidate", fake_materializer)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="runtime",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="runtime_config_search",
        unet_checkpoint_path=tmp_path / "checkpoint.pt",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert predicted_configs == [(8, 1), (8, 2)]
    assert materialize_profiles == ["mac_mini_m4_local_dev", "mac_mini_m4_local_dev"]
    assert report["trajectory"][1]["metadata"]["source"] == "runtime_config_search"
    assert report["trajectory"][1]["metadata"]["inference_batch_size"] == 8
    assert report["trajectory"][2]["metadata"]["torch_num_threads"] == 2


def test_hermes_batch_optimization_loop_fast_baseline_family_generates_persistence(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    generated_methods: list[str] = []

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        generated_methods.append(str(kwargs["method"]))
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {"artifacts": {"submission_dir": str(submission_dir)}}

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 0.0 if Path(kwargs["submission_dir"]).name == "initial" else 70.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"train_time": 0.0, "inference_time": 0.01},
                    "score": {"task_total_score": score},
                }
            },
        }

    monkeypatch.setattr(loop_module, "build_competition_submission", fake_builder)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="auto",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="fast_baseline_family",
        judge=fake_judge,
        force=False,
    )

    assert generated_methods == ["persistence"]
    assert report["best_candidate"]["metadata"]["source"] == "fast_baseline_family"
    assert report["best_candidate"]["metadata"]["baseline_method"] == "persistence"


def test_hermes_batch_optimization_loop_hybrid_gate_tries_runtime_then_fast_baseline(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    events: list[str] = []

    def fake_predictor(**kwargs: Any) -> dict[str, Any]:
        events.append("runtime")
        report_path = Path(kwargs["output_dir"]) / "candidate_execution_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {"artifacts": {"candidate_execution_report_path": str(report_path)}}

    def fake_materializer(**kwargs: Any) -> dict[str, Any]:
        submission_dir = Path(kwargs["output_dir"]) / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {"artifacts": {"submission_dir": str(submission_dir)}}

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        events.append("fast")
        submission_dir = Path(kwargs["output_dir"]) / "submission"
        submission_dir.mkdir(parents=True)
        return {"artifacts": {"submission_dir": str(submission_dir)}}

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        parent = Path(kwargs["submission_dir"]).parent.name
        score = 0.0
        inference_time = 157.0
        if parent == "round_002":
            score = 70.0
            inference_time = 0.01
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"train_time": 0.0, "inference_time": inference_time},
                    "score": {
                        "task_total_score": score,
                        "timeout_zeroed": inference_time > 120.0,
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "predict_unet_competition", fake_predictor)
    monkeypatch.setattr(loop_module, "materialize_competition_candidate", fake_materializer)
    monkeypatch.setattr(loop_module, "build_competition_submission", fake_builder)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        learning_objective="auto",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="hybrid_gate",
        unet_checkpoint_path=tmp_path / "checkpoint.pt",
        judge=fake_judge,
        force=False,
    )

    assert events == ["runtime", "fast"]
    assert report["best_candidate"]["metadata"]["source"] == "fast_baseline_family"


def test_hermes_batch_optimization_loop_hypothesis_lab_generates_candidate(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    calls: list[tuple[str, Path]] = []

    def fake_library_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        calls.append(("library", output_path))
        return {"state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT", "library_path": str(output_path)}

    def fake_candidate_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.write_text("{}", encoding="utf-8")
        calls.append(("candidates", output_path))
        return {
            "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
            "algorithm_candidates_path": str(output_path),
        }

    def fake_hypothesis_lab(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "sandbox" / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        calls.append(("hypothesis_lab", output_dir))
        assert kwargs["algorithm_candidate_paths"] == [
            output_dir / "local_asset_candidates.json"
        ]
        assert kwargs["context_path"] == Path("task1_test.hdf5")
        assert kwargs["target_path"] == Path("task1_val.hdf5")
        assert kwargs["reference_path"] == Path("task1_train_reference.hdf5")
        assert kwargs["timing_profile"] == "mac_mini_m4_local_dev"
        return {
            "state_hint": "HYPOTHESIS_LAB_RAN",
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "hypothesis_lab_report_path": str(output_dir / "hypothesis_lab_report.json"),
            },
            "sandbox": {
                "best_candidate": {
                    "candidate_id": "task1_retrieval_prior_k20_affine",
                    "params": {"k": 20, "mode": "affine"},
                },
                "verifier": {
                    "score_delta_vs_baseline": 23.0,
                    "decision": "promote_to_strategy_proposal",
                },
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 86.0
        if "submission_package" in str(kwargs["submission_dir"]):
            score = 109.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {"1": {"score": {"task_total_score": score}}},
        }

    monkeypatch.setattr(loop_module, "build_local_strategy_library", fake_library_builder)
    monkeypatch.setattr(
        loop_module,
        "build_local_asset_algorithm_candidates",
        fake_candidate_builder,
    )
    monkeypatch.setattr(loop_module, "run_hypothesis_lab", fake_hypothesis_lab)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=1.0,
        submit_score_threshold=120.0,
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        candidate_generator="hypothesis_lab",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert [name for name, _ in calls] == ["library", "candidates", "hypothesis_lab"]
    assert report["best_candidate"]["candidate_id"] == "task1_retrieval_prior_k20_affine"
    assert report["best_candidate"]["score"] == 109.0
    assert report["best_candidate"]["metadata"]["source"] == "hypothesis_lab"
    assert report["best_candidate"]["metadata"]["score_delta_vs_baseline"] == 23.0


def test_hermes_batch_optimization_loop_hypothesis_lab_reference_search_sweeps_limits(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    limits_seen: list[int | None] = []

    def fake_library_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {"state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT"}

    def fake_candidate_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {"state_hint": "ALGORITHM_CANDIDATES_COMPILED"}

    def fake_hypothesis_lab(**kwargs: Any) -> dict[str, Any]:
        limit = kwargs.get("reference_sample_limit")
        limits_seen.append(limit)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "sandbox" / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "hypothesis_lab_report_path": str(output_dir / "hypothesis_lab_report.json"),
            },
            "sandbox": {
                "reference_audit": {
                    "uses_target_as_reference": False,
                    "submission_validity": "submission_plausible_reference",
                    "reference_sample_limit": limit,
                },
                "best_candidate": {
                    "candidate_id": f"candidate_limit_{limit}",
                    "params": {"reference_sample_limit": limit},
                },
                "verifier": {"decision": "promote_to_strategy_proposal"},
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        score = 80.0 if Path(kwargs["submission_dir"]).name == "initial" else 90.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "build_local_strategy_library", fake_library_builder)
    monkeypatch.setattr(
        loop_module,
        "build_local_asset_algorithm_candidates",
        fake_candidate_builder,
    )
    monkeypatch.setattr(loop_module, "run_hypothesis_lab", fake_hypothesis_lab)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=3,
        patience=3,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limits=[64, 128, 256],
        candidate_generator="hypothesis_lab_reference_search",
        judge=fake_judge,
        force=False,
    )

    assert limits_seen == [64, 128, 256]
    assert report["round_count"] == 4
    assert report["trajectory"][1]["metadata"]["reference_audit"][
        "reference_sample_limit"
    ] == 64


def test_hermes_batch_optimization_loop_hypothesis_lab_early_search_runs_blueprint(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_early_refinement",
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
                "reference_sample_limit": 128,
            },
            "best_candidate": {
                "candidate_id": "task1_early_refinement_k48_aPhase",
                "params": {"k": 48, "early_anchor": "phase_aligned_retrieval"},
                "task1_total_score_proxy": 88.0,
                "segment_score": 40.0,
                "segments": {"early": {"score": 38.0, "rel_mse": 0.048}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
                "early_segment_diagnosis": {"bottleneck_type": "phase_shift"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 80.0 if is_initial else 88.0
        early_score = 3.0 if is_initial else 38.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.048}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limit=128,
        candidate_generator="hypothesis_lab_early_search",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert sandbox_calls[0]["blueprint"]["candidate_family"] == "task1_early_refinement"
    assert sandbox_calls[0]["reference_path"] == Path("task1_train_reference.hdf5")
    assert sandbox_calls[0]["reference_sample_limit"] == 128
    assert sandbox_calls[0]["timing_profile"] == "mac_mini_m4_local_dev"
    assert report["best_candidate"]["candidate_id"] == "task1_early_refinement_k48_aPhase"
    assert report["best_candidate"]["metadata"]["source"] == "hypothesis_lab_early_search"
    assert report["best_candidate"]["metadata"]["early_segment"]["score"] == 38.0


def test_checkpoint_window_gate_candidate_patches_selected_samples(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_checkpoint_window_gate_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    context[:, 9, :] = 0.25
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 34, :] = 0.25
    base[:, 35:57, :] = 1.0
    checkpoint = np.array(base, copy=True)
    checkpoint[0, 35, :] = 0.251
    checkpoint[1, 35, :] = 1.25
    checkpoint[:, 36:57, :] = 2.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(checkpoint_dir / "task1_pred.hdf5", checkpoint)

    candidate = _build_checkpoint_window_gate_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        checkpoint_submission_dir=checkpoint_dir,
        task1_test_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        feature_name="checkpoint_jump35",
        direction="gt",
        threshold=0.1,
        window_start=35,
        window_stop=57,
        force=False,
    )

    result_path = Path(candidate["submission_dir"]) / "task1_pred.hdf5"
    h5py = __import__("h5py")
    with h5py.File(result_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert np.allclose(prediction[0, 35:57, :], base[0, 35:57, :])
    assert np.allclose(prediction[1, 35:57, :], checkpoint[1, 35:57, :])
    assert candidate["selected_sample_count"] == 1


def test_checkpoint_window_gate_specs_sweep_sources_windows_and_features() -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _checkpoint_window_gate_specs,
    )

    context = np.zeros((4, 10, 8), dtype=np.float32)
    context[:, 8, :] = np.arange(4, dtype=np.float32)[:, None] * 0.1
    context[:, 9, :] = context[:, 8, :] + np.arange(4, dtype=np.float32)[:, None] * 0.01
    base = np.zeros((4, 200, 8), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 34, :] = context[:, 9, :]
    base[:, 10:57, :] = 0.5
    checkpoint_a = np.array(base, copy=True)
    checkpoint_b = np.array(base, copy=True)
    checkpoint_a[:, 35:57, :] += np.arange(4, dtype=np.float32)[:, None, None] * 0.02
    checkpoint_b[:, 10:20, :] += np.arange(4, dtype=np.float32)[:, None, None] * 0.03

    specs = _checkpoint_window_gate_specs(
        checkpoint_submission_dirs=[Path("checkpoint_a"), Path("checkpoint_b")],
        context=context,
        base_prediction=base,
        checkpoint_predictions={
            Path("checkpoint_a"): checkpoint_a,
            Path("checkpoint_b"): checkpoint_b,
        },
    )

    assert any(spec["checkpoint_submission_dir"] == Path("checkpoint_b") for spec in specs)
    assert any(
        spec["window_start"] == 10 and spec["window_stop"] == 20 for spec in specs
    )
    assert any(
        spec["feature_name"] == "base_checkpoint_disagreement_10_20" for spec in specs
    )
    assert all(0 < spec["expected_selected_sample_count"] < 4 for spec in specs)
    assert Path("checkpoint_b") in {
        spec["checkpoint_submission_dir"] for spec in specs[:8]
    }


def test_sem_window_blend_candidate_blends_selected_window(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_blend_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 10:105, :] = 1.0
    sem_prediction = np.array(base, copy=True)
    sem_prediction[:, 10:105, :] = 3.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_sem_window_blend_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_prediction=sem_prediction,
        sem_params={"dt": 0.05, "element_count": 16},
        window_start=10,
        window_stop=105,
        blend_weight=0.25,
        force=False,
    )

    result_path = Path(candidate["submission_dir"]) / "task1_pred.hdf5"
    h5py = __import__("h5py")
    with h5py.File(result_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert np.allclose(prediction[:, :10, :], context)
    assert np.allclose(prediction[:, 10:105, :], 1.5)
    assert np.allclose(prediction[:, 105:, :], base[:, 105:, :])
    assert candidate["source"] == "sem_window_blend"
    assert candidate["blend_weight"] == 0.25


def test_physics_residual_nudge_candidate_preserves_context(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_physics_residual_nudge_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    context[:, 9, :] = 0.1
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 10:105, :] = 0.2

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_physics_residual_nudge_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        correction_weight=0.04,
        stop_step=105,
        viscosity=0.001,
        dt=0.05,
        dx=1.0 / 256.0,
        spectral_strength=0.0,
        force=False,
    )

    result_path = Path(candidate["submission_dir"]) / "task1_pred.hdf5"
    h5py = __import__("h5py")
    with h5py.File(result_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert np.allclose(prediction[:, :10, :], context)
    assert prediction.shape == base.shape
    assert candidate["source"] == "physics_residual_nudge"
    assert candidate["correction_weight"] == 0.04


def test_sem_window_gate_blend_candidate_blends_selected_samples(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    sem_prediction = np.array(base, copy=True)
    sem_prediction[0, 10:105, :] = 0.5
    sem_prediction[1, 10:105, :] = 3.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_sem_window_gate_blend_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_prediction=sem_prediction,
        sem_params={"dt": 0.05},
        feature_name="sem_base_disagreement_20_57",
        direction="lt",
        threshold=1.0,
        window_start=10,
        window_stop=105,
        blend_weight=0.5,
        force=False,
    )

    result_path = Path(candidate["submission_dir"]) / "task1_pred.hdf5"
    h5py = __import__("h5py")
    with h5py.File(result_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert np.allclose(prediction[0, 10:105, :], 0.25)
    assert np.allclose(prediction[1, 10:105, :], base[1, 10:105, :])
    assert candidate["selected_sample_count"] == 1
    assert candidate["source"] == "sem_window_gate_blend"


def test_sem_window_gate_blend_search_materializes_best_spec_only(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_search_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    target = np.array(base, copy=True)
    target[0, 10:105, :] = 0.25
    sem_prediction = np.array(base, copy=True)
    sem_prediction[0, 10:105, :] = 0.5
    sem_prediction[1, 10:105, :] = 3.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_sem_window_gate_blend_search_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_prediction=sem_prediction,
        sem_params={"dt": 0.05},
        specs=[
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "gt",
                "threshold": 1.0,
                "window_start": 10,
                "window_stop": 105,
                "blend_weight": 0.5,
            },
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "lt",
                "threshold": 1.0,
                "window_start": 10,
                "window_stop": 105,
                "blend_weight": 0.5,
            },
        ],
        force=False,
    )

    result_path = Path(candidate["submission_dir"]) / "task1_pred.hdf5"
    h5py = __import__("h5py")
    with h5py.File(result_path, "r") as handle:
        prediction = handle["tensor"][:]
    assert np.allclose(prediction[0, 10:105, :], 0.25)
    assert np.allclose(prediction[1, 10:105, :], 0.0)
    assert candidate["source"] == "sem_window_gate_blend_search"
    assert candidate["search_evaluated_count"] == 2
    assert candidate["direction"] == "lt"
    assert len(list((tmp_path / "candidate").rglob("task1_pred.hdf5"))) == 1


def test_sem_window_gate_blend_multi_search_selects_best_sem_source(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_multi_search_candidate,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    target = np.array(base, copy=True)
    target[0, 35:57, :] = 0.75
    weak_sem = np.array(base, copy=True)
    weak_sem[0, 35:57, :] = 0.2
    strong_sem = np.array(base, copy=True)
    strong_sem[0, 35:57, :] = 1.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_sem_window_gate_blend_multi_search_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_candidates=[
            {"sem_prediction": weak_sem, "sem_params": {"dt": 0.03}},
            {"sem_prediction": strong_sem, "sem_params": {"dt": 0.05}},
        ],
        specs=[
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "gt",
                "threshold": 0.0,
                "window_start": 35,
                "window_stop": 57,
                "blend_weight": 0.75,
            }
        ],
        force=False,
    )

    assert candidate["source"] == "sem_window_gate_blend_multi_search"
    assert candidate["sem_params"]["dt"] == 0.05
    assert candidate["search_evaluated_count"] == 2


def test_sem_window_gate_blend_multi_search_can_select_by_early_segment(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_multi_search_candidate,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    base = np.zeros((1, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    target = np.array(base, copy=True)
    target[:, 10:57, :] = 1.0
    target[:, 57:105, :] = 5.0
    sem_prediction = np.array(base, copy=True)
    sem_prediction[:, 10:57, :] = 1.0
    sem_prediction[:, 57:105, :] = 5.0

    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_target.hdf5", target)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)

    candidate = _build_sem_window_gate_blend_multi_search_candidate(
        output_dir=tmp_path / "candidate",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_candidates=[
            {"sem_prediction": sem_prediction, "sem_params": {"dt": 0.05}},
        ],
        specs=[
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "gt",
                "threshold": 0.0,
                "window_start": 57,
                "window_stop": 105,
                "blend_weight": 1.0,
            },
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "gt",
                "threshold": 0.0,
                "window_start": 10,
                "window_stop": 57,
                "blend_weight": 1.0,
            },
        ],
        selection_metric="early_segment_score",
        force=False,
    )

    assert candidate["window_start"] == 10
    assert candidate["window_stop"] == 57
    assert candidate["validation_selection_metric"] == "early_segment_score"


def test_sem_window_gate_blend_multi_search_runner_includes_pseudospectral(
    tmp_path: Path, monkeypatch: object
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module
    from pde_agent.core.hermes_batch_optimization_loop import (
        _sem_window_gate_blend_multi_search_runner,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    base = np.zeros((1, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    captured: dict[str, Any] = {}

    def fake_sem_rollout(**kwargs: Any) -> np.ndarray:
        return np.zeros((1, 200, 256), dtype=np.float32)

    def fake_pseudospectral_rollout(**kwargs: Any) -> np.ndarray:
        prediction = np.zeros((1, 200, 256), dtype=np.float32)
        prediction[:, 10:, :] = 2.0
        return prediction

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        captured["sem_candidates"] = kwargs["sem_candidates"]
        submission_dir = tmp_path / "candidate_submission"
        submission_dir.mkdir()
        return {
            "candidate_id": "fake",
            "submission_dir": str(submission_dir),
            "source": "sem_window_gate_blend_multi_search",
        }

    monkeypatch.setattr(loop_module, "sem_burgers_rollout", fake_sem_rollout)
    monkeypatch.setattr(
        loop_module,
        "pseudospectral_burgers_rollout",
        fake_pseudospectral_rollout,
    )
    monkeypatch.setattr(
        loop_module,
        "_build_sem_window_gate_blend_multi_search_candidate",
        fake_builder,
    )

    runner = _sem_window_gate_blend_multi_search_runner(
        initial_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile=None,
        timing_factor=None,
        force=False,
    )
    candidate = runner(round_index=1, output_dir=tmp_path / "round")

    assert candidate["source"] == "sem_window_gate_blend_multi_search"
    assert any(
        item["sem_params"].get("source") == "pseudospectral"
        for item in captured["sem_candidates"]
    )


def test_sem_window_gate_blend_multi_search_smoke_avoids_slow_sem_rollout(
    tmp_path: Path, monkeypatch: object
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module
    from pde_agent.core.hermes_batch_optimization_loop import (
        _sem_window_gate_blend_multi_search_runner,
    )

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    captured: dict[str, Any] = {}

    def fail_sem_rollout(**kwargs: Any) -> np.ndarray:
        raise AssertionError("smoke mode should not call slow SEM rollout")

    def fake_pseudospectral_rollout(**kwargs: Any) -> np.ndarray:
        prediction = np.zeros((2, 200, 256), dtype=np.float32)
        prediction[1:, 10:, :] = 1.0
        return prediction

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        captured["sem_candidates"] = kwargs["sem_candidates"]
        captured["specs"] = kwargs["specs"]
        submission_dir = tmp_path / "candidate_submission"
        submission_dir.mkdir()
        return {
            "candidate_id": "fake_smoke",
            "submission_dir": str(submission_dir),
            "source": "sem_window_gate_blend_multi_search",
        }

    monkeypatch.setattr(loop_module, "sem_burgers_rollout", fail_sem_rollout)
    monkeypatch.setattr(
        loop_module,
        "pseudospectral_burgers_rollout",
        fake_pseudospectral_rollout,
    )
    monkeypatch.setattr(
        loop_module,
        "_build_sem_window_gate_blend_multi_search_candidate",
        fake_builder,
    )

    runner = _sem_window_gate_blend_multi_search_runner(
        initial_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile=None,
        timing_factor=None,
        force=False,
        selection_metric="early_segment_score",
        search_budget="smoke",
    )
    candidate = runner(round_index=1, output_dir=tmp_path / "round")

    assert candidate["candidate_id"] == "fake_smoke"
    assert 1 <= len(captured["sem_candidates"]) <= 2
    assert 1 <= len(captured["specs"]) <= 300
    assert all(
        item["sem_params"].get("source") == "pseudospectral"
        for item in captured["sem_candidates"]
    )


def test_sem_window_gate_blend_multi_search_uses_near_miss_memory_guidance(
    tmp_path: Path, monkeypatch: object
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module
    from pde_agent.core.hermes_batch_optimization_loop import (
        _sem_window_gate_blend_multi_search_runner,
    )

    context = np.zeros((4, 10, 256), dtype=np.float32)
    base = np.zeros((4, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    captured: dict[str, Any] = {}

    def fail_sem_rollout(**kwargs: Any) -> np.ndarray:
        raise AssertionError("smoke mode should not call slow SEM rollout")

    def fake_pseudospectral_rollout(**kwargs: Any) -> np.ndarray:
        prediction = np.zeros((4, 200, 256), dtype=np.float32)
        for index in range(4):
            prediction[index, 20:57, :] = float(index)
        return prediction

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        captured["specs"] = kwargs["specs"]
        submission_dir = tmp_path / "candidate_submission"
        submission_dir.mkdir()
        return {
            "candidate_id": "fake_guided",
            "submission_dir": str(submission_dir),
            "source": "sem_window_gate_blend_multi_search",
        }

    monkeypatch.setattr(loop_module, "sem_burgers_rollout", fail_sem_rollout)
    monkeypatch.setattr(
        loop_module,
        "pseudospectral_burgers_rollout",
        fake_pseudospectral_rollout,
    )
    monkeypatch.setattr(
        loop_module,
        "_build_sem_window_gate_blend_multi_search_candidate",
        fake_builder,
    )

    runner = _sem_window_gate_blend_multi_search_runner(
        initial_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=test_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile=None,
        timing_factor=None,
        force=False,
        search_guidance=[
            {
                "guidance_type": "refine_gating_after_near_miss",
                "reason": "near_miss_with_empty_gate_selection",
                "source": "sem_window_gate_blend_multi_search",
                "feature_name": "sem_base_disagreement_20_57",
                "window_start": 10,
                "window_stop": 20,
            }
        ],
        selection_metric="early_segment_score",
        search_budget="smoke",
    )
    candidate = runner(round_index=1, output_dir=tmp_path / "round")

    guided_specs = [
        spec
        for spec in captured["specs"]
        if spec.get("memory_guidance_reason") == "near_miss_with_empty_gate_selection"
    ]
    assert candidate["candidate_id"] == "fake_guided"
    assert guided_specs
    assert any(spec["blend_weight"] == 0.01 for spec in guided_specs)
    assert all(spec["selected_sample_count"] > 0 for spec in guided_specs)


def test_sem_window_gate_blend_multi_search_no_improvement_uses_valid_noop_window(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_multi_search_candidate,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", target)

    candidate = _build_sem_window_gate_blend_multi_search_candidate(
        output_dir=tmp_path / "round",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_candidates=[
            {
                "sem_prediction": np.ones((1, 200, 256), dtype=np.float32),
                "sem_params": {"source": "pseudospectral"},
            }
        ],
        specs=[],
        selection_metric="early_segment_score",
        force=False,
    )

    assert candidate["blend_weight"] == 0.0
    assert candidate["window_stop"] > candidate["window_start"]


def test_sem_window_gate_blend_multi_search_reports_memory_guidance_spec_count(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        _build_sem_window_gate_blend_multi_search_candidate,
    )

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    base_dir = tmp_path / "base"
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", target)

    candidate = _build_sem_window_gate_blend_multi_search_candidate(
        output_dir=tmp_path / "round",
        base_submission_dir=base_dir,
        task1_test_path=test_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        sem_candidates=[
            {
                "sem_prediction": np.ones((1, 200, 256), dtype=np.float32),
                "sem_params": {"source": "pseudospectral"},
            }
        ],
        specs=[
            {
                "feature_name": "sem_base_disagreement_20_57",
                "direction": "gt",
                "threshold": 0.1,
                "window_start": 10,
                "window_stop": 20,
                "blend_weight": 0.01,
                "selected_sample_count": 1,
                "memory_guidance_reason": "near_miss_with_empty_gate_selection",
                "memory_guidance_action": (
                    "relax_or_recenter_gate_threshold_and_window_before_repeating_family"
                ),
            }
        ],
        selection_metric="early_segment_score",
        force=False,
    )
    search_report = json.loads(Path(candidate["search_report_path"]).read_text())

    assert candidate["memory_guidance_spec_count"] == 1
    assert search_report["memory_guidance_spec_count"] == 1


def test_hermes_batch_optimization_loop_hypothesis_lab_early_transition_search_runs_blueprint(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_early_transition_expert",
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
                "reference_sample_limit": 128,
            },
            "best_candidate": {
                "candidate_id": "task1_early_transition_expert_k48_velocity_ridge_h47",
                "params": {"k": 48, "transition_mode": "velocity_ridge"},
                "task1_total_score_proxy": 89.0,
                "segment_score": 41.0,
                "segments": {"early": {"score": 42.0, "rel_mse": 0.043}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
                "early_segment_diagnosis": {"bottleneck_type": "transport_lag"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 80.0 if is_initial else 89.0
        early_score = 3.0 if is_initial else 42.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.043}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limit=128,
        candidate_generator="hypothesis_lab_early_transition_search",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert sandbox_calls[0]["blueprint"]["candidate_family"] == (
        "task1_early_transition_expert"
    )
    assert sandbox_calls[0]["blueprint"]["sandbox"]["allowed_executor"] == (
        "controlled_numpy_task1_early_transition_expert"
    )
    assert sandbox_calls[0]["reference_sample_limit"] == 128
    assert report["best_candidate"]["candidate_id"].startswith(
        "task1_early_transition_expert"
    )
    assert report["best_candidate"]["metadata"]["source"] == (
        "hypothesis_lab_early_transition_search"
    )
    assert report["best_candidate"]["metadata"]["early_segment"]["score"] == 42.0


def test_hermes_batch_optimization_loop_front25_dynamics_smoke_runs_only_front25(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_front25_dynamics",
            "reference_audit": {"reference_sample_limit": 10},
            "best_candidate": {
                "candidate_id": "task1_front25_dynamics_smoke",
                "params": {"dynamics_mode": "pseudospectral"},
                "task1_total_score_proxy": 88.0,
                "segment_score": 40.0,
                "segments": {"early": {"score": 43.0, "rel_mse": 0.041}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
                "early_segment_diagnosis": {"bottleneck_type": "transport_lag"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 80.0 if is_initial else 88.0
        early_score = 3.0 if is_initial else 43.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.041}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limit=10,
        candidate_generator="hypothesis_lab_front25_dynamics_smoke",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert len(sandbox_calls) == 1
    assert sandbox_calls[0]["blueprint"]["candidate_family"] == "task1_front25_dynamics"
    assert len(sandbox_calls[0]["blueprint"]["parameter_grid"]["candidate_params"]) <= 12
    assert report["best_candidate"]["metadata"]["source"] == (
        "hypothesis_lab_front25_dynamics_smoke"
    )


def test_hermes_batch_optimization_loop_short_window_corrector_smoke_runs_new_family(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_short_window_corrector",
            "reference_audit": {
                "reference_sample_limit": 10,
                "uses_target_as_reference": False,
            },
            "best_candidate": {
                "candidate_id": "task1_short_window_corrector_smoke",
                "params": {"start_step": 20, "stop_step": 35},
                "task1_total_score_proxy": 91.0,
                "segment_score": 44.0,
                "segments": {"early": {"score": 44.0, "rel_mse": 0.039}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
                "early_segment_diagnosis": {"bottleneck_type": "early_middle_window"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 80.0 if is_initial else 91.0
        early_score = 3.0 if is_initial else 44.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.039}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="official_train.hdf5",
        hypothesis_reference_sample_limit=10,
        candidate_generator="hypothesis_lab_short_window_corrector_smoke",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert len(sandbox_calls) == 1
    assert sandbox_calls[0]["blueprint"]["candidate_family"] == (
        "task1_short_window_corrector"
    )
    assert sandbox_calls[0]["blueprint"]["sandbox"]["allowed_executor"] == (
        "controlled_numpy_task1_short_window_corrector"
    )
    assert sandbox_calls[0]["reference_sample_limit"] == 10
    assert report["best_candidate"]["metadata"]["source"] == (
        "hypothesis_lab_short_window_corrector_smoke"
    )


def test_hermes_batch_optimization_loop_short_window_corrector_full_uses_larger_grid(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_short_window_corrector",
            "reference_audit": {"uses_target_as_reference": False},
            "best_candidate": {
                "candidate_id": "task1_short_window_corrector_full",
                "params": {"start_step": 35, "stop_step": 57},
                "task1_total_score_proxy": 92.0,
                "segment_score": 45.0,
                "segments": {"early": {"score": 45.0, "rel_mse": 0.038}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 80.0 if is_initial else 92.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {"early": {"score": 45.0, "rel_mse": 0.038}}
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="early_accuracy",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="official_train.hdf5",
        hypothesis_reference_sample_limit=10,
        candidate_generator="hypothesis_lab_short_window_corrector_full",
        judge=fake_judge,
        force=False,
    )

    full_grid = sandbox_calls[0]["blueprint"]["parameter_grid"]["candidate_params"]
    assert len(full_grid) > 20
    assert {row["start_step"] for row in full_grid} >= {10, 20, 35}
    assert all(
        row["stop_step"] <= (200 + row["reference_time_stride"] - 1) // row["reference_time_stride"]
        for row in full_grid
    )


def test_hermes_batch_optimization_loop_short_window_corrector_patch_sources_composes_source(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []
    patch_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_short_window_corrector",
            "reference_audit": {"uses_target_as_reference": False},
            "best_candidate": {
                "candidate_id": "task1_short_window_corrector_source",
                "params": {"start_step": 20, "stop_step": 35},
                "task1_total_score_proxy": 92.0,
                "segment_score": 45.0,
                "segments": {"early": {"score": 45.0, "rel_mse": 0.038}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "early_rel_mse"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = tmp_path / "patched"
        submission_dir.mkdir()
        return {
            "candidate_id": "short_window_patch_sources_candidate",
            "submission_dir": str(submission_dir),
            "source": "hypothesis_lab_short_window_corrector_patch_sources",
            "window_sources": [],
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_patched = Path(kwargs["submission_dir"]).name == "patched"
        score = 93.0 if is_patched else 80.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {"early": {"score": 45.0, "rel_mse": 0.038}}
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    monkeypatch.setattr(
        loop_module,
        "_build_early_subsegment_patch_candidate",
        fake_patch_builder,
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="official_train.hdf5",
        hypothesis_reference_sample_limit=10,
        candidate_generator="hypothesis_lab_short_window_corrector_patch_sources",
        judge=fake_judge,
        force=False,
    )

    assert sandbox_calls[0]["blueprint"]["candidate_family"] == (
        "task1_short_window_corrector"
    )
    assert patch_calls[0]["source_candidates"][0]["candidate_id"] == (
        "task1_short_window_corrector_source"
    )
    assert patch_calls[0]["source_candidates"][0]["submission_dir"].endswith(
        "submission_package/submission"
    )
    assert report["best_candidate"]["candidate_id"] == (
        "short_window_patch_sources_candidate"
    )


def test_hermes_batch_optimization_loop_checkpoint_front_window_adapter_composes_sources(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    checkpoint = np.array(base, copy=True)
    checkpoint[:, 20:35, :] = 2.0
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", base)
    initial_dir = tmp_path / "initial"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_hdf5_tensor(initial_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(checkpoint_dir / "task1_pred.hdf5", checkpoint)
    patch_calls: list[dict[str, Any]] = []

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = tmp_path / "patched"
        submission_dir.mkdir()
        return {
            "candidate_id": "checkpoint_front_window_adapter_patch_sources",
            "submission_dir": str(submission_dir),
            "source": "checkpoint_front_window_adapter_patch_sources",
            "window_sources": [],
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        score = 110.0 if Path(kwargs["submission_dir"]).name == "patched" else 100.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {"early": {"score": 1.0, "rel_mse": 0.2}}
                    },
                }
            },
        }

    monkeypatch.setattr(
        loop_module,
        "_build_early_subsegment_patch_candidate",
        fake_patch_builder,
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path=test_path,
        task1_target_path=target_path,
        candidate_submission_dirs=[checkpoint_dir],
        candidate_generator="checkpoint_front_window_adapter_patch_sources",
        judge=fake_judge,
        force=False,
    )

    source_candidates = patch_calls[0]["source_candidates"]
    assert {item["adapter_kind"] for item in source_candidates} >= {
        "direct",
        "blend",
        "residual",
    }
    assert {item["window_start"] for item in source_candidates} >= {20, 35}
    assert all(item["rules_passed"] for item in source_candidates)
    assert report["best_candidate"]["candidate_id"] == (
        "checkpoint_front_window_adapter_patch_sources"
    )


def test_hermes_batch_optimization_loop_transition_then_early_patch_uses_transition_as_base(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []
    patch_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_early_transition_expert",
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
                "reference_sample_limit": 256,
            },
            "best_candidate": {
                "candidate_id": "task1_early_transition_expert_k64_delta_mean_h47",
                "params": {"k": 64, "transition_mode": "delta_mean"},
                "task1_total_score_proxy": 107.3,
                "segment_score": 43.1,
                "segments": {"early": {"score": 0.16, "rel_mse": 0.32}},
            },
            "verifier": {"decision": "revise_hypothesis"},
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = Path(kwargs["output_dir"]) / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "candidate_id": (
                "early_segment_patch_initial_into_"
                "task1_early_transition_expert_k64_delta_mean_h47"
            ),
            "submission_dir": str(submission_dir),
            "source": "early_segment_patch",
            "base_candidate_id": kwargs["base_candidate"]["candidate_id"],
            "early_source_candidate_id": kwargs["early_source_candidate"][
                "candidate_id"
            ],
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        name = str(kwargs["submission_dir"])
        if "round_002" in name:
            score = 108.2
            early_score = 4.5
        elif "round_001" in name:
            score = 107.3
            early_score = 0.16
        else:
            score = 107.6
            early_score = 4.5
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 75.0,
                        "segment_score": score - 64.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.15}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    monkeypatch.setattr(
        loop_module,
        "_build_early_segment_patch_candidate",
        fake_patch_builder,
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limit=256,
        candidate_generator="hypothesis_lab_transition_early_patch",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert sandbox_calls[0]["blueprint"]["candidate_family"] == (
        "task1_early_transition_expert"
    )
    assert patch_calls[0]["base_candidate"]["candidate_id"].startswith(
        "task1_early_transition_expert"
    )
    assert patch_calls[0]["early_source_candidate"]["candidate_id"] == "initial"
    assert report["best_candidate"]["candidate_id"].startswith("early_segment_patch")
    assert report["best_candidate"]["metadata"]["base_candidate_id"].startswith(
        "task1_early_transition_expert"
    )


def test_hermes_batch_optimization_loop_middle_late_patch_search_runs_blueprint(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    sandbox_calls: list[dict[str, Any]] = []
    patch_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "blueprint_id": kwargs["blueprint"]["blueprint_id"],
            "candidate_family": "task1_retrieval_refinement_local_search",
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
                "reference_sample_limit": 256,
            },
            "best_candidate": {
                "candidate_id": "task1_retrieval_refinement_k72_affine_eb0_pw0_s0",
                "params": {"k": 72, "mode": "affine"},
                "task1_total_score_proxy": 107.5,
                "segment_score": 43.8,
                "segments": {"early": {"score": 0.1, "rel_mse": 0.31}},
            },
            "verifier": {
                "decision": "revise_hypothesis",
                "selection_objective": {"primary": "middle_late_score"},
            },
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "hypothesis_lab_sandbox_report.json"),
            },
        }

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = Path(kwargs["output_dir"]) / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "candidate_id": (
                "early_segment_patch_initial_into_"
                "task1_retrieval_refinement_k72_affine_eb0_pw0_s0"
            ),
            "submission_dir": str(submission_dir),
            "source": "early_segment_patch",
            "base_candidate_id": kwargs["base_candidate"]["candidate_id"],
            "early_source_candidate_id": kwargs["early_source_candidate"][
                "candidate_id"
            ],
            "reference_audit": kwargs["base_candidate"]["reference_audit"],
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        name = str(kwargs["submission_dir"])
        if "round_002" in name:
            score = 108.5
            early_score = 4.5
        elif "round_001" in name:
            score = 107.5
            early_score = 0.1
        else:
            score = 107.6
            early_score = 4.5
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 75.0,
                        "segment_score": score - 64.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.15}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    monkeypatch.setattr(
        loop_module,
        "_build_early_segment_patch_candidate",
        fake_patch_builder,
    )
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        hypothesis_reference_path="task1_train_reference.hdf5",
        hypothesis_reference_sample_limit=256,
        candidate_generator="hypothesis_lab_middle_late_patch_search",
        timing_profile="mac_mini_m4_local_dev",
        judge=fake_judge,
        force=False,
    )

    assert sandbox_calls[0]["blueprint"]["candidate_family"] == (
        "task1_retrieval_refinement_local_search"
    )
    assert sandbox_calls[0]["blueprint"]["sandbox"]["selection_objective"] == (
        "middle_late_score"
    )
    assert patch_calls[0]["base_candidate"]["candidate_id"].startswith(
        "task1_retrieval_refinement"
    )
    assert patch_calls[0]["early_source_candidate"]["candidate_id"] == "initial"
    assert report["best_candidate"]["candidate_id"].startswith("early_segment_patch")
    assert report["best_candidate"]["metadata"]["reference_audit"][
        "uses_target_as_reference"
    ] is False


def test_early_segment_patch_preserves_top_level_reference_audit(
    tmp_path: Path,
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    early = np.ones((2, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    early[:, :10, :] = context
    base_dir = tmp_path / "base"
    early_dir = tmp_path / "early"
    base_dir.mkdir()
    early_dir.mkdir()
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(early_dir / "task1_pred.hdf5", early)
    context_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    reference_audit = {
        "uses_target_as_reference": False,
        "submission_validity": "submission_plausible_reference",
        "reference_sample_limit": 256,
    }

    candidate = loop_module._build_early_segment_patch_candidate(
        output_dir=tmp_path / "patch",
        base_candidate={
            "candidate_id": "transition",
            "submission_dir": str(base_dir),
            "reference_audit": reference_audit,
        },
        early_source_candidate={
            "candidate_id": "initial",
            "submission_dir": str(early_dir),
        },
        task1_test_path=context_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        force=False,
    )

    assert candidate["reference_audit"] == reference_audit


def test_early_subsegment_patch_selects_best_source_per_window(
    tmp_path: Path,
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    target[:, 10:20, :] = 1.0
    target[:, 20:35, :] = 2.0
    target[:, 35:57, :] = 3.0
    target[:, 57:, :] = 9.0
    base = np.zeros((1, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    base[:, 57:, :] = 9.0
    source_a = np.zeros((1, 200, 256), dtype=np.float32)
    source_a[:, :10, :] = context
    source_a[:, 10:20, :] = 1.0
    source_a[:, 20:35, :] = 9.0
    source_a[:, 35:57, :] = 9.0
    source_b = np.zeros((1, 200, 256), dtype=np.float32)
    source_b[:, :10, :] = context
    source_b[:, 10:20, :] = 8.0
    source_b[:, 20:35, :] = 2.0
    source_b[:, 35:57, :] = 3.0

    context_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    base_dir = tmp_path / "base"
    source_a_dir = tmp_path / "source_a"
    source_b_dir = tmp_path / "source_b"
    base_dir.mkdir()
    source_a_dir.mkdir()
    source_b_dir.mkdir()
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(source_a_dir / "task1_pred.hdf5", source_a)
    _write_hdf5_tensor(source_b_dir / "task1_pred.hdf5", source_b)

    candidate = loop_module._build_early_subsegment_patch_candidate(
        output_dir=tmp_path / "patch",
        base_candidate={
            "candidate_id": "middle_late_base",
            "submission_dir": str(base_dir),
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
            },
        },
        source_candidates=[
            {"candidate_id": "source_a", "submission_dir": str(source_a_dir)},
            {"candidate_id": "source_b", "submission_dir": str(source_b_dir)},
        ],
        task1_test_path=context_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        force=False,
    )

    patched = _load_test_hdf5(Path(candidate["submission_dir"]) / "task1_pred.hdf5")
    assert np.all(patched[:, 10:20, :] == 1.0)
    assert np.all(patched[:, 20:35, :] == 2.0)
    assert np.all(patched[:, 35:57, :] == 3.0)
    assert np.all(patched[:, 57:, :] == 9.0)
    assert candidate["window_sources"] == [
        {
            "time_start": 10,
            "time_stop": 20,
            "source_candidate_id": "source_a",
            "rel_mse": 0.0,
        },
        {
            "time_start": 20,
            "time_stop": 35,
            "source_candidate_id": "source_b",
            "rel_mse": 0.0,
        },
        {
            "time_start": 35,
            "time_stop": 57,
            "source_candidate_id": "source_b",
            "rel_mse": 0.0,
        },
    ]


def test_early_subsegment_patch_source_stability_guard_uses_single_best_source(
    tmp_path: Path,
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    target[:, 10:20, :] = 1.0
    target[:, 20:35, :] = 2.0
    target[:, 35:57, :] = 3.0
    base = np.zeros((1, 200, 256), dtype=np.float32)
    base[:, :10, :] = context
    source_a = np.zeros((1, 200, 256), dtype=np.float32)
    source_a[:, :10, :] = context
    source_a[:, 10:20, :] = 1.0
    source_a[:, 20:35, :] = 9.0
    source_a[:, 35:57, :] = 9.0
    source_b = np.zeros((1, 200, 256), dtype=np.float32)
    source_b[:, :10, :] = context
    source_b[:, 10:20, :] = 1.2
    source_b[:, 20:35, :] = 2.0
    source_b[:, 35:57, :] = 3.0

    context_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    base_dir = tmp_path / "base"
    source_a_dir = tmp_path / "source_a"
    source_b_dir = tmp_path / "source_b"
    base_dir.mkdir()
    source_a_dir.mkdir()
    source_b_dir.mkdir()
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(source_a_dir / "task1_pred.hdf5", source_a)
    _write_hdf5_tensor(source_b_dir / "task1_pred.hdf5", source_b)

    candidate = loop_module._build_early_subsegment_patch_candidate(
        output_dir=tmp_path / "patch",
        base_candidate={"candidate_id": "middle_late_base", "submission_dir": str(base_dir)},
        source_candidates=[
            {"candidate_id": "source_a", "submission_dir": str(source_a_dir)},
            {"candidate_id": "source_b", "submission_dir": str(source_b_dir)},
        ],
        task1_test_path=context_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        source_stability_guard=True,
        force=False,
    )

    patched = _load_test_hdf5(Path(candidate["submission_dir"]) / "task1_pred.hdf5")
    assert np.allclose(patched[:, 10:20, :], 1.2)
    assert np.allclose(patched[:, 20:35, :], 2.0)
    assert np.allclose(patched[:, 35:57, :], 3.0)
    assert {row["source_candidate_id"] for row in candidate["window_sources"]} == {
        "source_b"
    }
    assert candidate["source_stability_guard"] is True


def test_early_subsegment_patch_handoff_smoothing_blends_source_with_base(
    tmp_path: Path,
) -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.ones((1, 200, 256), dtype=np.float32)
    target[:, :10, :] = context
    base = np.zeros((1, 200, 256), dtype=np.float32)
    source = np.ones((1, 200, 256), dtype=np.float32)
    source[:, :10, :] = context
    context_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "source"
    base_dir.mkdir()
    source_dir.mkdir()
    _write_hdf5_tensor(base_dir / "task1_pred.hdf5", base)
    _write_hdf5_tensor(source_dir / "task1_pred.hdf5", source)

    candidate = loop_module._build_early_subsegment_patch_candidate(
        output_dir=tmp_path / "patch",
        base_candidate={"candidate_id": "middle_late_base", "submission_dir": str(base_dir)},
        source_candidates=[{"candidate_id": "source", "submission_dir": str(source_dir)}],
        task1_test_path=context_path,
        task1_target_path=target_path,
        dataset_key="tensor",
        prediction_dataset_key="tensor",
        submission_id="team_hermes",
        timing_profile="mac_mini_m4_local_dev",
        timing_factor=None,
        handoff_smoothing_weight=0.5,
        force=False,
    )

    patched = _load_test_hdf5(Path(candidate["submission_dir"]) / "task1_pred.hdf5")
    assert np.allclose(patched[:, 10:57, :], 0.5)
    assert candidate["handoff_smoothing_weight"] == 0.5


def test_hermes_batch_optimization_loop_early_subsegment_patch_search_runs_stages(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    reference_path = _write_hdf5_tensor(tmp_path / "reference.hdf5", target)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    _write_hdf5_tensor(initial_dir / "task1_pred.hdf5", target)
    sandbox_calls: list[dict[str, Any]] = []
    patch_calls: list[dict[str, Any]] = []

    def fake_blueprint_sandbox(**kwargs: Any) -> dict[str, Any]:
        sandbox_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        _write_hdf5_tensor(submission_dir / "task1_pred.hdf5", target)
        family = kwargs["blueprint"]["candidate_family"]
        if kwargs["reference_sample_limit"] == 64:
            params = kwargs["blueprint"]["parameter_grid"]["candidate_params"]
            for row in params:
                for key in ("k", "base_k"):
                    if key in row:
                        assert row[key] <= 64
        if family == "task1_early_refinement":
            assert kwargs["reference_sample_limit"] == 64
            candidate_id = "early_refinement_candidate"
            selection = {"primary": "early_rel_mse"}
        elif family == "task1_front25_dynamics":
            candidate_id = "front25_dynamics_candidate"
            selection = {"primary": "early_rel_mse"}
        elif family == "task1_early_transition_expert":
            candidate_id = "early_transition_candidate"
            selection = {"primary": "early_rel_mse"}
        else:
            candidate_id = "middle_late_candidate"
            selection = {"primary": "middle_late_score"}
        return {
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "sandbox_report_path": str(output_dir / "sandbox.json"),
            },
            "best_candidate": {
                "candidate_id": candidate_id,
                "params": {},
                "segments": {"early": {"score": 2.0, "rel_mse": 0.2}},
            },
            "verifier": {
                "decision": "promote_to_strategy_proposal",
                "selection_objective": selection,
            },
            "reference_audit": {
                "uses_target_as_reference": False,
                "submission_validity": "submission_plausible_reference",
            },
        }

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = tmp_path / "patched"
        submission_dir.mkdir()
        return {
            "candidate_id": "early_subsegment_patch_candidate",
            "submission_dir": str(submission_dir),
            "source": "early_subsegment_patch",
            "window_sources": [],
            "reference_audit": kwargs["base_candidate"].get("reference_audit"),
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        name = Path(kwargs["submission_dir"]).name
        score = {
            "initial": 105.0,
            "submission": 101.0,
            "patched": 110.0,
        }[name]
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": 1.0, "rel_mse": 0.2}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_blueprint_sandbox", fake_blueprint_sandbox)
    monkeypatch.setattr(
        loop_module,
        "_build_early_subsegment_patch_candidate",
        fake_patch_builder,
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=5,
        patience=5,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path=test_path,
        task1_target_path=target_path,
        hypothesis_reference_path=reference_path,
        hypothesis_reference_sample_limit=64,
        candidate_generator="hypothesis_lab_early_subsegment_patch_search",
        submission_id="team_patch_test",
        judge=fake_judge,
        force=False,
    )

    assert [
        call["blueprint"]["candidate_family"] for call in sandbox_calls
    ] == [
        "task1_early_refinement",
        "task1_front25_dynamics",
        "task1_early_transition_expert",
        "task1_retrieval_refinement_local_search",
    ]
    assert sandbox_calls[3]["blueprint"]["sandbox"]["selection_objective"] == (
        "middle_late_score"
    )
    assert patch_calls[0]["base_candidate"]["candidate_id"] == "initial"
    assert {
        item["candidate_id"] for item in patch_calls[0]["source_candidates"]
    } >= {
        "initial",
        "early_refinement_candidate",
        "front25_dynamics_candidate",
        "early_transition_candidate",
    }
    assert report["best_candidate"]["candidate_id"] == "early_subsegment_patch_candidate"


def test_hermes_batch_optimization_loop_early_subsegment_patch_sources_uses_candidate_dirs(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    initial_dir = tmp_path / "initial"
    source_dir = tmp_path / "source_a"
    initial_dir.mkdir()
    source_dir.mkdir()
    _write_hdf5_tensor(initial_dir / "task1_pred.hdf5", target)
    _write_hdf5_tensor(source_dir / "task1_pred.hdf5", target)
    patch_calls: list[dict[str, Any]] = []

    def fake_patch_builder(**kwargs: Any) -> dict[str, Any]:
        patch_calls.append(kwargs)
        submission_dir = tmp_path / "patched"
        submission_dir.mkdir()
        return {
            "candidate_id": "early_subsegment_patch_sources_candidate",
            "submission_dir": str(submission_dir),
            "source": "early_subsegment_patch_sources",
            "window_sources": [],
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        score = 110.0 if Path(kwargs["submission_dir"]).name == "patched" else 100.0
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {"early": {"score": 1.0, "rel_mse": 0.2}}
                    },
                }
            },
        }

    monkeypatch.setattr(
        loop_module,
        "_build_early_subsegment_patch_candidate",
        fake_patch_builder,
    )

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path=test_path,
        task1_target_path=target_path,
        candidate_submission_dirs=[source_dir],
        candidate_generator="early_subsegment_patch_sources",
        submission_id="team_patch_sources_test",
        judge=fake_judge,
        force=False,
    )

    assert patch_calls[0]["base_candidate"]["candidate_id"] == "initial"
    assert patch_calls[0]["source_candidates"] == [
        {
            "candidate_id": "source_000_source_a",
            "submission_dir": str(source_dir),
            "rules_passed": True,
            "source": "candidate_submission_dir",
        }
    ]
    assert report["best_candidate"]["candidate_id"] == (
        "early_subsegment_patch_sources_candidate"
    )


def test_early_refinement_candidate_params_include_short_window_micro_search() -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    params = loop_module._early_refinement_candidate_params()

    assert any(
        row["early_blend_steps"] == 10
        and row["persistence_weight_start"] == 0.25
        and row["spectral_strength"] == 0.0
        and row["early_anchor"] == "linear_velocity"
        for row in params
    )
    assert any(
        row["early_blend_steps"] == 25
        and row["persistence_weight_start"] == 0.25
        and row["spectral_strength"] == 0.0
        and row["early_anchor"] == "retrieval_derivative_match"
        for row in params
    )
    assert any(
        row["k"] == 96
        and row["early_blend_steps"] == 25
        and row["persistence_weight_start"] == 0.5
        and row["early_anchor"] == "phase_aligned_retrieval"
        for row in params
    )


def test_front25_dynamics_candidate_params_include_pseudospectral_source() -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    params = loop_module._front25_dynamics_candidate_params()

    assert any(
        row["dynamics_mode"] == "pseudospectral"
        and row["horizon_steps"] == 25
        and row["pseudospectral_dt"] == 0.03
        and row["pseudospectral_substeps"] == 4
        for row in params
    )


def test_front25_dynamics_smoke_budget_is_small_and_diverse() -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    params = loop_module._front25_dynamics_candidate_params(search_budget="smoke")

    assert 1 <= len(params) <= 12
    assert {row["dynamics_mode"] for row in params} >= {
        "velocity_feature_ridge",
        "hybrid",
        "pseudospectral",
    }


def test_checkpoint_window_gate_specs_include_late_window() -> None:
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((2, 10, 256), dtype=np.float32)
    base = np.zeros((2, 200, 256), dtype=np.float32)
    checkpoint = np.array(base, copy=True)
    checkpoint[1:, 105:200, :] = 1.0
    checkpoint_dir = Path("checkpoint_late")

    specs = loop_module._checkpoint_window_gate_specs(
        checkpoint_submission_dirs=[checkpoint_dir],
        context=context,
        base_prediction=base,
        checkpoint_predictions={checkpoint_dir: checkpoint},
    )

    assert any(
        spec["window_start"] == 105
        and spec["window_stop"] == 200
        and spec["feature_name"] == "base_checkpoint_disagreement_105_200"
        for spec in specs
    )


def test_hermes_batch_optimization_loop_reference_then_early_patch_splices_best_segments(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    context = np.zeros((1, 10, 256), dtype=np.float32)
    target = np.zeros((1, 200, 256), dtype=np.float32)
    test_path = _write_hdf5_tensor(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5_tensor(tmp_path / "task1_val.hdf5", target)
    reference_path = _write_hdf5_tensor(tmp_path / "reference.hdf5", target)

    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    initial_prediction = np.zeros((1, 200, 256), dtype=np.float32)
    initial_prediction[:, 10:57, :] = 1.0
    _write_hdf5_tensor(initial_dir / "task1_pred.hdf5", initial_prediction)

    def fake_hypothesis_lab(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "sandbox" / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        reference_prediction = np.zeros((1, 200, 256), dtype=np.float32)
        reference_prediction[:, 10:57, :] = 2.0
        reference_prediction[:, 57:, :] = 5.0
        _write_hdf5_tensor(submission_dir / "task1_pred.hdf5", reference_prediction)
        return {
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "hypothesis_lab_report_path": str(output_dir / "hypothesis_lab_report.json"),
            },
            "sandbox": {
                "reference_audit": {
                    "uses_target_as_reference": False,
                    "submission_validity": "submission_plausible_reference",
                    "reference_sample_limit": 64,
                },
                "best_candidate": {
                    "candidate_id": "reference_middle_late",
                    "params": {},
                },
                "verifier": {"decision": "promote_to_strategy_proposal"},
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        prediction_path = Path(kwargs["submission_dir"]) / "task1_pred.hdf5"
        prediction = _load_test_hdf5(prediction_path)
        early_mean = float(np.mean(prediction[:, 10:57, :]))
        late_mean = float(np.mean(prediction[:, 57:, :]))
        if early_mean == 1.0 and late_mean == 0.0:
            score = 90.0
            early_score = 50.0
        elif early_mean == 2.0 and late_mean == 5.0:
            score = 106.0
            early_score = 1.0
        elif early_mean == 1.0 and late_mean == 5.0:
            score = 107.0
            early_score = 50.0
        else:
            raise AssertionError((early_mean, late_mean))
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                    "validation_score": {
                        "segments": {
                            "early": {"score": early_score, "rel_mse": 0.05}
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "run_hypothesis_lab", fake_hypothesis_lab)

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=2,
        patience=2,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="score",
        task1_test_path=test_path,
        task1_target_path=target_path,
        hypothesis_reference_path=reference_path,
        hypothesis_reference_sample_limits=[64],
        candidate_generator="hypothesis_lab_reference_early_patch",
        submission_id="team_patch_test",
        judge=fake_judge,
        force=False,
    )

    assert report["best_candidate"]["candidate_id"].startswith("early_segment_patch")
    assert report["best_candidate"]["score"] == 107.0
    assert report["best_candidate"]["metadata"]["source"] == "early_segment_patch"
    assert report["best_candidate"]["metadata"]["early_source_candidate_id"] == "initial"
    assert report["best_candidate"]["metadata"]["base_candidate_id"] == "reference_middle_late"
    assert report["best_candidate"]["metadata"]["reference_audit"] == {
        "uses_target_as_reference": False,
        "submission_validity": "submission_plausible_reference",
        "reference_sample_limit": 64,
    }
    patched = _load_test_hdf5(
        Path(report["best_candidate"]["submission_dir"]) / "task1_pred.hdf5"
    )
    assert np.all(patched[:, 10:57, :] == 1.0)
    assert np.all(patched[:, 57:, :] == 5.0)


def _load_test_hdf5(path: Path) -> np.ndarray:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["tensor"][:])


def test_hermes_batch_optimization_loop_blocks_target_reference_candidate_from_best(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    def fake_library_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {"state_hint": "LOCAL_STRATEGY_LIBRARY_BUILT"}

    def fake_candidate_builder(**kwargs: Any) -> dict[str, Any]:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return {"state_hint": "ALGORITHM_CANDIDATES_COMPILED"}

    def fake_hypothesis_lab(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        submission_dir = output_dir / "sandbox" / "submission_package" / "submission"
        submission_dir.mkdir(parents=True)
        return {
            "artifacts": {
                "best_submission_dir": str(submission_dir),
                "hypothesis_lab_report_path": str(output_dir / "hypothesis_lab_report.json"),
            },
            "sandbox": {
                "reference_audit": {
                    "uses_target_as_reference": True,
                    "submission_validity": "local_validation_only",
                },
                "best_candidate": {
                    "candidate_id": "target_reference_upper_bound",
                    "params": {},
                },
                "verifier": {"decision": "promote_to_strategy_proposal"},
            },
        }

    def fake_judge(**kwargs: Any) -> dict[str, Any]:
        report_path = Path(kwargs["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("{}", encoding="utf-8")
        is_initial = Path(kwargs["submission_dir"]).name == "initial"
        score = 90.0 if is_initial else 150.0
        return {
            "judge_report_path": str(report_path),
            "rules_passed": True,
            "total_score_proxy": score,
            "tasks": {
                "1": {
                    "timing": {"inference_time": 1.0, "train_time": 0.0},
                    "score": {
                        "task_total_score": score,
                        "prediction_accuracy_score": score - 70.0,
                        "segment_score": score - 65.0,
                    },
                }
            },
        }

    monkeypatch.setattr(loop_module, "build_local_strategy_library", fake_library_builder)
    monkeypatch.setattr(
        loop_module,
        "build_local_asset_algorithm_candidates",
        fake_candidate_builder,
    )
    monkeypatch.setattr(loop_module, "run_hypothesis_lab", fake_hypothesis_lab)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()

    report = run_hermes_batch_optimization_loop(
        output_dir=tmp_path / "loop",
        initial_submission_dir=initial_dir,
        max_rounds=1,
        patience=1,
        min_score_improvement=0.0,
        submit_score_threshold=120.0,
        learning_objective="auto",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        candidate_generator="hypothesis_lab",
        judge=fake_judge,
        force=False,
    )

    assert report["best_candidate"]["candidate_id"] == "initial"
    assert report["trajectory"][1]["candidate_id"] == "target_reference_upper_bound"
    assert report["trajectory"][1]["rules_passed"] is False
    assert report["trajectory"][1]["candidate_gate_issues"] == [
        "target_reference_local_validation_only"
    ]


def test_hermes_batch_optimization_loop_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_batch_optimization_loop as command_module
    from typer.testing import CliRunner

    def fake_loop(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "loop.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
            "loop_report_path": str(report_path),
            "round_count": 2,
            "best_candidate": {
                "candidate_id": "candidate_1",
                "score": 96.0,
                "objective": {
                    "selected_objective": "score",
                    "objective_score": 96.0,
                },
                "submission_dir": str(tmp_path / "candidate_1"),
            },
            "parameters": {"learning_objective": "score"},
            "stop_reason": "submit_threshold_met",
            "decision": "submit_best_candidate",
            "next_command": "submit-best-candidate",
        }

    monkeypatch.setattr(command_module, "run_hermes_batch_optimization_loop", fake_loop)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-batch-optimization-loop",
            "--output-dir",
            str(tmp_path / "loop"),
            "--initial-submission-dir",
            str(tmp_path / "initial"),
            "--task1-test-path",
            "task1_test.hdf5",
            "--task1-target-path",
            "task1_val.hdf5",
            "--max-rounds",
            "2",
            "--learning-objective",
            "auto",
            "--unet-checkpoint-path",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_BATCH_OPTIMIZATION_LOOP_RAN"
    assert payload["metrics"]["best_score"] == 96.0
    assert payload["metrics"]["learning_objective"] == "score"
    assert payload["next_hint"] == "submit-best-candidate"
