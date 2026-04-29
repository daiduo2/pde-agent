import json
from pathlib import Path
from typing import Any


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


def test_hermes_batch_optimization_loop_runtime_config_search_generates_unet_configs(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.core.hermes_batch_optimization_loop import (
        run_hermes_batch_optimization_loop,
    )
    import pde_agent.core.hermes_batch_optimization_loop as loop_module

    predicted_configs: list[tuple[int | None, int | None]] = []

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
        judge=fake_judge,
        force=False,
    )

    assert predicted_configs == [(8, 1), (8, 2)]
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
