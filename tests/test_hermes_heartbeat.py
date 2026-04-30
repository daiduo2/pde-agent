from pathlib import Path
from typing import Any


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
        learning_objective="auto",
        unet_checkpoint_path="checkpoint.pt",
        task1_test_path="task1_test.hdf5",
        task1_target_path="task1_val.hdf5",
        batch_runner=fake_batch_runner,
        memory_builder=fake_memory_builder,
        hypothesis_blueprint_path="blueprints.json",
        force=False,
    )

    assert len(batch_calls) == 2
    assert batch_calls[0]["initial_submission_dir"] == initial_dir
    assert batch_calls[0]["learning_objective"] == "auto"
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
    assert Path(report["heartbeat_report_path"]).exists()


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


def test_hermes_heartbeat_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.run_hermes_heartbeat as command_module
    from typer.testing import CliRunner

    def fake_heartbeat(**_: Any) -> dict[str, Any]:
        report_path = tmp_path / "heartbeat.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_HEARTBEAT_RAN",
            "heartbeat_report_path": str(report_path),
            "batch_count": 1,
            "best_candidate": {"candidate_id": "best", "score": 91.0},
            "best_score": 91.0,
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
        ],
    )

    assert result.exit_code == 0
    payload = __import__("json").loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_HEARTBEAT_RAN"
    assert payload["metrics"]["best_score"] == 91.0
