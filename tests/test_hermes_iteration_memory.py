import json
from pathlib import Path


def _write_loop_report(path: Path) -> Path:
    payload = {
        "state_hint": "HERMES_BATCH_OPTIMIZATION_LOOP_RAN",
        "stop_reason": "patience_exhausted",
        "decision": "iterate_more",
        "best_candidate": {
            "candidate_id": "postprocess_spectral_smooth_w0.375",
            "score": 88.0,
            "metadata": {
                "source": "postprocess_generator",
                "strategy": "spectral_smooth",
                "blend_weight": 0.375,
            },
        },
        "trajectory": [
            {
                "round_index": 0,
                "candidate_id": "initial",
                "score": 82.0,
                "rules_passed": True,
                "improved": True,
                "metadata": {"source": "initial_submission"},
            },
            {
                "round_index": 1,
                "candidate_id": "postprocess_context_anchor_blend_w0.25",
                "score": 84.0,
                "rules_passed": True,
                "improved": True,
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "context_anchor_blend",
                    "blend_weight": 0.25,
                },
            },
            {
                "round_index": 2,
                "candidate_id": "postprocess_spectral_smooth_w0.5",
                "score": 87.9,
                "rules_passed": True,
                "improved": False,
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "spectral_smooth",
                    "blend_weight": 0.5,
                },
            },
            {
                "round_index": 3,
                "candidate_id": "bad_rule_candidate",
                "score": 0.0,
                "rules_passed": False,
                "improved": False,
                "metadata": {
                    "source": "postprocess_generator",
                    "strategy": "context_anchor_blend",
                    "blend_weight": 1.0,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_hermes_iteration_memory_extracts_avoid_list_and_next_hints(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = _write_loop_report(tmp_path / "loop.json")
    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=None,
        task_ids=None,
        force=False,
    )

    memory_text = Path(report["artifacts"]["memory_markdown_path"]).read_text(
        encoding="utf-8"
    )
    memory_json = json.loads(
        Path(report["artifacts"]["memory_json_path"]).read_text(encoding="utf-8")
    )

    assert report["state_hint"] == "HERMES_ITERATION_MEMORY_BUILT"
    assert memory_json["best_candidate"]["candidate_id"] == (
        "postprocess_spectral_smooth_w0.375"
    )
    assert {
        "strategy": "context_anchor_blend",
        "blend_weight": 1.0,
        "reason": "rule_gate_failed",
    } in memory_json["avoid_repeating"]
    assert {
        "strategy": "spectral_smooth",
        "blend_weight": 0.5,
        "reason": "below_min_effective_improvement",
    } in memory_json["avoid_repeating"]
    assert "Do not repeat" in memory_text
    assert "spectral_smooth" in memory_text
    assert "Try local refinement around blend_weight=0.375" in memory_text


def test_build_hermes_iteration_memory_appends_to_submission_log(tmp_path: Path) -> None:
    from pde_agent.core.hermes_iteration_memory import build_hermes_iteration_memory

    loop_path = _write_loop_report(tmp_path / "loop.json")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    log_path = submission_dir / "task1_logs.log"
    log_path.write_text(
        "Hermes PDE Agent Scientific Log\n\nTrace:\n- prior\n\nExperiment Tracking:\n- prior\n",
        encoding="utf-8",
    )

    report = build_hermes_iteration_memory(
        batch_loop_report_path=loop_path,
        output_path=tmp_path / "memory.md",
        memory_json_path=tmp_path / "memory.json",
        submission_dir=submission_dir,
        task_ids=[1],
        force=False,
    )

    text = log_path.read_text(encoding="utf-8")
    assert report["metrics"]["updated_log_count"] == 1
    assert "Hermes Iteration Memory" in text
    assert "Do not repeat" in text
    assert "postprocess_spectral_smooth_w0.375" in text


def test_build_hermes_iteration_memory_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.build_hermes_iteration_memory as command_module
    from typer.testing import CliRunner

    loop_path = _write_loop_report(tmp_path / "loop.json")

    def fake_builder(**_: object) -> dict[str, object]:
        memory_path = tmp_path / "memory.md"
        json_path = tmp_path / "memory.json"
        memory_path.write_text("Hermes Iteration Memory\n", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_ITERATION_MEMORY_BUILT",
            "artifacts": {
                "memory_markdown_path": str(memory_path),
                "memory_json_path": str(json_path),
                "updated_log_paths": [],
            },
            "metrics": {
                "trajectory_count": 4,
                "avoid_count": 2,
                "updated_log_count": 0,
            },
            "next_command": "run-hermes-batch-optimization-loop",
        }

    monkeypatch.setattr(command_module, "build_hermes_iteration_memory", fake_builder)

    result = CliRunner().invoke(
        app,
        [
            "build-hermes-iteration-memory",
            "--batch-loop-report-path",
            str(loop_path),
            "--output-path",
            str(tmp_path / "memory.md"),
            "--memory-json-path",
            str(tmp_path / "memory.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_ITERATION_MEMORY_BUILT"
    assert payload["metrics"]["avoid_count"] == 2
