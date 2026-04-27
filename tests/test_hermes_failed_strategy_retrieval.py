from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_failed_strategy_retrieval.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_failed_strategy_retrieval",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _allowed_commands(run_state: str) -> list[str]:
    commands: list[str] = []
    for line in run_state.splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


def test_prepare_failed_strategy_retrieval_renders_external_proposal_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_failed_strategy_retrieval"
    fixture_root.mkdir(parents=True)
    source_fixture = (
        Path(__file__).resolve().parents[1]
        / "integration_tests"
        / "hermes_failed_strategy_retrieval"
    )
    for name in [
        "run_state.template.yaml",
        "bootstrap_prompt.md",
        "prompt_template.md",
        "task_spec.json",
    ]:
        (fixture_root / name).write_text(
            (source_fixture / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        output_dir=tmp_path / "runs" / "failed_strategy_retrieval_001",
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 3
    assert "recommend-strategy-next-action" in commands[0]
    assert "formalize-external-strategy-candidate" in commands[1]
    assert "record-strategy-proposal" in commands[2]
    assert "--source-type external_retrieval" in commands[2]
    assert str(outputs["failed_evidence_path"].resolve()) in commands[0]
    assert str(outputs["failed_evidence_path"].resolve()) in run_state
    assert str(outputs["output_dir"] / "strategy_decision.json") in run_state
    assert str(outputs["output_dir"] / "retrieved_strategy_candidate.json") in run_state
    assert outputs["failed_evidence_path"].exists()
