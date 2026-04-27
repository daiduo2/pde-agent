from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_fno_strategy_growth.py"
    spec = importlib.util.spec_from_file_location("prepare_fno_strategy_growth", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _allowed_commands(run_state: str) -> list[str]:
    commands: list[str] = []
    in_allowed_commands = False
    for line in run_state.splitlines():
        stripped = line.strip()
        if stripped == "allowed_commands:":
            in_allowed_commands = True
            continue
        if not in_allowed_commands:
            continue
        if line and not line.startswith("  "):
            break
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


def test_prepare_fno_strategy_growth_renders_gate_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_fno_strategy_growth"
    fixture_root.mkdir(parents=True)
    for name in [
        "run_state.template.yaml",
        "bootstrap_prompt.md",
        "prompt_template.md",
        "task_spec.json",
    ]:
        source = (
            Path(__file__).resolve().parents[1]
            / "integration_tests"
            / "hermes_fno_strategy_growth"
            / name
        )
        (fixture_root / name).write_text(source.read_text(encoding="utf-8"))
    sweep_summary_path = tmp_path / "fno_evidence_summary.json"
    output_dir = tmp_path / "runs" / "strategy_growth_001"
    sweep_summary_path.write_text(
        """
{
  "state_hint": "FNO_EVIDENCE_SWEEP_COMPLETED",
  "comparison_winners": [
    {"start_index": 0, "winner_score_delta": 1.0}
  ]
}
""",
        encoding="utf-8",
    )

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        fno_sweep_summary_path=sweep_summary_path,
        output_dir=output_dir,
        long_window_threshold=31,
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 5
    assert "convert-fno-evidence-summary" in commands[0]
    assert "record-strategy-proposal" in commands[1]
    assert "build-strategy-rubric" in commands[2]
    assert "evaluate-strategy-promotion" in commands[3]
    assert "append-strategy-ledger" in commands[4]
    assert str(output_dir / "strategy_evidence_summary.json") in run_state
    assert str(output_dir / "strategy_proposal.json") in run_state
    assert str(output_dir / "experiment_rubric.json") in run_state
    assert str(output_dir / "promotion_report.json") in run_state
    assert str(output_dir / "strategy_ledger.jsonl") in run_state
    assert "evaluate-fno-candidate" not in run_state
    assert "execute-fno-finetune-candidate" not in run_state
    assert "compare-fno-candidates" not in run_state


def test_prepare_fno_strategy_growth_rejects_missing_sweep_summary(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_fno_strategy_growth"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")

    with pytest.raises(SystemExit, match="FNO sweep summary does not exist"):
        module.prepare_runtime(
            repo_root=repo_root,
            fno_sweep_summary_path=tmp_path / "missing_summary.json",
            output_dir=tmp_path / "runs" / "strategy_growth_001",
            long_window_threshold=31,
        )

    assert not (fixture_root / "run_state.yaml").exists()


def test_prepare_fno_strategy_growth_rejects_invalid_sweep_summary_schema(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_fno_strategy_growth"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")
    sweep_summary_path = tmp_path / "fno_evidence_summary.json"
    sweep_summary_path.write_text(
        """
{
  "state_hint": "FNO_EVIDENCE_SWEEP_COMPLETED",
  "comparison_winners": [
    {"start_index": 0, "score_delta_finetuned_minus_baseline": 1.0}
  ]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
        match=r"comparison_winners\[0\]\.winner_score_delta must be numeric",
    ):
        module.prepare_runtime(
            repo_root=repo_root,
            fno_sweep_summary_path=sweep_summary_path,
            output_dir=tmp_path / "runs" / "strategy_growth_001",
            long_window_threshold=31,
        )

    assert not (fixture_root / "run_state.yaml").exists()
