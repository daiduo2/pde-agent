from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "prepare_deeponet_strategy_growth.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_deeponet_strategy_growth",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_eval(
    path: Path,
    *,
    state_hint: str = "DEEPONET_CANDIDATE_EVALUATED",
    experiment_id: str = "deeponet_a",
    start_index: int = 0,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": state_hint,
                "inputs": {"forecast_steps": 31},
                "candidate": {
                    "experiment_id": experiment_id,
                    "start_index": start_index,
                    "train_steps": 1,
                    "training_strategy": "branch_trunk_rollout",
                },
                "baseline_comparison": {
                    "baseline_score": 40.0,
                    "score_delta": 1.0,
                    "improved": True,
                },
                "score": {
                    "competition_like_score_short": 41.0,
                    "rel_mse": 0.2,
                    "rmse": 0.3,
                },
                "error_diagnosis": {
                    "dominant_error_pattern": "long_horizon_drift",
                },
                "physics": {"physics_issue": "inconclusive"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _allowed_commands(run_state: str) -> list[str]:
    commands: list[str] = []
    for line in run_state.splitlines():
        stripped = line.strip()
        if stripped.startswith("command: "):
            commands.append(stripped.removeprefix("command: "))
    return commands


def test_prepare_deeponet_strategy_growth_renders_protocol(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_deeponet_strategy_growth"
    fixture_root.mkdir(parents=True)
    source_fixture = (
        Path(__file__).resolve().parents[1]
        / "integration_tests"
        / "hermes_deeponet_strategy_growth"
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
    eval_path = _write_eval(tmp_path / "candidate_0.json")
    eval_path_2 = _write_eval(
        tmp_path / "candidate_1.json",
        experiment_id="deeponet_b",
        start_index=1,
    )
    output_dir = tmp_path / "runs" / "deeponet_strategy_growth_001"

    outputs = module.prepare_runtime(
        repo_root=repo_root,
        evaluation_report_paths=[eval_path, eval_path_2],
        output_dir=output_dir,
        long_window_threshold=31,
    )

    run_state = outputs["run_state"].read_text(encoding="utf-8")
    commands = _allowed_commands(run_state)
    assert "{{" not in run_state
    assert len(commands) == 7
    assert "build-deeponet-evidence-sweep" in commands[0]
    assert "convert-operator-evidence-summary" in commands[1]
    assert "recommend-strategy-next-action" in commands[2]
    assert "record-strategy-proposal" in commands[3]
    assert "build-strategy-rubric" in commands[4]
    assert "evaluate-strategy-promotion" in commands[5]
    assert "append-strategy-ledger" in commands[6]
    assert str(eval_path.resolve()) in commands[0]
    assert str(eval_path_2.resolve()) in commands[0]
    assert str(output_dir / "deeponet_evidence" / "evidence_summary.json") in run_state
    assert str(output_dir / "strategy_decision.json") in run_state
    assert "convert-fno-evidence-summary" not in run_state
    assert "train-deeponet" not in run_state


def test_prepare_deeponet_strategy_growth_rejects_wrong_state_hint(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    repo_root = tmp_path / "repo"
    fixture_root = repo_root / "integration_tests" / "hermes_deeponet_strategy_growth"
    fixture_root.mkdir(parents=True)
    (fixture_root / "run_state.template.yaml").write_text("run_id: {{RUN_ID}}\n")
    eval_path = _write_eval(tmp_path / "candidate.json", state_hint="WRONG")

    with pytest.raises(SystemExit, match="DEEPONET_CANDIDATE_EVALUATED"):
        module.prepare_runtime(
            repo_root=repo_root,
            evaluation_report_paths=[eval_path],
            output_dir=tmp_path / "runs" / "deeponet_strategy_growth_001",
            long_window_threshold=31,
        )

    assert not (fixture_root / "run_state.yaml").exists()
