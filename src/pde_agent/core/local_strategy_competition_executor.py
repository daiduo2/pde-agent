"""Competition-pipeline executors for Hermes local strategy loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pde_agent.core.competition_task_pipeline import run_competition_task_pipeline


CompetitionExecutor = Callable[[dict[str, Any], Path], Path]


class LocalStrategyCompetitionExecutorError(RuntimeError):
    pass


def build_fno_competition_executor(
    *,
    task_id: int,
    test_path: str | Path,
    checkpoint_path: str | Path,
    submission_id: str,
    target_path: str | Path | None,
    dataset_key: str,
    force: bool,
) -> CompetitionExecutor:
    if task_id not in {1, 2}:
        raise LocalStrategyCompetitionExecutorError("task_id must be 1 or 2")
    test_file = Path(test_path)
    checkpoint_file = Path(checkpoint_path)
    target_file = Path(target_path) if target_path is not None else None

    def executor(claim: dict[str, Any], workspace_dir: Path) -> Path:
        selected = claim.get("selected_experiment")
        if not isinstance(selected, dict):
            raise LocalStrategyCompetitionExecutorError(
                "claim must include selected_experiment"
            )
        next_command = selected.get("next_command")
        if next_command != "predict-fno-competition":
            raise LocalStrategyCompetitionExecutorError(
                "FNO competition executor can only execute predict-fno-competition "
                f"claims, got {next_command!r}"
            )
        report = run_competition_task_pipeline(
            task_id=task_id,
            model_family="fno",
            test_path=test_file,
            checkpoint_path=checkpoint_file,
            output_dir=workspace_dir / "competition_pipeline",
            submission_id=submission_id,
            target_path=target_file,
            dataset_key=dataset_key,
            force=force,
        )
        artifacts = report["artifacts"]
        score_path = artifacts.get("score_json_path")
        if isinstance(score_path, str):
            return Path(score_path)
        pipeline_path = artifacts.get("pipeline_report_path")
        if isinstance(pipeline_path, str):
            return Path(pipeline_path)
        raise LocalStrategyCompetitionExecutorError(
            "competition pipeline did not produce score or pipeline report path"
        )

    return executor
