"""Audit the local strategy loop with deterministic smoke scenarios."""


import json
from pathlib import Path
from typing import Any

from pde_agent.core.local_strategy_leaderboard import record_local_strategy_result
from pde_agent.core.local_strategy_library import (
    build_local_strategy_library,
    lookup_local_strategies,
)
from pde_agent.core.local_strategy_scheduler import (
    claim_next_scheduled_experiment,
    schedule_local_strategy_evaluation,
)


AUDIT_STATE_HINT = "LOCAL_STRATEGY_LOOP_AUDITED"


class LocalStrategyLoopAuditError(RuntimeError):
    pass


def audit_local_strategy_loop(
    *,
    output_dir: str | Path,
    report_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_file = Path(report_path)
    if report_file.exists() and not force:
        raise LocalStrategyLoopAuditError(f"audit report already exists: {report_file}")
    root.mkdir(parents=True, exist_ok=True)
    scenarios = [
        _positive_acceptance(root / "positive_acceptance", force=force),
        _negative_continue(root / "negative_continue", force=force),
        _failure_exhaustion(root / "failure_exhaustion", force=force),
        _competition_like_ingestion(root / "competition_like_ingestion", force=force),
    ]
    passed = all(scenario["passed"] for scenario in scenarios)
    summary = {
        "accepted_count": sum(
            scenario["observed_decision"] == "accept_best_strategy"
            for scenario in scenarios
        ),
        "continued_count": sum(
            scenario["observed_decision"] == "continue_scheduled_experiments"
            for scenario in scenarios
        ),
        "exhausted_count": sum(
            scenario["observed_decision"] == "local_strategy_queue_exhausted"
            for scenario in scenarios
        ),
    }
    report = {
        "schema_version": 1,
        "state_hint": AUDIT_STATE_HINT,
        "audit_report_path": str(report_file),
        "output_dir": str(root),
        "decision": "passed" if passed else "failed",
        "scenario_count": len(scenarios),
        "summary": summary,
        "scenarios": scenarios,
        "next_command": "use-local-strategy-loop",
    }
    _write_json(report_file, report)
    return report


def _positive_acceptance(root: Path, *, force: bool) -> dict[str, Any]:
    paths = _prepare_base(root, max_results=2, max_experiments=2, force=force)
    result_path = root / "positive_result.json"
    _write_json(
        result_path,
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "score": 76.0,
            "baseline_score": 72.0,
            "inference_time_seconds": 50.0,
        },
    )
    leaderboard = record_local_strategy_result(
        schedule_path=paths["schedule_path"],
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=result_path,
        leaderboard_path=root / "leaderboard.json",
        result_status="completed",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    return _scenario(
        scenario_id="positive_acceptance",
        expected_decision="accept_best_strategy",
        expected_next_command="record-strategy-proposal",
        leaderboard=leaderboard,
        artifact_paths={**paths, "result_path": str(result_path)},
    )


def _negative_continue(root: Path, *, force: bool) -> dict[str, Any]:
    paths = _prepare_base(root, max_results=2, max_experiments=2, force=force)
    result_path = root / "negative_result.json"
    _write_json(
        result_path,
        {
            "state_hint": "LOCAL_EXPERIMENT_RESULT",
            "score": 70.0,
            "baseline_score": 72.0,
            "inference_time_seconds": 50.0,
        },
    )
    leaderboard = record_local_strategy_result(
        schedule_path=paths["schedule_path"],
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=result_path,
        leaderboard_path=root / "leaderboard.json",
        result_status="completed",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    claim = claim_next_scheduled_experiment(
        schedule_path=paths["schedule_path"],
        completed_experiment_ids=leaderboard["completed_experiment_ids"],
        output_path=root / "second_claim.json",
    )
    return _scenario(
        scenario_id="negative_continue",
        expected_decision="continue_scheduled_experiments",
        expected_next_command="claim-next-scheduled-experiment",
        leaderboard=leaderboard,
        artifact_paths={
            **paths,
            "result_path": str(result_path),
            "second_claim_path": claim["claim_path"],
        },
        extra_checks={
            "second_claim_strategy_id": claim["selected_experiment"]["strategy_id"],
            "second_claim_next_command": claim["next_command"],
        },
    )


def _failure_exhaustion(root: Path, *, force: bool) -> dict[str, Any]:
    paths = _prepare_base(root, max_results=2, max_experiments=2, force=force)
    first_result = root / "failed_first.json"
    second_result = root / "failed_second.json"
    _write_json(
        first_result,
        {"state_hint": "LOCAL_EXPERIMENT_RESULT", "error": "checkpoint incompatible"},
    )
    _write_json(
        second_result,
        {"state_hint": "LOCAL_EXPERIMENT_RESULT", "error": "training budget exhausted"},
    )
    record_local_strategy_result(
        schedule_path=paths["schedule_path"],
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=first_result,
        leaderboard_path=root / "leaderboard.json",
        result_status="failed",
        failure_reason="checkpoint incompatible",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    leaderboard = record_local_strategy_result(
        schedule_path=paths["schedule_path"],
        experiment_id="fno_rollout_consistency_finetune_v0_002",
        result_summary_path=second_result,
        leaderboard_path=root / "leaderboard.json",
        result_status="failed",
        failure_reason="training budget exhausted",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    return _scenario(
        scenario_id="failure_exhaustion",
        expected_decision="local_strategy_queue_exhausted",
        expected_next_command="build-research-chronicle",
        leaderboard=leaderboard,
        artifact_paths={
            **paths,
            "first_result_path": str(first_result),
            "second_result_path": str(second_result),
        },
    )


def _competition_like_ingestion(root: Path, *, force: bool) -> dict[str, Any]:
    paths = _prepare_base(root, max_results=1, max_experiments=1, force=force)
    result_path = root / "competition_like_result.json"
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "state_hint": "COMPETITION_PREDICTION_SCORED",
            "segment_score": 82.5,
            "baseline_score": 80.0,
            "inference_time": 59.7,
        },
    )
    leaderboard = record_local_strategy_result(
        schedule_path=paths["schedule_path"],
        experiment_id="fno_spectral_viscosity_filter_v0_001",
        result_summary_path=result_path,
        leaderboard_path=root / "leaderboard.json",
        result_status="completed",
        min_score_delta=1.0,
        max_inference_seconds=120.0,
    )
    return _scenario(
        scenario_id="competition_like_ingestion",
        expected_decision="accept_best_strategy",
        expected_next_command="record-strategy-proposal",
        leaderboard=leaderboard,
        artifact_paths={**paths, "result_path": str(result_path)},
    )


def _prepare_base(
    root: Path,
    *,
    max_results: int,
    max_experiments: int,
    force: bool,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    library_path = root / "library.json"
    lookup_path = root / "lookup.json"
    schedule_path = root / "schedule.json"
    build_local_strategy_library(output_path=library_path, force=force)
    lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="FNO",
        failure_modes=["long_horizon_drift", "spectral_dissipation_mismatch"],
        max_results=max_results,
        output_path=lookup_path,
    )
    schedule_local_strategy_evaluation(
        lookup_report_path=lookup_path,
        output_path=schedule_path,
        workspace_root=root / "workspace",
        max_experiments=max_experiments,
        max_train_seconds=3600.0,
        max_inference_seconds=120.0,
        force=force,
    )
    return {
        "library_path": str(library_path),
        "lookup_path": str(lookup_path),
        "schedule_path": str(schedule_path),
    }


def _scenario(
    *,
    scenario_id: str,
    expected_decision: str,
    expected_next_command: str,
    leaderboard: dict[str, Any],
    artifact_paths: dict[str, str],
    extra_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = leaderboard["decision"]
    observed_decision = str(decision["decision"])
    observed_next_command = str(decision["next_command"])
    passed = (
        observed_decision == expected_decision
        and observed_next_command == expected_next_command
    )
    return {
        "scenario_id": scenario_id,
        "passed": passed,
        "observed_decision": observed_decision,
        "observed_next_command": observed_next_command,
        "expected_decision": expected_decision,
        "expected_next_command": expected_next_command,
        "leaderboard_path": leaderboard["leaderboard_path"],
        "result_count": leaderboard["result_count"],
        "artifact_paths": artifact_paths,
        "extra_checks": extra_checks or {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
