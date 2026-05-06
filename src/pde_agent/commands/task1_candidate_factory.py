from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.task1_candidate_factory import (
    FACTORY_BENCHMARK_STATE_HINT,
    FAMILY_REPORT_STATE_HINT,
    SANDBOX_STATE_HINT,
    evaluate_candidate_factory_capability,
    generate_task1_candidate_families,
    run_candidate_family_sandbox,
)


def generate_task1_candidate_families_command(
    *,
    benchmark_path: str,
    chain_report_path: str,
    output_path: str,
    prior_registry_path: str | None,
    max_families: int,
    force: bool,
) -> dict[str, object]:
    report = generate_task1_candidate_families(
        benchmark_path=Path(benchmark_path),
        chain_report_path=Path(chain_report_path),
        output_path=Path(output_path),
        prior_registry_path=Path(prior_registry_path)
        if prior_registry_path is not None
        else None,
        max_families=max_families,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="generate-task1-candidate-families",
        state_hint=FAMILY_REPORT_STATE_HINT,
        artifacts={
            "candidate_family_report_json": output_path,
            "family_report_path": output_path,
        },
        metrics={
            "candidate_family_count": report["candidate_family_count"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )


def run_candidate_family_sandbox_command(
    *,
    family_report_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = run_candidate_family_sandbox(
        family_report_path=Path(family_report_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="run-candidate-family-sandbox",
        state_hint=SANDBOX_STATE_HINT,
        artifacts={
            "candidate_family_sandbox_json": output_path,
            "sandbox_report_path": output_path,
        },
        metrics={
            "candidate_family_count": report["candidate_family_count"],
            "planned_count": report["planned_count"],
            "blocked_count": report["blocked_count"],
            "decision": report["decision"],
        },
        next_hint=report["next_command"],
    )


def evaluate_candidate_factory_capability_command(
    *,
    family_report_path: str,
    sandbox_report_path: str,
    output_path: str,
    min_family_count: int,
    min_executable_count: int,
    min_mechanism_count: int,
    force: bool,
) -> dict[str, object]:
    report = evaluate_candidate_factory_capability(
        family_report_path=Path(family_report_path),
        sandbox_report_path=Path(sandbox_report_path),
        output_path=Path(output_path),
        min_family_count=min_family_count,
        min_executable_count=min_executable_count,
        min_mechanism_count=min_mechanism_count,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="evaluate-candidate-factory-capability",
        state_hint=FACTORY_BENCHMARK_STATE_HINT,
        artifacts={
            "candidate_factory_capability_json": output_path,
            "factory_benchmark_path": output_path,
        },
        metrics={
            "decision": report["decision"],
            "family_count": report["candidate_space"]["family_count"],
            "mechanism_count": report["candidate_space"]["mechanism_count"],
            "planned_count": report["sandbox_readiness"]["planned_count"],
        },
        next_hint=report["next_command"],
    )
